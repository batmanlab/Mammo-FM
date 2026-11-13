"""Run with python -m mammo_sae --help."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import analysis, data, training, experiments, localization
from .core import SpatialSAE, intervene, shapley, shapley_all_background


def json_file(path):
    return json.loads(Path(path).read_text()) if path else {}


def serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    return value


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Model-independent spatial SAE workflows")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser('prepare-manifest', help='Map dataset-specific columns to explicit image keys')
    prepare.add_argument('--input', required=True)
    prepare.add_argument('--columns', required=True)
    prepare.add_argument('--dataset')
    prepare.add_argument('--dataset-column', default='dataset')
    prepare.add_argument('--bu-image-root', help='Explicitly remap historical BU PNG paths to the shared image root')
    prepare.add_argument('--output', required=True)
    historical = sub.add_parser('import-metadata', help='Key historical patch metadata to an image manifest')
    historical.add_argument('--input', required=True)
    historical.add_argument('--manifest', required=True)
    historical.add_argument('--output', required=True)
    select = sub.add_parser("select", help="Select finding-balanced images from a labeled manifest")
    select.add_argument('--manifest', required=True)
    select.add_argument('--findings', nargs='+', required=True)
    select.add_argument('--per-finding', type=int, default=117)
    select.add_argument('--seed', type=int, default=42)
    select.add_argument('--output', required=True)
    associate = sub.add_parser('associations', help='Finding associations and patient-level tests')
    associate.add_argument('--summary', required=True)
    associate.add_argument('--labels', required=True)
    associate.add_argument('--findings', nargs='+', required=True)
    associate.add_argument('--pure', action='store_true')
    associate.add_argument('--output', required=True)
    boxes = sub.add_parser('evaluate-localization', help='Evaluate boxes on an explicit image cohort')
    boxes.add_argument('--detections', required=True)
    boxes.add_argument('--annotations', required=True)
    boxes.add_argument('--images', required=True)
    boxes.add_argument('--iou', type=float, default=0.5)
    boxes.add_argument('--output', required=True)
    extract = sub.add_parser("extract", help="Extract spatial features via a configured adapter")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--factory", required=True)
    extract.add_argument("--adapter-config", required=True)
    extract.add_argument("--output", required=True)
    split = sub.add_parser("split", help="Split features by patient (or an explicit grouping column)")
    split.add_argument("--features", required=True)
    split.add_argument("--metadata", required=True)
    split.add_argument("--group", default="patient_id")
    split.add_argument("--fraction", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--output", required=True)
    train = sub.add_parser("train", help="Train a Matryoshka SAE")
    train.add_argument("--train", required=True)
    train.add_argument("--val", required=True)
    train.add_argument("--model-config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--device", default="cpu")
    train.add_argument("--mean-center", action="store_true")
    train.add_argument("--target-norm", type=float)
    original = sub.add_parser("train-original", help="Original AdamW/linear-decay recipe with decoder constraints")
    for name in ["train", "val", "model-config", "output"]:
        original.add_argument("--" + name, required=True)
    original.add_argument("--epochs", type=int, default=30)
    original.add_argument("--batch-size", type=int, default=4096)
    original.add_argument("--device", default="cpu")
    for name in ["evaluate", "neurons", "heatmaps", "localize", "generalizability", "intervene", "shapley", "text"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--features", required=True)
        cmd.add_argument("--metadata", required=name not in ["evaluate", "text"])
        cmd.add_argument("--checkpoint", required=True)
        cmd.add_argument("--model-config", help="Required for historical checkpoints without model_config")
        cmd.add_argument("--preprocessing", choices=["checkpoint", "identity"], default="checkpoint")
        cmd.add_argument("--level", type=int, default=-1, help="Index in nesting_list; -1 is the largest K")
        cmd.add_argument("--batch-size", type=int, default=256)
        cmd.add_argument("--device", default="cpu")
        cmd.add_argument("--output", required=True)
        if name in ["heatmaps", "intervene", "shapley", "text"]:
            cmd.add_argument("--neurons", type=int, nargs="+", required=True)
        if name == "neurons":
            cmd.add_argument("--top-k", type=int, default=30)
        if name == 'localize':
            cmd.add_argument('--finding-config', required=True)
        if name == 'generalizability':
            cmd.add_argument('--labels', required=True)
            cmd.add_argument('--label', required=True)
            cmd.add_argument('--regularization', nargs='+', type=float, default=[0.01, 0.1, 1.0])
            cmd.add_argument('--seed', type=int, default=42)
        if name in ["intervene", "shapley", "text"]:
            cmd.add_argument("--factory", required=True)
            cmd.add_argument("--adapter-config", required=True)
        if name in ["intervene", "shapley"]:
            cmd.add_argument("--patient-id", required=True)
            cmd.add_argument("--exam-id", required=True)
        if name == "shapley":
            cmd.add_argument("--samples", type=int, default=100)
            cmd.add_argument("--background", choices=["selected", "all"], default="selected",
                             help="Coalition players: selected neurons or all SAE latents")
            cmd.add_argument("--seed", type=int, default=42)
        if name == "text":
            cmd.add_argument("--sentences", required=True, help="JSON list of sentences; features must describe one image")
            cmd.add_argument('--image-id', help='Select one image from a spatial feature cache')
            cmd.add_argument('--patient-id')
            cmd.add_argument('--exam-id')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a new path")
    if Path(str(output) + ".command.json").exists():
        parser.error("Command manifest already exists; choose a new output path")
    if args.command == "text" and (output.suffix != ".json" or output.with_suffix(".html").exists()):
        parser.error("Text output must be a new .json path with no existing companion .html")
    if getattr(args, "batch_size", 1) < 1:
        parser.error("batch-size must be positive")
    run = vars(args).copy()
    if getattr(args, "adapter_config", None):
        run["adapter_configuration"] = json_file(args.adapter_config)
        if run["adapter_configuration"].get("checkpoint"):
            run['adapter_checkpoint_sha256'] = file_hash(run['adapter_configuration']['checkpoint'])
    if getattr(args, "checkpoint", None):
        run["checkpoint_sha256"] = file_hash(args.checkpoint)
    for attribute in ['model_config', 'finding_config', 'columns']:
        if getattr(args, attribute, None):
            run[attribute + '_contents'] = json_file(getattr(args, attribute))
    if args.command in ['prepare-manifest', 'import-metadata']:
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == 'prepare-manifest':
            data.prepare_manifest(args.input, json_file(args.columns), output, args.dataset,
                                  args.dataset_column, args.bu_image_root)
        else:
            data.import_metadata(args.input, args.manifest, output)
    elif args.command == 'select':
        experiments.select_images(args.manifest, args.findings, args.per_finding, output, args.seed)
    elif args.command == 'associations':
        experiments.associations(args.summary, args.labels, args.findings, output, args.pure)
    elif args.command == 'evaluate-localization':
        output.parent.mkdir(parents=True, exist_ok=True)
        localization.evaluate(args.detections, args.annotations, args.images, output, args.iou)
    elif args.command == "extract":
        adapter = data.factory(args.factory, json_file(args.adapter_config))
        data.extract(args.manifest, adapter, output)
    elif args.command == "split":
        data.split(args.features, args.metadata, output, args.group, args.fraction, args.seed)
    elif args.command == "train-original":
        from .original_training import train as train_original
        train_original(args.train, args.val, json_file(args.model_config), output,
                       args.epochs, args.batch_size, args.device)
    elif args.command == "train":
        training.train(args.train, args.val, output, json_file(args.model_config), args.epochs,
                       args.batch_size, args.lr, args.seed, args.device, args.mean_center, args.target_norm)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        values, frame = data.read_features(args.features, args.metadata)
        sae = SpatialSAE.load(args.checkpoint, args.model_config, args.device, args.preprocessing)
        if args.command == "evaluate":
            loader = torch.utils.data.DataLoader(training.Features(values), batch_size=args.batch_size)
            errors = training.evaluate(sae, loader)
            output.write_text(json.dumps(dict(nesting_list=sae.model.nesting_list, mse=errors,
                                             feature_units="original encoder units"), indent=2))
        elif args.command == "neurons":
            analysis.neuron_summary(sae, values, frame, output, args.top_k, args.batch_size, args.level)
        elif args.command == "heatmaps":
            analysis.heatmaps(sae, values, frame, args.neurons, output, args.level, args.batch_size)
        elif args.command == 'localize':
            localization.localize(sae, values, frame, json_file(args.finding_config), output, args.level, args.batch_size)
        elif args.command == 'generalizability':
            experiments.generalizability(sae, values, frame, args.labels, args.label, output,
                                          args.level, args.batch_size, args.regularization, args.seed)
        elif args.command in ["intervene", "shapley"]:
            adapter = data.factory(args.factory, json_file(args.adapter_config))
            mask = (frame.patient_id == args.patient_id) & (frame.exam_id == args.exam_id)
            if not mask.any():
                raise ValueError("Requested examination is absent from metadata")
            subset = frame.loc[mask].reset_index(drop=True)
            predictor = adapter.bind(subset)
            patches = np.asarray(values[mask.to_numpy()])
            if args.command == "intervene":
                result = intervene(sae, patches, predictor, args.neurons, args.level)
            else:
                estimate = shapley_all_background if args.background == "all" else shapley
                result = estimate(sae, patches, predictor, args.neurons, args.samples, args.seed, args.level)
            result = dict(patient_id=args.patient_id, exam_id=args.exam_id,
                          values=serializable(result))
            output.write_text(json.dumps(result, indent=2))
        elif args.command == "text":
            selectors = [args.patient_id, args.exam_id, args.image_id]
            if any(selectors):
                if not all(selectors) or frame is None:
                    raise ValueError('Image selection requires metadata and patient, exam and image IDs')
                mask = (frame.patient_id == args.patient_id) & (frame.exam_id == args.exam_id) & (frame.image_id == args.image_id)
                values, frame = values[mask.to_numpy()], frame.loc[mask]
            if frame is not None and (not set(['patient_id','exam_id','image_id']).issubset(frame) or
                    len(frame[['patient_id','exam_id','image_id']].drop_duplicates()) != 1):
                raise ValueError("Text alignment features must describe exactly one image")
            sentences = json_file(args.sentences)
            if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
                raise ValueError("sentences must be a JSON list of strings")
            adapter = data.factory(args.factory, json_file(args.adapter_config))
            result = analysis.sentence_ablation(sae, np.asarray(values).copy(), sentences,
                                                adapter, args.neurons, args.level)
            if frame is not None:
                identity = frame.iloc[0][['patient_id','exam_id','image_id']].to_dict()
                result = [dict(identity, **row) for row in result]
            output.write_text(json.dumps(result, indent=2))
            output.with_suffix(".html").write_text(analysis.render_sentences(result))
    provenance = output / "command.json" if output.is_dir() else Path(str(output) + ".command.json")
    provenance.write_text(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
