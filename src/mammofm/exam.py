"""Four-view exam orchestration for risk probabilities and grounded reports."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import uuid

from ._assets import AssetStore
from ._registry import BU_RISK_ENCODER, DEFAULT_BU_RISK_MODEL, RISK_CHECKPOINTS
from ._risk import VIEW_ORDER


OUTPUT_FILENAMES = {
    "report": "report.json",
    "mayo_risk": "risk_mayo.json",
    "bu_risk": "risk_bu.json",
}

BU_RISK_MODEL_LABELS = {
    "s1s2": "Stage 1 MIRAI KD then Stage 2 BU outcomes",
    "s2_only": "Stage 2 BU outcomes only",
}

RISK_COMMENTS = {
    "mayo": (
        "Image encoder is Mammo-FM multi-institution checkpoint. "
        "Risk heads are finetuned with Mayo risks. "
        "Risk scores are not calibrated. These are probabilities."
    ),
    "bu": (
        "Image encoder is Mammo-FM multi-institution checkpoint. "
        "Risk heads are finetuned with BMC risks. "
        "Risk scores are not calibrated. These are probabilities."
    ),
}


def _device(requested=None):
    import torch

    selected = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    return selected


def _release_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _is_png(path):
    with Path(path).expanduser().open("rb") as stream:
        return stream.read(8) == b"\x89PNG\r\n\x1a\n"


def _view_paths(value):
    if not isinstance(value, dict):
        raise TypeError(f"view_paths must map exactly {VIEW_ORDER} to image paths")
    normalized = {}
    for name, path in value.items():
        if not isinstance(name, str) or name.upper() not in VIEW_ORDER:
            raise ValueError(f"View names must be exactly {VIEW_ORDER} (case-insensitive)")
        name = name.upper()
        if name in normalized:
            raise ValueError(f"Duplicate view {name}")
        normalized[name] = path
    if set(normalized) != set(VIEW_ORDER):
        raise ValueError(f"Four inputs are required: {VIEW_ORDER}")
    return {view: normalized[view] for view in VIEW_ORDER}


@contextmanager
def _prepared_views(view_paths):
    """Convert DICOMs to private temporary PNGs, then prepare one shared tensor."""
    import torch
    from ._inputs import convert_dicom_to_png, inference_tensor, load_image

    if not isinstance(view_paths, dict) or tuple(view_paths) != VIEW_ORDER:
        raise ValueError(f"view_paths must be ordered exactly {VIEW_ORDER}")
    with TemporaryDirectory(prefix="mammofm-exam-") as temporary:
        temporary = Path(temporary)
        tensors = []
        metadata = {}
        for view in VIEW_ORDER:
            source = Path(view_paths[view]).expanduser()
            if not source.is_file():
                raise FileNotFoundError(f"{view} input is not an existing file")
            if _is_png(source):
                image, details = load_image(source, dataset="general")
            else:
                converted = temporary / f"{view}.png"
                details = convert_dicom_to_png(source, converted, dataset="general")
                image, png_details = load_image(converted, dataset="general")
                if png_details["input_format"] != "png":
                    raise RuntimeError("Unified DICOM conversion did not produce a PNG")
            tensors.append(inference_tensor(image, branch="features", dataset="general").squeeze(0))
            metadata[view] = details
        images = torch.stack(tensors, dim=0).unsqueeze(0)
        if tuple(images.shape[:3]) != (1, 4, 3) or not torch.isfinite(images).all():
            raise RuntimeError("Prepared exam tensor is invalid")
        prepared_paths = {
            view: (temporary / f"{view}.png") if metadata[view]["input_format"] == "dicom"
            else Path(view_paths[view]).expanduser()
            for view in VIEW_ORDER
        }
        yield images, metadata, prepared_paths


def _risk_output(model_name, raw_probabilities):
    raw = [float(value) for value in raw_probabilities]
    if len(raw) != 5 or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in raw):
        raise ValueError("raw_probabilities must contain five finite values in [0, 1]")
    if model_name not in RISK_COMMENTS:
        raise ValueError("model_name must be 'mayo' or 'bu'")
    output = {f"{year}yr": value for year, value in enumerate(raw, start=1)}
    output["comment"] = RISK_COMMENTS[model_name]
    return output


def _run_one_risk(images, store, asset, device, label, *, encoder_asset=None):
    from ._risk import load_risk_model, predict_risks

    checkpoint = store.resolve(asset)
    encoder_checkpoint = store.resolve(encoder_asset) if encoder_asset is not None else None
    model = load_risk_model(
        checkpoint, device=device, encoder_checkpoint=encoder_checkpoint)
    try:
        raw = predict_risks(model, images)[0].detach().float().cpu().tolist()
    finally:
        del model
        _release_cuda()
    return _risk_output(label, raw)


def _run_risks(images, store, device, bu_model, progress):
    progress(
        "Estimating Mayo-trained risk "
        "(Stage 1 MIRAI KD then Stage 2 BU outcomes)"
    )
    mayo_asset = RISK_CHECKPOINTS["mayo"]
    mayo = _run_one_risk(images, store, mayo_asset, device, "mayo")
    progress(f"Estimating BMC-trained risk ({BU_RISK_MODEL_LABELS[bu_model]})")
    bu_asset = RISK_CHECKPOINTS["bu"][bu_model]
    bu = _run_one_risk(
        images, store, bu_asset, device, "bu", encoder_asset=BU_RISK_ENCODER)
    return mayo, bu


def _run_report(prepared_paths, store, device, input_metadata, *, base_model_path,
                seed, preliminary_max_new_tokens, grounding_max_new_tokens,
                progress):
    import re
    from .api import MammoFM

    progress("Generating classifier-grounded Stage-3 report")
    client = MammoFM(
        device=str(device), cache_dir=store.cache_dir,
        model_dir=store.model_dir, local_files_only=store.local_files_only,
    )
    generated = client.report_gen(
        {view.lower(): prepared_paths[view] for view in VIEW_ORDER},
        base_model_path=base_model_path,
        seed=seed,
        max_new_tokens=preliminary_max_new_tokens,
        grounding_max_new_tokens=grounding_max_new_tokens,
    )
    final_report = generated["grounded_report"]
    if not re.search(r"(?im)^\s*(?:\d+[.)]\s*)?findings\s*:?", final_report):
        raise RuntimeError("Stage 3 output is missing a Findings section")
    if not re.search(r"(?im)^\s*(?:\d+[.)]\s*)?impression\s*:?", final_report):
        raise RuntimeError("Stage 3 output is missing an Impression section")
    report_metadata = generated["findings"]["metadata"]
    return {
        "schema_version": "1.0",
        "output_type": "classifier_grounded_mammography_report",
        "stage": 3,
        "grounded_report": final_report,
        "models": {
            "report_encoder": report_metadata["visual_encoder"],
            "stage2": report_metadata["stage2"],
            "stage3_base": report_metadata["base_language_model"],
            "grounding_classifiers": {
                finding: {
                    "checkpoint": checkpoint,
                }
                for finding, checkpoint in report_metadata["checkpoints"].items()
            },
        },
        "metadata": {
            "research_only": True,
            "prediction_level": "four_view_exam",
            "view_order": list(VIEW_ORDER),
            "inputs": input_metadata,
            "preliminary_report_exposed": False,
            "grounding_predictions_exposed": False,
            "generation": report_metadata["generation"],
        },
    }


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_exam(view_paths, output_dir, *, bu_model=DEFAULT_BU_RISK_MODEL,
             device=None, cache_dir=None,
             model_dir=None, local_files_only=False, base_model_path=None,
             report_seed=0, preliminary_max_new_tokens=512,
             grounding_max_new_tokens=320, only="all", overwrite=False,
             progress=None):
    """Run a four-view exam and atomically publish the requested JSON outputs.

    ``only`` may be ``all``, ``risk`` or ``report``.  The default creates
    ``risk_mayo.json``, ``risk_bu.json`` and ``report.json``.  All expensive
    work completes before any output is published, so a failed report cannot
    leave risk files that look like a successful complete run.
    """
    if only not in {"all", "risk", "report"}:
        raise ValueError("only must be 'all', 'risk' or 'report'")
    if bu_model not in BU_RISK_MODEL_LABELS:
        raise ValueError("bu_model must be 's1s2' or 's2_only'")
    if not isinstance(report_seed, int) or isinstance(report_seed, bool) or not 0 <= report_seed < 2**32:
        raise ValueError("report_seed must be an integer in [0, 2**32)")
    for name, value in (
        ("preliminary_max_new_tokens", preliminary_max_new_tokens),
        ("grounding_max_new_tokens", grounding_max_new_tokens),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 4096:
            raise ValueError(f"{name} must be an integer between 1 and 4096")
    ordered_paths = _view_paths(view_paths)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = (
        ("mayo_risk", "bu_risk", "report") if only == "all"
        else (("mayo_risk", "bu_risk") if only == "risk" else ("report",))
    )
    destinations = {key: output_dir / OUTPUT_FILENAMES[key] for key in requested}
    existing = [path.name for path in destinations.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(sorted(existing)))
    progress = progress or (lambda _message: None)
    selected_device = _device(device)
    store = AssetStore(
        cache_dir=cache_dir,
        model_dir=model_dir,
        local_files_only=local_files_only,
    )
    results = {}
    progress("Preparing four views")
    with _prepared_views(ordered_paths) as (images, input_metadata, prepared_paths):
        if only in {"all", "risk"}:
            mayo, bu = _run_risks(
                images, store, selected_device, bu_model, progress)
            results["mayo_risk"] = mayo
            results["bu_risk"] = bu
        if only in {"all", "report"}:
            results["report"] = _run_report(
                prepared_paths,
                store,
                selected_device,
                input_metadata,
                base_model_path=base_model_path,
                seed=report_seed,
                preliminary_max_new_tokens=preliminary_max_new_tokens,
                grounding_max_new_tokens=grounding_max_new_tokens,
                progress=progress,
            )
    for key in requested:
        _write_json(destinations[key], results[key])
    progress("Complete")
    return results


__all__ = ["OUTPUT_FILENAMES", "run_exam"]
