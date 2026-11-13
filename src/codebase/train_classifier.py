import warnings

import torch

from Classifiers.experiments import do_experiments
from utils import get_Paths, seed_all

warnings.filterwarnings("ignore")
import argparse
import os
import pickle


def config():
    parser = argparse.ArgumentParser()
    eval_root = os.environ.get("MAMMO_FM_EVAL_DIR")
    parser.add_argument('--tensorboard-path', metavar='DIR',
                        default=os.path.join(eval_root, "tensorboard") if eval_root else None,
                        help='path to tensorboard logs outside the Mammo-FM repository')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default=os.path.join(eval_root, "checkpoints") if eval_root else None,
                        help='path to downstream checkpoints outside the Mammo-FM repository')
    parser.add_argument('--output_path', metavar='DIR',
                        default=os.path.join(eval_root, "outputs") if eval_root else None,
                        help='path to downstream outputs outside the Mammo-FM repository')
    parser.add_argument(
        "--data-dir",
        default="/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset",
        type=str, help="Path to data file"
    )
    parser.add_argument(
        "--img-dir", default="RSNA_Cancer_Detection/train_images_png", type=str, help="Path to image file"
    )

    parser.add_argument("--clip_chk_pt_path", default=None, type=str, help="Path to a Mammo-FM checkpoint")
    parser.add_argument("--csv-file", default="RSNA_Cancer_Detection/final_rsna.csv", type=str,
                        help="data csv file")
    parser.add_argument("--dataset", default="RSNA", type=str, help="Dataset name? (RSNA or VinDr)")
    parser.add_argument("--data_frac", default=1.0, type=str, help="Fraction of data to be used for training")
    parser.add_argument(
        "--arch", default="breast_clip_det_b5_period_n_ft", type=str,
        help="For b5 classification, [breast_clip_det_b5_period_n_lp for linear probe and  reast_clip_det_b5_period_n_ft for finetuning]. "
             "For b2 classification, [breast_clip_det_b2_period_n_lp for linear probe and  reast_clip_det_b2_period_n_ft for finetuning].")
    parser.add_argument("--label", default="cancer", type=str,
                        help="cancer for RSNA or Mass, Suspicious_Calcification, density for VinDr")
    parser.add_argument("--detector-threshold", default=0.1, type=float)
    parser.add_argument("--swin_encoder", default="microsoft/swin-tiny-patch4-window7-224", type=str)
    parser.add_argument("--pretrained_swin_encoder", default="y", type=str)
    parser.add_argument("--swin_model_type", default="y", type=str)
    parser.add_argument("--VER", default="084", type=str)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--num_cycles", default=0.5, type=float)
    parser.add_argument("--alpha", default=10, type=float)
    parser.add_argument("--sigma", default=15, type=float)
    parser.add_argument("--p", default=1.0, type=float)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument("--focal-alpha", default=0.6, type=float)
    parser.add_argument("--focal-gamma", default=2.0, type=float)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--density-metadata", default=None,
                        help="Density-only LUMINA CSV keyed by Cancer cohort and patient_id, with density=0..3")
    parser.add_argument("--n_folds", default=4, type=int)
    parser.add_argument("--start-fold", default=0, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--warmup-epochs", default=1, type=float)
    parser.add_argument("--img-size", nargs=2, type=int, default=[1520, 912],
                        metavar=("HEIGHT", "WIDTH"), help="Required image geometry: 1520 high x 912 wide")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)
    parser.add_argument("--print-freq", default=5000, type=int)
    parser.add_argument("--log-freq", default=1000, type=int)
    parser.add_argument("--running-interactive", default='n', type=str)
    parser.add_argument("--inference-mode", default='n', type=str)
    parser.add_argument('--model-type', default="Classifier", type=str)
    parser.add_argument("--weighted-BCE", default='n', type=str)
    parser.add_argument("--balanced-dataloader", default='n', type=str)

    args = parser.parse_args()
    if args.img_size != [1520, 912]:
        parser.error("All classification datasets require --img-size 1520 912 (height width)")
    if args.label.lower() == "density":
        args.num_classes = 4
    elif args.num_classes != 1:
        parser.error("Only density uses four classes; other classification tasks are binary")
    missing_output_args = [
        name for name in ("tensorboard_path", "checkpoints", "output_path")
        if getattr(args, name) is None
    ]
    if missing_output_args:
        parser.error(
            "Set MAMMO_FM_EVAL_DIR or pass --tensorboard-path, --checkpoints, and --output_path. "
            "Evaluation artifacts must be stored outside the Mammo-FM repository."
        )
    return args


def main(args):
    print(f"args.data_frac: {args.data_frac}")
    print(f"args.dataset: {args.dataset}")
    print(f"args.label: {args.label}")
    print(f"args.arch: {args.arch}")
    print(f"args.data_dir: {args.data_dir}")
    print(f"args.img_dir: {args.img_dir}")
    print(f"args.csv_file: {args.csv_file}")
    args.data_frac = float(args.data_frac)
    seed_all(args.seed)
    # get paths
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no GPU is available")
    # args.root = f"lr_{args.lr}_epochs_{args.epochs}_weighted_BCE_{args.weighted_BCE}_balanced_dataloader_{args.balanced_dataloader}_{args.label}_data_frac_{args.data_frac}_post_miccai"
    args.root = f"lr_{args.lr}_epochs_{args.epochs}_weighted_BCE_{args.weighted_BCE}_{args.label}_data_frac_{args.data_frac}"
    args.apex = True if args.apex == "y" else False
    args.pretrained_swin_encoder = True if args.pretrained_swin_encoder == "y" else False
    args.swin_model_type = True if args.swin_model_type == "y" else False
    args.running_interactive = True if args.running_interactive == "y" else False

    chk_pt_path, output_path, tb_logs_path = get_Paths(args)
    args.chk_pt_path = chk_pt_path
    args.output_path = output_path
    args.tb_logs_path = tb_logs_path

    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    print("====================> Paths <====================")
    print(f"checkpoint_path: {chk_pt_path}")
    print(f"output_path: {output_path}")
    print(f"tb_logs_path: {tb_logs_path}")
    print('device:', device)
    print('torch version:', torch.__version__)
    print("====================> Paths <====================")

    pickle.dump(args, open(os.path.join(output_path, f"seed_{args.seed}_train_configs.pkl"), "wb"))
    torch.cuda.empty_cache()

    # if args.weighted_BCE == "y" and args.dataset.lower() == "rsna" and args.label.lower() == "cancer":
    #     args.BCE_weights = {
    #         "fold0": 46.48148148148148,
    #         "fold1": 46.01830663615561,
    #         "fold2": 46.41339491916859,
    #         "fold3": 46.05747126436781
    #     }
    # elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "mass":
    #     args.BCE_weights = {
    #         "fold0": 15.573306370070778,
    #         "fold1": 15.573306370070778,
    #         "fold2": 15.573306370070778,
    #         "fold3": 15.573306370070778
    #     }
    # elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "suspicious_calcification":
    #     args.BCE_weights = {
    #         "fold0": 37.296728971962615,
    #         "fold1": 37.296728971962615,
    #         "fold2": 37.296728971962615,
    #         "fold3": 37.296728971962615,
    #     }
    #
    # elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "focal_asymmetry":
    #     args.BCE_weights = {
    #         "fold0": 67.95945945945945,
    #         "fold1": 67.95945945945945,
    #         "fold2": 67.95945945945945,
    #         "fold3": 67.95945945945945,
    #     }
    # elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "asymmetry":
    #     args.BCE_weights = {
    #         "fold0": 230.9545454545455,
    #         "fold1": 230.9545454545455,
    #         "fold2": 230.9545454545455,
    #         "fold3": 230.9545454545455,
    #     }
    # elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "architectural_distortion":
    #     args.BCE_weights = {
    #         "fold0": 67.95945945945945,
    #         "fold1": 67.95945945945945,
    #         "fold2": 67.95945945945945,
    #         "fold3": 67.95945945945945,
    #     }
    # elif args.weighted_BCE == "y" and args.dataset.lower() == "image_label" and args.label.lower() == "calc":
    #     args.BCE_weights = {
    #         "fold0": 21.391733762748256,
    #         "fold1": 20.844534328977392,
    #         "fold2": 20.90514333895447,
    #         "fold3": 21.021697713801863,
    #     }
    # elif args.weighted_BCE == "y" and args.dataset.lower() == "image_label" and args.label.lower() == "mass":
    #     args.BCE_weights = {
    #         "fold0": 22.859414321665522,
    #         "fold1": 23.397660141318198,
    #         "fold2": 22.9801546094381,
    #         "fold3": 22.920556449758564,
    #     }
    # elif args.weighted_BCE == "y" and args.dataset.lower() == "image_label" and args.label.lower() == "cancer":
    #     args.BCE_weights = {
    #         "fold0": 185.73142345568488,
    #         "fold1": 192.94567219152856,
    #         "fold2": 187.0868778280543,
    #         "fold3": 197.53148854961833,
    #     }
    # elif args.weighted_BCE == "y" and args.dataset.lower() == "image_label" and args.label.lower() == "arch_distortion":
    #     args.BCE_weights = {
    #         "fold0": 155.70848985725019,
    #         "fold1": 155.94858420268255,
    #         "fold2": 153.29547141796584,
    #         "fold3": 157.70404271548438,
    #     }

    do_experiments(args, device)


if __name__ == "__main__":
    args = config()
    main(args)
