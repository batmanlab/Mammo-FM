"""Command-line interface for four-view Mammo-FM inference."""

from __future__ import annotations

import argparse
import json
import sys

from .exam import run_exam


def _parser():
    parser = argparse.ArgumentParser(
        prog="mammofm",
        description=(
            "Research-only four-view mammography report generation and "
            "direct, uncalibrated five-horizon risk probabilities."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    exam = subcommands.add_parser(
        "exam",
        help="Run one LMLO/LCC/RMLO/RCC exam from PNG or DICOM images.",
    )
    for view in ("lmlo", "lcc", "rmlo", "rcc"):
        exam.add_argument(
            f"--{view}", required=True, metavar="IMAGE",
            help=f"{view.upper()} DICOM or 8-bit PNG image",
        )
    exam.add_argument(
        "--output-dir", required=True,
        help="Directory for report.json, risk_mayo.json and risk_bu.json",
    )
    exam.add_argument(
        "--only", choices=("all", "risk", "report"), default="all",
        help="Generate all outputs (default), only both risk objects, or only the report",
    )
    exam.add_argument(
        "--bu-model", choices=("s1s2", "s2-only"), default="s1s2",
        help=("BU training protocol: Stage 1 MIRAI KD then Stage 2 outcomes "
              "(s1s2, default), or Stage 2 outcomes only (s2-only)"),
    )
    exam.add_argument(
        "--device", default=None,
        help="Torch device such as cuda, cuda:0, or cpu (default: CUDA when available)",
    )
    exam.add_argument(
        "--model-dir", default=None,
        help="Local directory mirroring the batmanLab/Mammo-FM Hugging Face tree",
    )
    exam.add_argument("--cache-dir", default=None, help="Hugging Face download cache")
    exam.add_argument(
        "--local-files-only", action="store_true",
        help="Disable model downloads and use only model-dir/cache files",
    )
    exam.add_argument(
        "--base-model-path", default=None,
        help=("Optional complete local Llama-3.1-8B-Instruct directory; by default "
              "the pinned gated Hugging Face snapshot is used"),
    )
    exam.add_argument(
        "--report-seed", type=int, default=0,
        help="Random seed for Stage-3 sampling (default: 0)",
    )
    exam.add_argument(
        "--preliminary-max-new-tokens", type=int, default=512,
        help="Maximum internal Stage-2 draft length (default: 512)",
    )
    exam.add_argument(
        "--grounding-max-new-tokens", type=int, default=320,
        help="Maximum final Stage-3 report length (default: 320)",
    )
    exam.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing requested JSON outputs",
    )
    exam.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages (the final JSON is still printed)",
    )
    return parser


def _validate_generation_options(args):
    for name in ("preliminary_max_new_tokens", "grounding_max_new_tokens"):
        if not 1 <= getattr(args, name) <= 4096:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must be between 1 and 4096")
    if not 0 <= args.report_seed < 2**32:
        raise ValueError("--report-seed must be in [0, 2**32)")


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != "exam":  # pragma: no cover - argparse enforces this.
        parser.error("a command is required")
    try:
        _validate_generation_options(args)
        progress = ((lambda message: None) if args.quiet else
                    (lambda message: print(f"[mammofm] {message}", file=sys.stderr, flush=True)))
        results = run_exam(
            {
                "LMLO": args.lmlo,
                "LCC": args.lcc,
                "RMLO": args.rmlo,
                "RCC": args.rcc,
            },
            args.output_dir,
            bu_model=args.bu_model.replace("-", "_"),
            device=args.device,
            cache_dir=args.cache_dir,
            model_dir=args.model_dir,
            local_files_only=args.local_files_only,
            base_model_path=args.base_model_path,
            report_seed=args.report_seed,
            preliminary_max_new_tokens=args.preliminary_max_new_tokens,
            grounding_max_new_tokens=args.grounding_max_new_tokens,
            only=args.only,
            overwrite=args.overwrite,
            progress=progress,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"mammofm: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(results, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
