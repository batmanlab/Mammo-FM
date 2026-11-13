#!/usr/bin/env python3
"""Generate reports from four precomputed Mammo-FM feature maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

GRG_ROOT = Path(__file__).resolve().parents[1]
if str(GRG_ROOT) not in sys.path:
    sys.path.insert(0, str(GRG_ROOT))

from mammo_grg.inference import generate_report, load_four_view_embeddings, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Downloaded stage-2 checkpoint directory")
    parser.add_argument("--model-base", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--embeddings",
        nargs=4,
        required=True,
        metavar=("LMLO", "LCC", "RMLO", "RCC"),
        help="Four .pt files in canonical view order",
    )
    parser.add_argument("--question", default="Generate a complete mammography report.")
    parser.add_argument("--output", default=None, help="Optional output JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer, model = load_model(
        args.model_path, args.model_base, device=args.device, load_4bit=args.load_4bit
    )
    embeddings = load_four_view_embeddings(
        args.embeddings, device=model.device, dtype=model.dtype
    )
    report = generate_report(
        model,
        tokenizer,
        embeddings,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    result = {"generated_report": report}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
