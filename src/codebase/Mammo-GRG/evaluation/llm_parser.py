#!/usr/bin/env python3
"""Parse generated mammography reports with a local instruction model.

The finding schema and extraction rules are adapted from
``analysis/extract_gt_findings_final.py`` in the research checkout. Dataset paths
and hosted-model calls are intentionally absent from this release.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

try:
    from ._local_generation import (
        GenerationConfig,
        LocalGenerator,
        extract_json_object,
    )
except ImportError:
    from _local_generation import GenerationConfig, LocalGenerator, extract_json_object


FINDINGS = ("mass", "calcification", "architectural_distortion", "asymmetry")

SYSTEM_PROMPT = """You are a breast-imaging report parser. Extract only facts explicitly
stated in the supplied report. Return exactly one JSON object and no prose.

The object must contain:
{
  "mass": 0,
  "mass_laterality": "no mention",
  "mass_view": "no mention",
  "calcification": 0,
  "calcification_laterality": "no mention",
  "calcification_view": "no mention",
  "architectural_distortion": 0,
  "architectural_distortion_laterality": "no mention",
  "architectural_distortion_view": "no mention",
  "asymmetry": 0,
  "asymmetry_laterality": "no mention",
  "asymmetry_view": "no mention",
  "birads": "no mention"
}

Use integer 0 or 1 for finding presence. Laterality must be left, right,
bilateral, or no mention. View must be CC, MLO, both, or no mention. When a
finding is absent, its laterality and view must both be no mention. BI-RADS must
be only its category number or no mention. Do not infer omitted information."""


def _binary(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) in (0, 1):
        return int(value)
    if str(value).strip().lower() in {"1", "present", "yes", "true"}:
        return 1
    return 0


def _choice(value: Any, allowed: Mapping[str, str]) -> str:
    normalized = str(value or "").strip().lower().replace("_", " ")
    return allowed.get(normalized, "no mention")


def normalize_parser_output(raw: Mapping[str, Any]) -> Dict[str, Any]:
    aliases = {
        "calcification": ("calcification", "calc"),
        "architectural_distortion": (
            "architectural_distortion",
            "arch_distortion",
        ),
    }
    laterality_allowed = {
        "left": "left",
        "right": "right",
        "bilateral": "bilateral",
        "both": "bilateral",
        "no mention": "no mention",
    }
    view_allowed = {
        "cc": "CC",
        "mlo": "MLO",
        "both": "both",
        "cc and mlo": "both",
        "no mention": "no mention",
    }
    normalized: Dict[str, Any] = {}

    for finding in FINDINGS:
        names = aliases.get(finding, (finding,))
        source_name = next((name for name in names if name in raw), names[0])
        present = _binary(raw.get(source_name, 0))
        normalized[finding] = present
        if present:
            normalized[f"{finding}_laterality"] = _choice(
                raw.get(f"{source_name}_laterality"), laterality_allowed
            )
            normalized[f"{finding}_view"] = _choice(
                raw.get(f"{source_name}_view"), view_allowed
            )
        else:
            normalized[f"{finding}_laterality"] = "no mention"
            normalized[f"{finding}_view"] = "no mention"

    birads = raw.get("birads", raw.get("BI-RADS", raw.get("bi-rads", "no mention")))
    birads_text = str(birads).strip()
    normalized["birads"] = birads_text if birads_text.isdigit() else "no mention"
    return normalized


def validate_parser_output(raw: Mapping[str, Any]) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("Parser output must be a JSON object")
    accepted_names = {
        "mass": ("mass",),
        "calcification": ("calcification", "calc"),
        "architectural_distortion": (
            "architectural_distortion",
            "arch_distortion",
        ),
        "asymmetry": ("asymmetry",),
    }
    missing = [
        finding
        for finding, names in accepted_names.items()
        if not any(name in raw for name in names)
    ]
    if not any(name in raw for name in ("birads", "BI-RADS", "bi-rads")):
        missing.append("birads")
    if missing:
        raise ValueError("Missing parser fields: " + ", ".join(missing))


def read_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        return list(reader), list(reader.fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-col", default="generated_report")
    parser.add_argument("--model", required=True, help="Local path or model identifier")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    args = parser.parse_args()

    rows, input_fields = read_rows(args.input_csv)
    if args.report_col not in input_fields:
        raise ValueError(f"Missing report column: {args.report_col}")

    generator = LocalGenerator(
        GenerationConfig(
            model=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            dtype=args.dtype,
        )
    )
    prompts = [f"Report:\n{row.get(args.report_col, '')}" for row in rows]
    raw_outputs = generator.generate(SYSTEM_PROMPT, prompts)

    parsed_rows: List[Dict[str, Any]] = []
    for row, raw_output in zip(rows, raw_outputs):
        result: Dict[str, Any] = dict(row)
        try:
            raw_parsed = extract_json_object(raw_output)
            validate_parser_output(raw_parsed)
            result.update(normalize_parser_output(raw_parsed))
            result["parser_status"] = "ok"
        except (TypeError, ValueError) as exc:
            result.update(normalize_parser_output({}))
            result["parser_status"] = f"invalid: {exc}"
        result["parser_raw_output"] = raw_output
        parsed_rows.append(result)

    parsed_fields = [
        item
        for finding in FINDINGS
        for item in (finding, f"{finding}_laterality", f"{finding}_view")
    ] + ["birads", "parser_status", "parser_raw_output"]
    output_fields = input_fields + [field for field in parsed_fields if field not in input_fields]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(parsed_rows)

    print(json.dumps({"rows": len(parsed_rows), "output": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
