#!/usr/bin/env python3
"""Evaluate report pairs with a local Mammo-GREEN clinical judge.

The six clinical error categories and scoring rule follow the later
``analysis/mammo_green_gpt.py`` research script. The judge here is a local
Transformers causal model; malformed generations are marked invalid and are
excluded from the aggregate rather than silently receiving a score of zero.
"""

import argparse
import csv
import json
import statistics
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


ERROR_FIELDS = (
    "a_false_report",
    "b_missing_finding",
    "c_mischaracterization",
    "d_location_laterality",
    "e_incorrect_birads",
    "f_density_mismatch",
)

SYSTEM_PROMPT = """You are a breast-imaging report evaluator. Compare a candidate
mammography report with its reference and judge clinical content only.

Count these clinically significant errors:
a_false_report: a finding appears only in the candidate.
b_missing_finding: a reference finding is absent from the candidate.
c_mischaracterization: a shared finding has an important incorrect property.
d_location_laterality: a shared finding has incorrect side or location.
e_incorrect_birads: the final assessments meaningfully differ. Treat categories
1 and 2 as clinically equivalent. If both omit an assessment, count zero.
f_density_mismatch: overall breast-composition categories differ, or only one
report states breast composition. Do not confuse this with focal-lesion density.

matched_findings counts correctly matched clinically meaningful non-density
findings. matched_density is 1 only when both reports explicitly state the same
overall breast-composition category; otherwise it is 0. Do not count wording
overlap. Return exactly one JSON object and no prose, with this structure:
{
  "matched_findings": 0,
  "matched_density": 0,
  "matched_findings_explanations": [],
  "significant_errors": {
    "a_false_report": 0,
    "b_missing_finding": 0,
    "c_mischaracterization": 0,
    "d_location_laterality": 0,
    "e_incorrect_birads": 0,
    "f_density_mismatch": 0
  },
  "significant_error_explanations": {
    "a_false_report": [],
    "b_missing_finding": [],
    "c_mischaracterization": [],
    "d_location_laterality": [],
    "e_incorrect_birads": [],
    "f_density_mismatch": []
  },
  "style_or_wording_errors": 0
}
All counts must be non-negative integers. Explanation values must be arrays of
short strings. Return zero and an empty array for any category with no error."""


def nonnegative_integer(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def normalize_judgment(raw: Mapping[str, Any]) -> Dict[str, Any]:
    raw_errors = raw.get("significant_errors", {})
    if not isinstance(raw_errors, Mapping):
        raw_errors = {}
    raw_explanations = raw.get("significant_error_explanations", {})
    if not isinstance(raw_explanations, Mapping):
        raw_explanations = {}

    errors = {field: nonnegative_integer(raw_errors.get(field, 0)) for field in ERROR_FIELDS}
    explanations = {
        field: string_list(raw_explanations.get(field, [])) for field in ERROR_FIELDS
    }
    matched_findings = nonnegative_integer(raw.get("matched_findings", 0))
    matched_density = min(1, nonnegative_integer(raw.get("matched_density", 0)))
    return {
        "matched_findings": matched_findings,
        "matched_density": matched_density,
        "matched_findings_explanations": string_list(
            raw.get("matched_findings_explanations", [])
        ),
        "significant_errors": errors,
        "significant_error_explanations": explanations,
        "style_or_wording_errors": nonnegative_integer(
            raw.get("style_or_wording_errors", 0)
        ),
    }


def validate_judgment(raw: Mapping[str, Any]) -> None:
    required = {
        "matched_findings",
        "matched_density",
        "matched_findings_explanations",
        "significant_errors",
        "significant_error_explanations",
        "style_or_wording_errors",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError("Missing Mammo-GREEN fields: " + ", ".join(missing))
    for container_name in ("significant_errors", "significant_error_explanations"):
        container = raw.get(container_name)
        if not isinstance(container, Mapping):
            raise ValueError(container_name + " must be a JSON object")
        missing_categories = [field for field in ERROR_FIELDS if field not in container]
        if missing_categories:
            raise ValueError(
                "Missing Mammo-GREEN categories in "
                + container_name
                + ": "
                + ", ".join(missing_categories)
            )


def mammo_green_score(judgment: Mapping[str, Any]) -> float:
    matched = nonnegative_integer(judgment["matched_findings"]) + min(
        1, nonnegative_integer(judgment["matched_density"])
    )
    significant = sum(
        nonnegative_integer(value)
        for value in judgment["significant_errors"].values()
    )
    return matched / (matched + significant) if matched else 0.0


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
    parser.add_argument("--reference-col", default="ground_truth_report")
    parser.add_argument("--hypothesis-col", default="generated_report")
    parser.add_argument("--model", required=True, help="Local path or model identifier")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    args = parser.parse_args()

    rows, input_fields = read_rows(args.input_csv)
    missing = {args.reference_col, args.hypothesis_col} - set(input_fields)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

    generator = LocalGenerator(
        GenerationConfig(
            model=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            dtype=args.dtype,
        )
    )
    prompts = [
        "Reference report:\n"
        + str(row.get(args.reference_col) or "")
        + "\n\nCandidate report:\n"
        + str(row.get(args.hypothesis_col) or "")
        for row in rows
    ]
    raw_outputs = generator.generate(SYSTEM_PROMPT, prompts)

    output_rows: List[Dict[str, Any]] = []
    valid_scores: List[float] = []
    for row, raw_output in zip(rows, raw_outputs):
        result: Dict[str, Any] = dict(row)
        try:
            raw_judgment = extract_json_object(raw_output)
            validate_judgment(raw_judgment)
            judgment = normalize_judgment(raw_judgment)
            score = mammo_green_score(judgment)
            valid_scores.append(score)
            result["mammo_green_status"] = "ok"
            result["mammo_green_score"] = score
            result["matched_findings"] = judgment["matched_findings"]
            result["matched_density"] = judgment["matched_density"]
            result["significant_errors"] = json.dumps(
                judgment["significant_errors"], sort_keys=True
            )
            result["significant_error_explanations"] = json.dumps(
                judgment["significant_error_explanations"], sort_keys=True
            )
            result["matched_findings_explanations"] = json.dumps(
                judgment["matched_findings_explanations"]
            )
            result["style_or_wording_errors"] = judgment["style_or_wording_errors"]
        except (TypeError, ValueError) as exc:
            result["mammo_green_status"] = f"invalid: {exc}"
            result["mammo_green_score"] = ""
        result["mammo_green_raw_output"] = raw_output
        output_rows.append(result)

    added_fields = [
        "mammo_green_status",
        "mammo_green_score",
        "matched_findings",
        "matched_density",
        "significant_errors",
        "significant_error_explanations",
        "matched_findings_explanations",
        "style_or_wording_errors",
        "mammo_green_raw_output",
    ]
    output_fields = input_fields + [field for field in added_fields if field not in input_fields]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "n_total": len(output_rows),
        "n_valid": len(valid_scores),
        "n_invalid": len(output_rows) - len(valid_scores),
        "mean_mammo_green": (
            sum(valid_scores) / len(valid_scores) if valid_scores else None
        ),
        "std_mammo_green": statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0,
        "model": args.model,
    }
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "output": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
