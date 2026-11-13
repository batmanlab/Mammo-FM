"""Combine list-valued JSON files into one instruction-tuning dataset."""

import argparse
import json
from pathlib import Path
from typing import List


def combine_json_files(input_files: List[Path], output_file: Path) -> None:
    """Merge list-valued JSON files in the order supplied by the caller."""
    combined_data = []

    for input_file in input_files:
        with input_file.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        if not isinstance(data, list):
            raise TypeError(f"Expected a JSON list in {input_file}")
        combined_data.extend(data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file_handle:
        json.dump(combined_data, file_handle, indent=2, ensure_ascii=False)

    print(f"All JSON files have been merged into {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge list-valued JSON files into a single JSON list."
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        type=Path,
        required=True,
        help="Input JSON files, in the order in which they should be merged.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Destination for the merged JSON list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combine_json_files(args.input_files, args.output_file)


if __name__ == "__main__":
    main()
