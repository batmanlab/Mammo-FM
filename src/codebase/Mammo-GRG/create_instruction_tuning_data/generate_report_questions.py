"""Create deterministic report-generation instruction data from a CSV file."""

import argparse
import json
import random
from pathlib import Path


MAMMO_PROMPTS = [
    "Can you generate the report for the following 2D screening mammogram?",
    "Please provide the radiology report for the 2D screening mammogram mentioned.",
    "I need the radiology report for the given 2D screening mammogram.",
    "Could you create a report for this 2D screening mammogram?",
    "Would you mind generating the radiology report for the specified 2D screening mammogram?",
    "Please generate the report for the 2D screening mammogram provided.",
    "Can you produce the radiology report for the attached 2D screening mammogram?",
    "I need a detailed report for the given 2D screening mammogram.",
    "Could you write the radiology report for this 2D screening mammogram?",
    "Please give the radiology report for the specified 2D screening mammogram.",
    "Generate radiology report for the 2D screening mammogram.",
    "Produce the report for this 2D screening mammogram.",
    "Write a radiology report for the following 2D screening mammogram.",
    "Create a report for this 2D screening mammogram.",
    "Provide the radiology report for this 2D screening mammogram.",
]

REQUIRED_COLUMNS = {
    "patient_exam_id",
    "file_path",
    "Report",
    "dataset",
    "patient_id",
    "exam_id",
    "fold",
    "mass_present",
    "calc_present",
    "arch_distortion_present",
    "asymmetry_present",
}


def generate_report_questions(
    input_file_name: Path,
    save_file_name: str,
    chunk_index: int,
    seed: int,
) -> None:
    """Generate report conversations while preserving the original schema."""
    import pandas as pd
    import tqdm

    dataframe = pd.read_csv(input_file_name)
    missing = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

    random_generator = random.Random(seed)
    all_outputs = []

    for idx, (_, row) in enumerate(
        tqdm.tqdm(dataframe.iterrows(), total=len(dataframe))
    ):
        patient_exam_id = row["patient_exam_id"]
        file_path = row["file_path"]
        report = row["Report"]
        chosen_prompt = random_generator.choice(MAMMO_PROMPTS)

        conversation = [
            {
                "type": "report_generation",
                "from": "human",
                "value": f"<image>\n{chosen_prompt}<report_generation>",
            },
            {
                "from": "gpt",
                "value": report,
            },
        ]

        outputs = {
            "id": f"report_generation_{idx}",
            "patient_exam_id": patient_exam_id,
            "image": file_path,
            "conversations": conversation,
            "dataset": row["dataset"],
            "patient_id": row["patient_id"],
            "exam_id": row["exam_id"],
            "fold": row["fold"],
            "mass_present": row["mass_present"],
            "calc_present": row["calc_present"],
            "arch_distortion_present": row["arch_distortion_present"],
            "asymmetry_present": row["asymmetry_present"],
        }

        all_outputs.append(outputs)

    output_file = Path(save_file_name.format(chunk_index))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file_handle:
        json.dump(all_outputs, file_handle, indent=2, ensure_ascii=False)

    print(f"Saved conversations to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-generation instruction data from a CSV file."
    )
    parser.add_argument(
        "--input-file-name",
        "--input_file_name",
        "--input-file",
        dest="input_file_name",
        type=Path,
        required=True,
        help="Input CSV containing reports and exam metadata.",
    )
    parser.add_argument(
        "--save-file-name",
        "--save_file_name",
        "--output",
        dest="save_file_name",
        required=True,
        help="Output JSON path; it may contain a {0} chunk-index placeholder.",
    )
    parser.add_argument(
        "--chunk-index",
        "--chunk_index",
        dest="chunk_index",
        type=int,
        default=0,
        help="Value substituted for {0} in --save-file-name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic prompt selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_report_questions(
        input_file_name=args.input_file_name,
        save_file_name=args.save_file_name,
        chunk_index=args.chunk_index,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
