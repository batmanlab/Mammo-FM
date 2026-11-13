"""Generate short-answer, image-grounded finding questions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


PROMPT = """
You are a radiologist VQA dataset creator specializing in screening breast mammogram reports.
Create short-answer question/answer pairs that let a reader verify, on the images themselves, the presence, side, view, and location of every one of the following possible findings:
1. Mass
2. Suspicious Calcification
3. Asymmetry
4. Focal Asymmetry
5. Global Asymmetry
6. Architectural Distortion
7. Skin Thickening
8. Skin Retraction
9. Nipple Retraction
10. Suspicious Lymph Node

For each finding, generate four concise Q/A pairs in the exact sequence below:
a. Is the finding present?
b. If present, which breast (laterality) contains it? (Right, Left, or Both)
c. On which view is it best visualized? (MLO, CC, or Both)
d. What is its location within the breast? (clock-face or quadrant + depth)

If the finding is absent, answer the first question "No" and answer the next three with "Not applicable — no ... is present."
Do not reference the report verbatim or mention its specifics (e.g., who read the exam, the software used, or the date). All questions and answers must be visually driven, meaning that someone would need to look at the actual mammogram images to confirm the answer.
The question-answer pairs must be derived from the given radiology report only, nothing else.

Output everything inside a single <short_answer>...</short_answer> tag, with each question immediately followed by its answer:
<short_answer><q>Question text</q><a>Answer text</a>...</short_answer>
"""

REQUIRED_COLUMNS = {
    "patient_exam_id", "file_path", "Findings", "Impression", "dataset",
    "patient_id", "exam_id", "fold",
}


def parse_conversation(text: str) -> list[dict[str, str]]:
    match = re.search(r"<short_answer>(.*?)</short_answer>", text, re.DOTALL)
    if not match:
        return []
    turns: list[dict[str, str]] = []
    for index, (question, answer) in enumerate(
        re.findall(r"<q>(.*?)</q>\s*<a>(.*?)</a>", match.group(1), re.DOTALL)
    ):
        prefix = "<image>\n" if index == 0 else ""
        turns.extend(
            [
                {
                    "from": "human",
                    "value": f"{prefix}{question.strip()}<short_answer>",
                    "type": "short_answer",
                },
                {"from": "gpt", "value": answer.strip()},
            ]
        )
    return turns


def _generate(client: OpenAI, report: str, model: str, max_retries: int) -> list[dict[str, str]]:
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": report},
            ],
            temperature=0.0,
        )
        turns = parse_conversation(response.choices[0].message.content or "")
        if turns and all(turn["value"].strip() for turn in turns):
            return turns
        if attempt < max_retries:
            print(f"Invalid structured response; retrying ({attempt}/{max_retries}).")
    raise RuntimeError(f"Could not obtain valid short answers after {max_retries} attempts")


def _report_from_row(row: pd.Series) -> str:
    findings = str(row["Findings"]).strip() if pd.notna(row["Findings"]) else ""
    impression = str(row["Impression"]).strip() if pd.notna(row["Impression"]) else ""
    sections = []
    if findings:
        sections.append(f"Findings: {findings}")
    if impression:
        sections.append(f"Impression: {impression}")
    return ". ".join(sections)


def main(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.input_file)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

    start = args.chunk_index * args.chunk_size
    stop = min(start + args.chunk_size, len(frame))
    chunk = frame.iloc[start:stop]
    print(f"Processing rows {start}:{stop} ({len(chunk)} rows).")

    client = OpenAI()
    outputs: list[dict[str, object]] = []
    output_index = args.start_id + start
    for _, row in tqdm(chunk.iterrows(), total=len(chunk)):
        patient_exam_id = str(row["patient_exam_id"])
        outputs.append(
            {
                "id": f"short_answer_{output_index}",
                "patient_exam_id": patient_exam_id,
                "image": row["file_path"],
                "conversations": _generate(
                    client, _report_from_row(row), args.model, args.max_retries
                ),
                "dataset": row["dataset"],
                "patient_id": row["patient_id"],
                "exam_id": row["exam_id"],
                "fold": row["fold"],
            }
        )
        output_index += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(outputs)} conversations to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate short-answer finding questions.")
    parser.add_argument("--input-file", "--input_file_name", dest="input_file", type=Path, required=True)
    parser.add_argument("--output", "--save_file_name", dest="output", type=Path, required=True)
    parser.add_argument("--chunk-index", "--chunk_index", dest="chunk_index", type=int, default=0)
    parser.add_argument("--chunk-size", "--chunk_size", dest="chunk_size", type=int, default=3000)
    parser.add_argument("--start-id", "--start", dest="start_id", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
