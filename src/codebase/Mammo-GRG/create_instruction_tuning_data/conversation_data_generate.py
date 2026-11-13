"""Generate multi-turn, image-grounded mammography conversations.

Authentication for the hosted language-model client is read from the process
environment. Source reports and generated JSON must be kept outside the source
repository.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


PROMPT = """
You are a radiologist specializing in creating VQA (Visual Question Answering) datasets for screening breast mammograms.
Using the following mammography report as background, generate three sets of conversation-style Q&A pairs that are visually grounded in the mammogram images.
Generate three sets of conversation-style Q&A pairs based on the 2D screening mammograms, using the provided report as background without referring directly to it.
Enclose each set within tags <1>...</1>, <2>...</2>, and <3>...</3>. Use <q>...</q> for questions and <a>...</a> for answers, ensuring most answers are derived from the images.

Example: <1><q>Question 1</q><a>Answer 1</a><q>Question 2</q><a>Answer 2</a>...</1><2><q>Question 1</q><a>Answer 1</a><q>Question 2</q><a>Answer 2</a>...</2><3><q>Question 1</q><a>Answer 1</a><q>Question 2</q><a>Answer 2</a>...</3>.

Here are the constraints and guidelines:
1. Use the mammogram report strictly for context regarding findings and impressions.
Do not reference the report verbatim or mention its specifics (e.g., who read the exam, the software used, or the date).

2. All questions and answers must be visually driven, meaning that someone would need to look at the actual mammogram images to confirm the answer.
Also, the question-answer must be derived from the given radiology report only, nothing else.

3. Focus strictly on the core findings or impressions.

4. Do not include random or irrelevant questions with respect to the report. All questions should relate to what can be observed or concluded from the mammogram images.

5. Also include complex questions that are relevant to the report accompanied by the 2D mammogram images only. Provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized.

6. You can include questions about BI-RADS only if BI-RADS assessment is mentioned in the report explicitly. If BI-RADS is not mentioned in the report, do not include questions on the overall BI-RADS assessment.
You can include as many question and answer couples as you find appropriate. Here is the report for you to get information about 2D screening mammograms:
"""

REQUIRED_COLUMNS = {
    "patient_exam_id",
    "file_path",
    "Findings",
    "Impression",
    "dataset",
    "patient_id",
    "exam_id",
    "fold",
}


def parse_conversation(text: str) -> dict[str, list[dict[str, str]]]:
    """Parse numbered XML-like conversation blocks."""
    parsed: dict[str, list[dict[str, str]]] = {}
    for set_number, content in re.findall(
        r"<(\d+)>(.*?)</\1>", text, flags=re.DOTALL
    ):
        turns: list[dict[str, str]] = []
        for index, (question, answer) in enumerate(
            re.findall(
                r"<q>(.*?)</q>\s*<a>(.*?)</a>", content, flags=re.DOTALL
            )
        ):
            prefix = "<image>\n" if index == 0 else ""
            turns.extend(
                [
                    {
                        "from": "human",
                        "value": f"{prefix}{question.strip()}<long_answer>",
                        "type": "conversation_questions",
                    },
                    {"from": "gpt", "value": answer.strip()},
                ]
            )
        parsed[set_number] = turns
    return parsed


def _valid(parsed: dict[str, list[dict[str, str]]], expected: int) -> bool:
    expected_keys = {str(index) for index in range(1, expected + 1)}
    if set(parsed) != expected_keys:
        return False
    return all(
        turns and all(turn["value"].strip() for turn in turns)
        for turns in parsed.values()
    )


def _generate(
    client: OpenAI,
    report: str,
    model: str,
    max_retries: int,
) -> dict[str, list[dict[str, str]]]:
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": report},
            ],
            temperature=0.0,
        )
        parsed = parse_conversation(response.choices[0].message.content or "")
        if _valid(parsed, expected=3):
            return parsed
        if attempt < max_retries:
            print(f"Invalid structured response; retrying ({attempt}/{max_retries}).")
    raise RuntimeError(f"Could not obtain three valid conversation sets after {max_retries} attempts")


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
    output_index = args.start_id + (start * 3)

    for _, row in tqdm(chunk.iterrows(), total=len(chunk)):
        parsed = _generate(client, _report_from_row(row), args.model, args.max_retries)
        patient_exam_id = str(row["patient_exam_id"])
        for set_number in ("1", "2", "3"):
            sample: dict[str, object] = {
                "id": f"long_answer_{output_index}",
                "patient_exam_id": patient_exam_id,
                "image": row["file_path"],
                "conversations": parsed[set_number],
                "dataset": row["dataset"],
                "patient_id": row["patient_id"],
                "exam_id": row["exam_id"],
                "fold": row["fold"],
            }
            outputs.append(sample)
            output_index += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(outputs)} conversations to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate three image-grounded conversation sets per report."
    )
    parser.add_argument("--input-file", "--input_file_name", dest="input_file", type=Path, required=True)
    parser.add_argument("--output", "--save_file_name", dest="output", type=Path, required=True)
    parser.add_argument("--chunk-index", "--chunk_index", dest="chunk_index", type=int, default=0)
    parser.add_argument("--chunk-size", "--chunk_size", dest="chunk_size", type=int, default=30)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
