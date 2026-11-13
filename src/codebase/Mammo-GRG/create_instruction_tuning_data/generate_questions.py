"""Generate text-only breast-imaging question-answer instructions from articles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


PROMPT = """
You are a medical dataset creator specializing in generating question-and-answer sets based on radiology articles related to breast imaging, mammograms, and breast cancer research.
Generate 5 question-and-answer pairs specifically derived from the provided article. Enclose each question in <q>...</q>, each answer in <a>...</a>, and all output in one <1>...</1> block.

Question requirements:
- Questions should be directly relevant to the content, findings, or key concepts discussed in the article.
- Questions should be educational, research-focused, and thought-provoking for students, radiologists, or medical researchers.
- Do not generate arbitrary or general questions about mammograms unless the article discusses mammographic findings.

Answer requirements:
- Answers must be factually accurate and based on the given article.
- Answers should be informative, well explained, and useful for academic and medical discussion.
- Do not reference specific study data or cases; summarize general scientific findings and concepts from the article.
- Ensure the information is universally applicable to 2D mammograms when the question concerns mammography.

Example topics include standard screening views, BI-RADS categories, diagnostic workup of calcifications or asymmetry, benign calcifications, mass characterization, bilateral lymphadenopathy, prior breast-conserving therapy, and management of multiple bilateral waxing and waning masses.
"""

REQUIRED_COLUMNS = {"Title", "Full Text"}


def parse_conversation(text: str) -> list[dict[str, str]]:
    match = re.search(r"<1>(.*?)</1>", text, re.DOTALL)
    if not match:
        return []
    turns: list[dict[str, str]] = []
    for question, answer in re.findall(
        r"<q>(.*?)</q>\s*<a>(.*?)</a>", match.group(1), re.DOTALL
    ):
        turns.extend(
            [
                {
                    "from": "human",
                    "value": f"{question.strip()}<long_answer>",
                    "type": "only_text",
                },
                {"from": "gpt", "value": answer.strip()},
            ]
        )
    return turns


def _generate(
    client: OpenAI,
    title: str,
    article: str,
    model: str,
    max_retries: int,
) -> list[dict[str, str]]:
    source = f"{title}\n\n{article}"
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": source},
            ],
            temperature=0.0,
        )
        turns = parse_conversation(response.choices[0].message.content or "")
        if turns and all(turn["value"].strip() for turn in turns):
            return turns
        if attempt < max_retries:
            print(f"Invalid structured response; retrying ({attempt}/{max_retries}).")
    raise RuntimeError(f"Could not obtain valid article questions after {max_retries} attempts")


def main(args: argparse.Namespace) -> None:
    if args.input_file.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(args.input_file)
    else:
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
    for row_index, row in tqdm(chunk.iterrows(), total=len(chunk)):
        title = str(row["Title"])
        outputs.append(
            {
                "id": f"long_answer_{output_index}",
                "row_id": int(row_index) if isinstance(row_index, int) else str(row_index),
                "article_title": title,
                "conversations": _generate(
                    client,
                    title,
                    str(row["Full Text"]),
                    args.model,
                    args.max_retries,
                ),
            }
        )
        output_index += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(outputs)} article conversations to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text-only Q&A from breast-imaging articles.")
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
