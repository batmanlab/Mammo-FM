#!/usr/bin/env python3
"""Compute corpus BLEU-1..4 and mean ROUGE-L for report pairs.

The released research code selected BLEU and ROUGE-L through
``evaluations/new_llm_metrics.py`` and ``analysis/compute_scores_mammo_clip.py``.
This implementation is self-contained and removes experiment-specific paths.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def ngrams(tokens: Sequence[str], order: int) -> Counter:
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def corpus_bleu(
    references: Sequence[Sequence[str]], hypotheses: Sequence[Sequence[str]], max_order: int = 4
) -> Dict[str, float]:
    if len(references) != len(hypotheses):
        raise ValueError("Reference and hypothesis counts differ")
    matches = [0] * max_order
    possible = [0] * max_order
    reference_length = sum(len(tokens) for tokens in references)
    hypothesis_length = sum(len(tokens) for tokens in hypotheses)

    for reference, hypothesis in zip(references, hypotheses):
        for order in range(1, max_order + 1):
            reference_ngrams = ngrams(reference, order)
            hypothesis_ngrams = ngrams(hypothesis, order)
            matches[order - 1] += sum((reference_ngrams & hypothesis_ngrams).values())
            possible[order - 1] += max(len(hypothesis) - order + 1, 0)

    brevity_penalty = 0.0
    if hypothesis_length > 0:
        brevity_penalty = (
            1.0
            if hypothesis_length > reference_length
            else math.exp(1.0 - reference_length / hypothesis_length)
        )

    scores: Dict[str, float] = {}
    precisions = [
        matches[index] / possible[index] if possible[index] else 0.0
        for index in range(max_order)
    ]
    for order in range(1, max_order + 1):
        included = precisions[:order]
        if brevity_penalty == 0.0 or any(value == 0.0 for value in included):
            score = 0.0
        else:
            score = brevity_penalty * math.exp(sum(math.log(value) for value in included) / order)
        scores[f"BLEU_{order}"] = score
    return scores


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l(reference: Sequence[str], hypothesis: Sequence[str], beta: float = 1.2) -> float:
    if not reference or not hypothesis:
        return 0.0
    common = lcs_length(reference, hypothesis)
    precision = common / len(hypothesis)
    recall = common / len(reference)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    beta_squared = beta * beta
    return ((1.0 + beta_squared) * precision * recall) / (
        recall + beta_squared * precision
    )


def read_pairs(path: Path, reference_col: str, hypothesis_col: str) -> Iterable[Tuple[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {reference_col, hypothesis_col} - fields
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            reference = str(row.get(reference_col) or "").strip()
            hypothesis = str(row.get(hypothesis_col) or "").strip()
            if reference and hypothesis:
                yield reference, hypothesis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--reference-col", default="ground_truth_report")
    parser.add_argument("--hypothesis-col", default="generated_report")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    pairs = list(read_pairs(args.input_csv, args.reference_col, args.hypothesis_col))
    if not pairs:
        raise ValueError("No non-empty report pairs were found")
    references = [tokenize(reference) for reference, _ in pairs]
    hypotheses = [tokenize(hypothesis) for _, hypothesis in pairs]

    scores = corpus_bleu(references, hypotheses)
    rouge_scores = [
        rouge_l(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    ]
    scores["ROUGE_L"] = sum(rouge_scores) / len(rouge_scores)
    result = {"n": len(pairs), **scores}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
