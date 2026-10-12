#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from qa_cycle_detector.io import atomic_write_jsonl, read_jsonl


PREFIXES = (
    "Answer with only the short answer. ",
    "Give a brief, factual answer. ",
)


def clean_question(question: str) -> str:
    cleaned = question.strip()
    for prefix in PREFIXES:
        if cleaned.startswith(prefix):
            return cleaned[len(prefix) :].strip()
    return cleaned


def convert_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        correct = int(row.get("correct", -1))
        if correct not in (0, 1):
            continue
        question = clean_question(str(row.get("question", "")))
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        identifier = str(row.get("fact_id", row.get("id", index)))
        dedupe_key = question.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        converted.append(
            {
                "id": identifier,
                "question": question,
                "answer": answer,
                "label": 1 - correct,
                "source_correct": correct,
                "grading_status": row.get("grading_status"),
            }
        )
    return converted


def stratified_split(
    rows: list[dict[str, Any]], seed: int, max_examples: int | None
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if max_examples is not None:
        shuffled = shuffled[:max_examples]

    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in shuffled:
        by_label[int(row["label"])].append(row)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        count = len(label_rows)
        train_end = int(0.70 * count)
        validation_end = train_end + int(0.15 * count)
        splits["train"].extend(label_rows[:train_end])
        splits["validation"].extend(label_rows[train_end:validation_end])
        splits["test"].extend(label_rows[validation_end:])
    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="0 keeps all eligible examples",
    )
    args = parser.parse_args()

    converted = convert_rows(read_jsonl(args.input))
    maximum = None if args.max_examples == 0 else args.max_examples
    splits = stratified_split(converted, args.seed, maximum)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        atomic_write_jsonl(rows, output_dir / f"{name}.jsonl")

    manifest = {
        "source": str(Path(args.input).resolve()),
        "seed": args.seed,
        "max_examples": maximum,
        "eligible_rows": len(converted),
        "splits": {
            name: {"rows": len(rows), "labels": dict(Counter(row["label"] for row in rows))}
            for name, rows in splits.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
