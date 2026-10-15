#!/usr/bin/env python3
"""Summarize structured-answer compliance and possible question leakage."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from qa_cycle_detector.io import read_jsonl


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "for",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
}


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
        if token not in STOPWORDS and len(token) > 1
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    valid = 0
    overlap_values: list[float] = []
    forward_valid = 0
    forward_total = 0
    for row in rows:
        samples = row.get("samples", {})
        canonical = samples.get("canonicalization")
        flags = samples.get("forward_format_valid") or []
        forward_valid += sum(bool(flag) for flag in flags)
        forward_total += len(flags)
        if not isinstance(canonical, dict) or not canonical.get("valid"):
            reason = (
                canonical.get("failure_reason", "missing_metadata")
                if isinstance(canonical, dict)
                else "missing_metadata"
            )
            failures[str(reason)] += 1
            continue
        valid += 1
        answer_tokens = content_tokens(str(samples.get("original_answer", "")))
        question_tokens = content_tokens(str(row.get("question", ""))) - answer_tokens
        qualifier_tokens = content_tokens(str(canonical.get("qualifier", ""))) - answer_tokens
        overlap = len(question_tokens & qualifier_tokens) / max(1, len(qualifier_tokens))
        overlap_values.append(overlap)

    total = len(rows)
    flagged = sum(value >= 0.5 for value in overlap_values)
    return {
        "rows": total,
        "canonicalization_valid": valid,
        "canonicalization_valid_rate": valid / total if total else 0.0,
        "canonicalization_failures": dict(failures),
        "forward_samples": forward_total,
        "forward_format_valid_rate": (
            forward_valid / forward_total if forward_total else 0.0
        ),
        "mean_qualifier_question_overlap": (
            sum(overlap_values) / len(overlap_values) if overlap_values else 0.0
        ),
        "possible_leakage_rate_at_0.5": (
            flagged / len(overlap_values) if overlap_values else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    report = {
        split: summarize(read_jsonl(run_root / "features" / split))
        for split in ("train", "validation", "test")
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
