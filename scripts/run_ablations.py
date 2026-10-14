#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_cycle_detector.io import features_from_row, read_jsonl
from qa_cycle_detector.model import HallucinationClassifier, evaluate, feature_matrix


FEATURE_SETS = {
    "forward_only": ["forward_entropy"],
    "cycle_only": ["cycle_surprisal"],
    "competition_only": ["competing_margin"],
    "interrogate_calibrated": ["interrogate_risk"],
    "forward_cycle": ["forward_entropy", "cycle_surprisal"],
    "forward_interrogate": ["forward_entropy", "interrogate_risk"],
    "full": ["forward_entropy", "cycle_surprisal", "competing_margin"],
}


def load(path: str):
    rows = read_jsonl(path)
    return [features_from_row(row) for row in rows], [int(row["label"]) for row in rows]


def train(input_path: str, output_dir: Path) -> None:
    features, labels = load(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "interrogate_raw": evaluate(
            labels, feature_matrix(features, ["interrogate_risk"])[:, 0]
        )
    }
    for name, feature_names in FEATURE_SETS.items():
        classifier = HallucinationClassifier.fit(
            features, labels, feature_names=feature_names
        )
        classifier.save(output_dir / f"{name}.json")
        metrics[name] = evaluate(labels, classifier.predict_proba(features))
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def evaluate_all(input_path: str, model_dir: Path, output: Path) -> None:
    features, labels = load(input_path)
    metrics = {
        "interrogate_raw": evaluate(
            labels, feature_matrix(features, ["interrogate_risk"])[:, 0]
        )
    }
    for name in FEATURE_SETS:
        classifier = HallucinationClassifier.load(model_dir / f"{name}.json")
        metrics[name] = evaluate(labels, classifier.predict_proba(features))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--output-dir", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input", required=True)
    evaluate_parser.add_argument("--model-dir", required=True)
    evaluate_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.input, Path(args.output_dir))
    else:
        evaluate_all(args.input, Path(args.model_dir), Path(args.output))


if __name__ == "__main__":
    main()
