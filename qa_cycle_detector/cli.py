from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import DetectorConfig, GenerationConfig
from .features import CycleFeatureExtractor
from .generation import HuggingFaceGenerator
from .io import features_from_row, read_jsonl, write_jsonl
from .model import HallucinationClassifier, evaluate, save_metrics
from .semantics import SentenceTransformerEncoder


def _extractor(args: argparse.Namespace) -> CycleFeatureExtractor:
    config = DetectorConfig(
        forward=GenerationConfig(
            samples=args.forward_samples,
            temperature=args.forward_temperature,
            max_new_tokens=args.answer_tokens,
        ),
        inverse=GenerationConfig(
            samples=args.inverse_samples,
            temperature=args.inverse_temperature,
            max_new_tokens=args.question_tokens,
        ),
        equivalence_threshold=args.threshold,
        smoothing=args.smoothing,
    )
    generator = HuggingFaceGenerator(
        model_name=args.generator_model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    encoder = SentenceTransformerEncoder(args.embedding_model, device=args.embedding_device)
    return CycleFeatureExtractor(generator, encoder, config)


def _extract(args: argparse.Namespace) -> None:
    extractor = _extractor(args)
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(args.input), start=1):
        try:
            question = str(row["question"])
            answer = str(row["answer"])
        except KeyError as error:
            raise ValueError(f"row {index} needs question and answer fields") from error
        features, details = extractor.extract(question, answer)
        output = dict(row)
        output["features"] = features.to_dict()
        if args.keep_samples:
            output["samples"] = details
        output_rows.append(output)
        print(f"extracted {index}", flush=True)
    write_jsonl(output_rows, args.output)


def _train(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    features = [features_from_row(row) for row in rows]
    labels = [int(row["label"]) for row in rows]
    classifier = HallucinationClassifier.fit(features, labels)
    classifier.save(args.output)
    train_metrics = evaluate(labels, classifier.predict_proba(features))
    print(json.dumps({"training_metrics": train_metrics}, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    features = [features_from_row(row) for row in rows]
    labels = [int(row["label"]) for row in rows]
    classifier = HallucinationClassifier.load(args.model)
    scores = classifier.predict_proba(features)
    metrics = evaluate(labels, scores)
    if args.output:
        save_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2))


def _score(args: argparse.Namespace) -> None:
    extractor = _extractor(args)
    features, details = extractor.extract(args.question, args.answer)
    result: dict[str, Any] = {"features": features.to_dict(), "samples": details}
    if args.model:
        classifier = HallucinationClassifier.load(args.model)
        result["hallucination_probability"] = float(
            classifier.predict_proba([features])[0]
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-mpnet-base-v2"
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--embedding-device")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward-samples", type=int, default=10)
    parser.add_argument("--inverse-samples", type=int, default=10)
    parser.add_argument("--forward-temperature", type=float, default=0.8)
    parser.add_argument("--inverse-temperature", type=float, default=0.8)
    parser.add_argument("--answer-tokens", type=int, default=48)
    parser.add_argument("--question-tokens", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--smoothing", type=float, default=0.5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qa-cycle")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract features from JSONL")
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--keep-samples", action="store_true")
    _generation_arguments(extract)
    extract.set_defaults(handler=_extract)

    train = subparsers.add_parser("train", help="train logistic regression")
    train.add_argument("--input", required=True)
    train.add_argument("--output", required=True)
    train.set_defaults(handler=_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a model")
    evaluate_parser.add_argument("--input", required=True)
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--output")
    evaluate_parser.set_defaults(handler=_evaluate)

    score = subparsers.add_parser("score", help="score one question-answer pair")
    score.add_argument("--question", required=True)
    score.add_argument("--answer", required=True)
    score.add_argument("--model")
    _generation_arguments(score)
    score.set_defaults(handler=_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
