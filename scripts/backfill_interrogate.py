#!/usr/bin/env python3
"""Add InterrogateLLM mean-cosine scores to existing cached feature shards.

This reuses saved inverse-question samples and only runs the embedding model. It
never invokes the text generator. Source shards are left unchanged; augmented
copies are written below ``features_interrogate`` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qa_cycle_detector.io import atomic_write_jsonl, read_jsonl
from qa_cycle_detector.semantics import (
    SentenceTransformerEncoder,
    similarities_to_target,
)


SPLITS = ("train", "validation", "test")


def augment_rows(
    rows: list[dict[str, Any]], encoder: SentenceTransformerEncoder
) -> list[dict[str, Any]]:
    """Return rows augmented with mean inverse-query cosine similarity."""

    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for index, row in enumerate(rows, start=1):
        question = str(row.get("question", "")).strip()
        samples = row.get("samples")
        reconstructed = (
            samples.get("reconstructed_questions")
            if isinstance(samples, dict)
            else None
        )
        if not question:
            raise ValueError(f"row {index} has no question")
        if not isinstance(reconstructed, list) or not reconstructed:
            raise ValueError(
                f"row {index} has no saved reconstructed questions; "
                "the original extraction must use --keep-samples"
            )
        start = len(texts)
        texts.extend([question, *(str(item) for item in reconstructed)])
        spans.append((start, len(reconstructed)))

    embeddings = encoder.encode(texts)
    augmented: list[dict[str, Any]] = []
    for row, (start, sample_count) in zip(rows, spans, strict=True):
        similarities = similarities_to_target(
            embeddings[start + 1 : start + 1 + sample_count], embeddings[start]
        )
        output = dict(row)
        features = dict(output.get("features", {}))
        features["interrogate_similarity"] = float(similarities.mean())
        output["features"] = features
        samples = dict(output["samples"])
        samples["similarities_to_original"] = similarities.tolist()
        output["samples"] = samples
        augmented.append(output)
    return augmented


def backfill_split(
    source: Path,
    destination: Path,
    encoder: SentenceTransformerEncoder,
) -> None:
    shard_paths = sorted(source.glob("shard_*.jsonl"))
    if not shard_paths:
        raise ValueError(f"no shard_*.jsonl files found in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for number, shard_path in enumerate(shard_paths, start=1):
        target = destination / shard_path.name
        if target.exists():
            continue
        atomic_write_jsonl(augment_rows(read_jsonl(shard_path), encoder), target)
        print(
            f"completed {source.name} shard {number}/{len(shard_paths)}",
            flush=True,
        )
    success = {"complete": True, "num_shards": len(shard_paths)}
    (destination / "_SUCCESS.json").write_text(
        json.dumps(success, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    source_root = run_root / "features"
    output_root = run_root / "features_interrogate"
    configuration = {
        "schema_version": 1,
        "source_root": str(source_root.resolve()),
        "embedding_model": args.embedding_model,
        "splits": list(args.splits),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "_RUN.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != configuration:
            raise RuntimeError(
                f"existing backfill configuration differs in {output_root}; "
                "remove it or select a different run root"
            )
    else:
        manifest.write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )

    encoder = SentenceTransformerEncoder(
        args.embedding_model,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    for split in args.splits:
        backfill_split(source_root / split, output_root / split, encoder)


if __name__ == "__main__":
    main()
