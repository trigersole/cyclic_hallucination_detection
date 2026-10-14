from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .features import CycleFeatures


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        shard_paths = sorted(path.glob("shard_*.jsonl"))
        if not shard_paths:
            raise ValueError(f"no shard_*.jsonl files found in {path}")
        return [row for shard in shard_paths for row in read_jsonl(shard)]
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} is not an object")
            rows.append(row)
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(rows, temporary)
    os.replace(temporary, path)


def features_from_row(row: dict[str, Any]) -> CycleFeatures:
    payload = row.get("features", row)
    return CycleFeatures(
        forward_entropy=float(payload["forward_entropy"]),
        cycle_surprisal=float(payload["cycle_surprisal"]),
        competing_margin=float(payload["competing_margin"]),
        original_question_support=float(payload.get("original_question_support", 0.0)),
        inverse_entropy=float(payload.get("inverse_entropy", 0.0)),
        forward_cluster_count=int(payload.get("forward_cluster_count", 0)),
        inverse_cluster_count=int(payload.get("inverse_cluster_count", 0)),
        interrogate_similarity=float(
            payload.get("interrogate_similarity", float("nan"))
        ),
    )

