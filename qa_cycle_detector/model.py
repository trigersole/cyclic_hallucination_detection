from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import CycleFeatures


DEFAULT_FEATURE_NAMES = ["forward_entropy", "cycle_surprisal", "competing_margin"]
AVAILABLE_FEATURE_NAMES = {
    "forward_entropy",
    "cycle_surprisal",
    "competing_margin",
    "inverse_entropy",
    "original_question_support",
    "interrogate_similarity",
    "interrogate_risk",
}


def feature_matrix(
    features: Sequence[CycleFeatures], feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES
) -> np.ndarray:
    if not features:
        raise ValueError("at least one feature row is required")
    names = list(feature_names)
    if not names or not set(names) <= AVAILABLE_FEATURE_NAMES:
        raise ValueError(f"invalid feature names: {names}")
    matrix = np.asarray(
        [[float(getattr(row, name)) for name in names] for row in features],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError(
            "selected features contain missing or non-finite values; "
            "backfill InterrogateLLM features for legacy extraction outputs"
        )
    return matrix


@dataclass
class HallucinationClassifier:
    """Standardized binary logistic regression implemented with NumPy."""

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-values))

    @classmethod
    def fit(
        cls,
        features: Sequence[CycleFeatures],
        labels: Sequence[int],
        class_weight: str | None = "balanced",
        random_state: int = 42,
        learning_rate: float = 0.05,
        max_iter: int = 5000,
        l2: float = 1e-3,
        feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    ) -> "HallucinationClassifier":
        del random_state  # Optimization is deliberately deterministic.
        y = np.asarray(labels, dtype=np.int64)
        if len(features) != len(y):
            raise ValueError("features and labels must have equal length")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("training labels must contain both 0 and 1")

        names = list(feature_names)
        x = feature_matrix(features, names)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        z = (x - mean) / scale

        sample_weights = np.ones(len(y), dtype=np.float64)
        if class_weight == "balanced":
            for label in (0, 1):
                count = int((y == label).sum())
                sample_weights[y == label] = len(y) / (2.0 * count)
        elif class_weight is not None:
            raise ValueError("class_weight must be 'balanced' or None")

        weights = np.zeros(z.shape[1], dtype=np.float64)
        bias = 0.0
        normalizer = sample_weights.sum()
        for _ in range(max_iter):
            probabilities = cls._sigmoid(z @ weights + bias)
            error = (probabilities - y) * sample_weights
            weight_gradient = (z.T @ error) / normalizer + l2 * weights
            bias_gradient = float(error.sum() / normalizer)
            step = learning_rate * weight_gradient
            weights -= step
            bias -= learning_rate * bias_gradient
            if np.linalg.norm(step) < 1e-9 and abs(bias_gradient) < 1e-9:
                break
        return cls(
            weights=weights,
            bias=bias,
            mean=mean,
            scale=scale,
            feature_names=names,
        )

    def predict_proba(self, features: Sequence[CycleFeatures]) -> np.ndarray:
        x = (feature_matrix(features, self.feature_names) - self.mean) / self.scale
        return self._sigmoid(x @ self.weights + self.bias)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "feature_names": self.feature_names,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HallucinationClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        names = payload.get("feature_names")
        if not isinstance(names, list) or not set(names) <= AVAILABLE_FEATURE_NAMES:
            raise ValueError("model feature schema is incompatible")
        return cls(
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            feature_names=names,
        )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    positive_rank_sum = _average_ranks(scores)[labels == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_positives = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1)
    precision = cumulative_positives / ranks
    return float(precision[sorted_labels == 1].sum() / labels.sum())


def evaluate(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(scores, dtype=np.float64)
    if len(y) != len(probabilities):
        raise ValueError("labels and scores must have equal length")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("evaluation requires both label classes")
    accepted_order = np.argsort(probabilities, kind="mergesort")
    selective_risk = np.cumsum(y[accepted_order]) / np.arange(1, len(y) + 1)
    return {
        "auroc": _auroc(y, probabilities),
        "auprc": _average_precision(y, probabilities),
        "aurc": float(selective_risk.mean()),
    }


def save_metrics(metrics: dict[str, float], path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
