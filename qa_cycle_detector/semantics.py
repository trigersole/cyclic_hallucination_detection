from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class SemanticEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized embeddings with shape (n_texts, dimension)."""


@dataclass
class SentenceTransformerEncoder:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            local_files_only=self.local_files_only,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float64,
        )


def _normalized(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = _normalized(embeddings)
    return normalized @ normalized.T


def cluster_embeddings(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Cluster by connected components of the thresholded similarity graph.

    This is deterministic and order-independent. The threshold should be validated
    on a manually labeled set of question/answer pairs for a serious experiment.
    """

    count = len(embeddings)
    if count == 0:
        return []
    similarities = similarity_matrix(embeddings)
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(count):
        for right in range(left + 1, count):
            if similarities[left, right] >= threshold:
                union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).append(index)
    return sorted(grouped.values(), key=lambda cluster: cluster[0])


def equivalent_to_target(
    sample_embeddings: np.ndarray,
    target_embedding: np.ndarray,
    threshold: float,
) -> np.ndarray:
    samples = _normalized(sample_embeddings)
    target = _normalized(np.asarray(target_embedding).reshape(1, -1))[0]
    return (samples @ target) >= threshold

