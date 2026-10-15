from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Any

import numpy as np

from .config import DetectorConfig
from .generation import (
    TextGenerator,
    answer_prompt,
    canonicalization_prompt,
    inverse_prompt,
    parse_structured_answer,
)
from .semantics import SemanticEncoder, cluster_embeddings, similarities_to_target


def semantic_entropy(cluster_sizes: list[int]) -> float:
    total = sum(cluster_sizes)
    if total == 0:
        raise ValueError("semantic entropy needs at least one sample")
    probabilities = np.asarray(cluster_sizes, dtype=np.float64) / total
    return float(-(probabilities * np.log(probabilities)).sum())


@dataclass(frozen=True)
class CycleFeatures:
    forward_entropy: float
    cycle_surprisal: float
    competing_margin: float
    original_question_support: float
    inverse_entropy: float
    forward_cluster_count: int
    inverse_cluster_count: int
    interrogate_similarity: float = float("nan")

    @property
    def interrogate_risk(self) -> float:
        """InterrogateLLM-style risk; larger values indicate more mismatch."""

        return 1.0 - self.interrogate_similarity

    @property
    def model_vector(self) -> list[float]:
        # This is the deliberately narrow, agreed detector specification.
        return [self.forward_entropy, self.cycle_surprisal, self.competing_margin]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CycleFeatureExtractor:
    def __init__(
        self,
        generator: TextGenerator,
        encoder: SemanticEncoder,
        config: DetectorConfig | None = None,
    ) -> None:
        self.generator = generator
        self.encoder = encoder
        self.config = config or DetectorConfig()

    def extract(self, question: str, initial_answer: str) -> tuple[CycleFeatures, dict[str, Any]]:
        if not question.strip() or not initial_answer.strip():
            raise ValueError("question and initial_answer must be non-empty")

        forward_answers = self.generator.generate(
            [answer_prompt(question, self.config.answer_format)], self.config.forward
        )[0]

        inverse_answer = initial_answer
        canonicalization: dict[str, Any] | None = None
        if self.config.answer_format == "structured":
            raw_canonicalization = self.generator.generate(
                [canonicalization_prompt(question, initial_answer)],
                self.config.canonicalization,
            )[0][0]
            canonicalization = parse_structured_answer(
                raw_canonicalization, expected_answer=initial_answer
            )
            if canonicalization["valid"]:
                inverse_answer = " || ".join(
                    [
                        canonicalization["value"],
                        canonicalization["type"],
                        canonicalization["qualifier"],
                    ]
                )
        reconstructed_questions = self.generator.generate(
            [inverse_prompt(inverse_answer)], self.config.inverse
        )[0]
        if not forward_answers or not reconstructed_questions:
            raise RuntimeError("generator returned no samples")

        forward_embeddings = self.encoder.encode(forward_answers)
        forward_clusters = cluster_embeddings(
            forward_embeddings, self.config.equivalence_threshold
        )
        h_forward = semantic_entropy([len(cluster) for cluster in forward_clusters])

        inverse_embeddings = self.encoder.encode(reconstructed_questions)
        inverse_clusters = cluster_embeddings(
            inverse_embeddings, self.config.equivalence_threshold
        )
        h_inverse = semantic_entropy([len(cluster) for cluster in inverse_clusters])

        original_embedding = self.encoder.encode([question])[0]
        inverse_similarities = similarities_to_target(
            inverse_embeddings, original_embedding
        )
        interrogate_similarity = float(inverse_similarities.mean())
        matching = inverse_similarities >= self.config.equivalence_threshold
        matching_count = int(matching.sum())

        # If Q matches no observed cluster, smoothing includes Q as an unobserved
        # target category. Otherwise, it is one of the observed categories.
        categories = len(inverse_clusters) + (1 if matching_count == 0 else 0)
        denominator = len(reconstructed_questions) + self.config.smoothing * categories
        support = (matching_count + self.config.smoothing) / denominator

        alternative_counts = [
            sum(not bool(matching[index]) for index in cluster)
            for cluster in inverse_clusters
        ]
        largest_alternative = max(alternative_counts, default=0)
        alternative_support = (
            largest_alternative + self.config.smoothing
        ) / denominator

        eps = self.config.epsilon
        cycle_surprisal = -log(support + eps)
        competing_margin = log((alternative_support + eps) / (support + eps))

        features = CycleFeatures(
            forward_entropy=h_forward,
            cycle_surprisal=cycle_surprisal,
            competing_margin=competing_margin,
            original_question_support=support,
            inverse_entropy=h_inverse,
            forward_cluster_count=len(forward_clusters),
            inverse_cluster_count=len(inverse_clusters),
            interrogate_similarity=interrogate_similarity,
        )
        details = {
            "forward_answers": forward_answers,
            "forward_format_valid": [
                parse_structured_answer(answer)["valid"]
                for answer in forward_answers
            ]
            if self.config.answer_format == "structured"
            else None,
            "answer_format": self.config.answer_format,
            "original_answer": initial_answer,
            "inverse_answer": inverse_answer,
            "canonicalization": canonicalization,
            "forward_clusters": forward_clusters,
            "reconstructed_questions": reconstructed_questions,
            "inverse_clusters": inverse_clusters,
            "similarities_to_original": inverse_similarities.tolist(),
            "matches_original": matching.tolist(),
        }
        return features, details
