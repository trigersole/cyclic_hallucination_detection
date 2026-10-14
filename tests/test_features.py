from __future__ import annotations

import numpy as np
import pytest

from qa_cycle_detector.config import DetectorConfig, GenerationConfig
from qa_cycle_detector.features import CycleFeatureExtractor, semantic_entropy


class FakeGenerator:
    def __init__(self, forward: list[str], inverse: list[str]) -> None:
        self.outputs = [forward, inverse]

    def generate(self, prompts, config):
        return [self.outputs.pop(0)]


class LookupEncoder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts):
        return np.asarray([self.mapping[text] for text in texts], dtype=float)


def test_semantic_entropy_for_two_equal_clusters() -> None:
    assert semantic_entropy([2, 2]) == pytest.approx(np.log(2))


def test_cycle_features_capture_dominant_alternative() -> None:
    question = "What is the capital of France?"
    forward = ["Rome", "Paris", "Paris", "Rome"]
    inverse = [
        "What is the capital of Italy?",
        "Which city is Italy's capital?",
        "What is the capital of Italy?",
        "What is France's capital?",
    ]
    mapping = {
        "Rome": [1.0, 0.0, 0.0],
        "Paris": [0.0, 1.0, 0.0],
        "What is the capital of Italy?": [1.0, 0.0, 0.0],
        "Which city is Italy's capital?": [0.99, 0.01, 0.0],
        "What is France's capital?": [0.0, 1.0, 0.0],
        question: [0.0, 1.0, 0.0],
    }
    config = DetectorConfig(
        forward=GenerationConfig(samples=4),
        inverse=GenerationConfig(samples=4),
        equivalence_threshold=0.9,
        smoothing=0.5,
    )
    extractor = CycleFeatureExtractor(
        FakeGenerator(forward, inverse), LookupEncoder(mapping), config
    )

    features, details = extractor.extract(question, "Rome")

    assert features.forward_entropy == pytest.approx(np.log(2))
    assert features.original_question_support == pytest.approx(1.5 / 5.0)
    assert features.competing_margin > 0
    assert features.interrogate_similarity == pytest.approx(
        np.mean([0.0, 0.01 / np.sqrt(0.99**2 + 0.01**2), 0.0, 1.0])
    )
    assert features.interrogate_risk == pytest.approx(
        1.0 - features.interrogate_similarity
    )
    assert details["matches_original"] == [False, False, False, True]


def test_missing_original_question_gets_unobserved_smoothed_category() -> None:
    question = "What is the capital of France?"
    inverse = ["What is the capital of Italy?"] * 4
    mapping = {
        "Rome": [1.0, 0.0],
        "What is the capital of Italy?": [1.0, 0.0],
        question: [0.0, 1.0],
    }
    config = DetectorConfig(
        forward=GenerationConfig(samples=2),
        inverse=GenerationConfig(samples=4),
        equivalence_threshold=0.9,
        smoothing=0.5,
    )
    extractor = CycleFeatureExtractor(
        FakeGenerator(["Rome", "Rome"], inverse), LookupEncoder(mapping), config
    )

    features, _ = extractor.extract(question, "Rome")

    assert features.original_question_support == pytest.approx(0.5 / 5.0)
    assert features.competing_margin == pytest.approx(np.log(4.5 / 0.5))

