from __future__ import annotations

import numpy as np
import pytest

from qa_cycle_detector.config import DetectorConfig, GenerationConfig
from qa_cycle_detector.features import CycleFeatureExtractor, semantic_entropy
from qa_cycle_detector.generation import parse_structured_answer


class FakeGenerator:
    def __init__(self, *outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompts, config):
        self.prompts.append(prompts[0])
        return [self.outputs.pop(0)]


class LookupEncoder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts):
        return np.asarray([self.mapping[text] for text in texts], dtype=float)


def test_semantic_entropy_for_two_equal_clusters() -> None:
    assert semantic_entropy([2, 2]) == pytest.approx(np.log(2))


def test_structured_answer_parser_preserves_supplied_answer() -> None:
    parsed = parse_structured_answer(
        "ANSWER || TYPE || QUALIFIER\nRome || LOCATION || capital city of Italy",
        expected_answer="Rome",
    )
    assert parsed["valid"] is True
    assert parsed["type"] == "LOCATION"

    changed = parse_structured_answer(
        "Paris || LOCATION || capital city of France", expected_answer="Rome"
    )
    assert changed["valid"] is False
    assert changed["failure_reason"] == "answer_value_changed"


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


def test_structured_format_is_used_for_inverse_generation() -> None:
    question = "What is the capital of France?"
    forward = [
        "Paris || LOCATION || capital city of France",
        "Rome || LOCATION || capital city of Italy",
    ]
    canonical = ["Rome || LOCATION || capital city of Italy"]
    inverse = ["What is the capital of Italy?", "Which city is Italy's capital?"]
    mapping = {
        forward[0]: [0.0, 1.0],
        forward[1]: [1.0, 0.0],
        inverse[0]: [1.0, 0.0],
        inverse[1]: [1.0, 0.0],
        question: [0.0, 1.0],
    }
    config = DetectorConfig(
        forward=GenerationConfig(samples=2),
        inverse=GenerationConfig(samples=2),
        answer_format="structured",
    )
    extractor = CycleFeatureExtractor(
        FakeGenerator(forward, canonical, inverse), LookupEncoder(mapping), config
    )

    _, details = extractor.extract(question, "Rome")

    assert details["canonicalization"]["valid"] is True
    assert details["inverse_answer"] == "Rome || LOCATION || capital city of Italy"
    assert details["forward_format_valid"] == [True, True]


def test_invalid_canonicalization_falls_back_to_original_answer() -> None:
    question = "What is the capital of France?"
    forward = ["Paris || LOCATION || capital city of France"]
    changed_canonical = ["Paris || LOCATION || capital city of France"]
    inverse = ["What is the capital of Italy?"]
    mapping = {
        forward[0]: [0.0, 1.0],
        inverse[0]: [1.0, 0.0],
        question: [0.0, 1.0],
    }
    generator = FakeGenerator(forward, changed_canonical, inverse)
    config = DetectorConfig(
        forward=GenerationConfig(samples=1),
        inverse=GenerationConfig(samples=1),
        answer_format="structured",
    )
    extractor = CycleFeatureExtractor(generator, LookupEncoder(mapping), config)

    _, details = extractor.extract(question, "Rome")

    assert details["canonicalization"]["failure_reason"] == "answer_value_changed"
    assert details["inverse_answer"] == "Rome"
    assert "Answer: Rome\nQuestion:" in generator.prompts[-1]


def test_sentence_condition_uses_chat_instruction_without_rewriting_labeled_answer() -> None:
    question = "What is the capital of France?"
    forward = ["The capital of France is Paris."]
    inverse = ["What is the capital of Italy?"]
    generator = FakeGenerator(forward, inverse)
    mapping = {
        forward[0]: [0.0, 1.0],
        inverse[0]: [1.0, 0.0],
        question: [0.0, 1.0],
    }
    extractor = CycleFeatureExtractor(generator, LookupEncoder(mapping))

    _, details = extractor.extract(question, "Rome")

    assert generator.prompts[0] == (
        f"Answer the question concisely in one sentence.\n\nQuestion: {question}"
    )
    assert len(generator.prompts) == 2
    assert "Answer: Rome\nQuestion:" in generator.prompts[1]
    assert question not in generator.prompts[1]
    assert details["answer_format"] == "sentence"
    assert details["inverse_answer"] == "Rome"
    assert details["canonicalization"] is None
    assert details["forward_format_valid"] is None

