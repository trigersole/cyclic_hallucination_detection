from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling settings for one generation direction."""

    samples: int
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be at least 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the reference-free detector."""

    forward: GenerationConfig = GenerationConfig(samples=10)
    inverse: GenerationConfig = GenerationConfig(samples=10)
    canonicalization: GenerationConfig = GenerationConfig(
        samples=1, temperature=0.2, max_new_tokens=64
    )
    answer_format: str = "sentence"
    equivalence_threshold: float = 0.82
    smoothing: float = 0.5
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.answer_format not in {"short", "sentence", "structured"}:
            raise ValueError("answer_format must be 'short', 'sentence', or 'structured'")
        if not -1 <= self.equivalence_threshold <= 1:
            raise ValueError("equivalence_threshold must be in [-1, 1]")
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

