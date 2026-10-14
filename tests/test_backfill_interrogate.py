from __future__ import annotations

import numpy as np
import pytest

from scripts.backfill_interrogate import augment_rows


class LookupEncoder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts):
        return np.asarray([self.mapping[text] for text in texts], dtype=float)


def test_augment_rows_reuses_saved_questions() -> None:
    rows = [
        {
            "id": "one",
            "question": "Capital of France?",
            "features": {"cycle_surprisal": 1.0},
            "samples": {
                "reconstructed_questions": [
                    "Capital of France?",
                    "Capital of Italy?",
                ]
            },
        }
    ]
    encoder = LookupEncoder(
        {
            "Capital of France?": [1.0, 0.0],
            "Capital of Italy?": [0.0, 1.0],
        }
    )

    augmented = augment_rows(rows, encoder)

    assert augmented[0]["features"]["cycle_surprisal"] == 1.0
    assert augmented[0]["features"]["interrogate_similarity"] == pytest.approx(0.5)
    assert augmented[0]["samples"]["similarities_to_original"] == [1.0, 0.0]
    assert "interrogate_similarity" not in rows[0]["features"]


def test_augment_rows_requires_saved_questions() -> None:
    with pytest.raises(ValueError, match="--keep-samples"):
        augment_rows([{"question": "Question?", "features": {}}], LookupEncoder({}))
