from scripts.prepare_peft_triviaqa import clean_question, convert_rows, stratified_split


def test_conversion_inverts_correctness_and_deduplicates() -> None:
    rows = [
        {
            "fact_id": "a",
            "question": "Answer with only the short answer. Capital of France?",
            "answer": "Paris",
            "correct": 1,
        },
        {
            "fact_id": "b",
            "question": "Capital of France?",
            "answer": "Rome",
            "correct": 0,
        },
        {"fact_id": "c", "question": "Unknown?", "answer": "", "correct": -1},
    ]
    converted = convert_rows(rows)
    assert converted == [
        {
            "id": "a",
            "question": "Capital of France?",
            "answer": "Paris",
            "label": 0,
            "source_correct": 1,
            "grading_status": None,
        }
    ]


def test_stratified_split_is_deterministic_and_complete() -> None:
    rows = [
        {"id": str(index), "question": str(index), "answer": "x", "label": index % 2}
        for index in range(40)
    ]
    first = stratified_split(rows, seed=42, max_examples=None)
    second = stratified_split(rows, seed=42, max_examples=None)
    assert first == second
    assert sum(len(split) for split in first.values()) == 40
    assert {row["label"] for split in first.values() for row in split} == {0, 1}


def test_clean_question_preserves_plain_question() -> None:
    assert clean_question("Who wrote Hamlet?") == "Who wrote Hamlet?"
