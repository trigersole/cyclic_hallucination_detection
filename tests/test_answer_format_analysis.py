from scripts.analyze_answer_format import summarize


def test_structured_answer_summary_reports_compliance_and_overlap() -> None:
    rows = [
        {
            "question": "What is the capital of France?",
            "samples": {
                "original_answer": "Paris",
                "forward_format_valid": [True, False],
                "canonicalization": {
                    "valid": True,
                    "qualifier": "capital city of France",
                },
            },
        },
        {
            "question": "Who wrote Hamlet?",
            "samples": {
                "original_answer": "Shakespeare",
                "forward_format_valid": [True, True],
                "canonicalization": {
                    "valid": False,
                    "failure_reason": "answer_value_changed",
                },
            },
        },
    ]

    report = summarize(rows)

    assert report["canonicalization_valid_rate"] == 0.5
    assert report["forward_format_valid_rate"] == 0.75
    assert report["canonicalization_failures"] == {"answer_value_changed": 1}
    assert report["possible_leakage_rate_at_0.5"] == 1.0
