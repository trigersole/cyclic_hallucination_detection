from qa_cycle_detector.features import CycleFeatures
from qa_cycle_detector.model import HallucinationClassifier, evaluate


def row(
    forward: float, cycle: float, margin: float, similarity: float = 0.5
) -> CycleFeatures:
    return CycleFeatures(forward, cycle, margin, 0.0, 0.0, 0, 0, similarity)


def test_classifier_fits_scores_and_round_trips(tmp_path) -> None:
    features = [
        row(0.0, 0.1, -2.0),
        row(0.1, 0.2, -1.0),
        row(1.0, 2.0, 1.0),
        row(1.2, 3.0, 2.0),
    ]
    labels = [0, 0, 1, 1]
    classifier = HallucinationClassifier.fit(features, labels)
    scores = classifier.predict_proba(features)
    metrics = evaluate(labels, scores)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["aurc"] >= 0.0

    path = tmp_path / "detector.json"
    classifier.save(path)
    loaded = HallucinationClassifier.load(path)
    assert loaded.predict_proba(features).tolist() == scores.tolist()


def test_feature_subset_is_persisted(tmp_path) -> None:
    features = [row(0.0, 2.0, 1.0), row(0.1, 1.0, -1.0), row(1.0, 0.1, 0.0), row(1.1, 0.2, 0.0)]
    classifier = HallucinationClassifier.fit(
        features, [0, 0, 1, 1], feature_names=["forward_entropy"]
    )
    path = tmp_path / "forward.json"
    classifier.save(path)
    loaded = HallucinationClassifier.load(path)
    assert loaded.feature_names == ["forward_entropy"]


def test_interrogate_risk_can_be_calibrated(tmp_path) -> None:
    features = [
        row(0.0, 0.0, 0.0, 0.95),
        row(0.0, 0.0, 0.0, 0.85),
        row(0.0, 0.0, 0.0, 0.20),
        row(0.0, 0.0, 0.0, 0.10),
    ]
    classifier = HallucinationClassifier.fit(
        features, [0, 0, 1, 1], feature_names=["interrogate_risk"]
    )
    assert evaluate([0, 0, 1, 1], classifier.predict_proba(features))["auroc"] == 1.0
