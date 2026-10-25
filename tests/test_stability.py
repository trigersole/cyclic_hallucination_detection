from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from qa_cycle_detector.stability import (HFStabilityBackend, INSTRUCTIONS, MODES, LastPromptIntervention,
                                         PilotConfig, first_token_diagnostics, perturbation, prompts, random_basis,
                                         semantic_summary, wording_basis)
from qa_cycle_detector.stability_cli import (collect_example, prepare_run, read_json,
                                             run_pilot, score_example, selected_inputs, unlabeled_rows)


class FakeBackend:
    metadata = {"model": "fake", "layer": 1}

    def __init__(self, fail_at=None):
        self.captures = []
        self.calls = []
        self.fail_at = fail_at

    def capture(self, prompt):
        self.captures.append(prompt)
        i = next(i for i, instruction in enumerate(INSTRUCTIONS) if prompt.startswith(instruction))
        value = np.zeros(8)
        value[-1] = 10
        if i:
            value[i - 1] = .1
        return value

    def generate(self, prompt, max_new_tokens, seed, delta=None, temperature=None,
                 diagnostics=False, reference_logprobs=None):
        self.calls.append({"prompt": prompt, "seed": seed, "delta": delta, "temperature": temperature})
        if len(self.calls) == self.fail_at:
            raise RuntimeError("interrupted")
        changed = delta is not None and np.linalg.norm(delta[3:]) > .06
        result = {"text": "Bees" if changed else "Wasps", "input_tokens": 12, "output_tokens": 2,
                "mean_token_entropy": .2, "mean_token_logprob": -.1,
                "truncated": False, "intervention_calls": 1}
        norm = float(np.linalg.norm(delta)) if delta is not None else 0.
        result["intervention_measurement"] = ({"state_norm": 10., "requested_norm": norm,
            "applied_norm": norm, "requested_relative_norm": norm / 10, "applied_relative_norm": norm / 10,
            "norm_retention": 1. if norm else None, "nonzero_coordinates": 1, "dtype": "fake"}
            if delta is not None else None)
        if diagnostics:
            first, logp = first_token_diagnostics([3. - norm, norm], reference_logprobs)
            result.update(first_token=first, _first_logprobs=logp)
        return result


class FakeEncoder:
    def encode(self, texts):
        return np.asarray([[0., 1.] if t == "Bees" else [1., 0.] for t in texts])


def source():
    return {"id": "wasps", "question": "Which insects live in a vespiary?", "answer": "Wasps"}


def small_config():
    return PilotConfig(rank=3, directions=2, amplitudes=(.01, .02), baseline_samples=2)


def test_wording_views_preserve_question_exactly_and_never_receive_answer():
    question = "Which insects live in a vespiary?"
    for view in prompts(question):
        assert view.split("Question: ")[1] == question + "\nAnswer:"
        assert "Wasps" not in view


def test_labels_are_discarded_and_do_not_change_canonical_inputs():
    a = unlabeled_rows([source() | {"label": 0, "source_correct": 1, "features": {"risk": 0}}], 20)
    b = unlabeled_rows([source() | {"label": 1, "source_correct": 0, "features": {"risk": 1}}], 20)
    assert a == b == [source()]
    assert unlabeled_rows([source()], 1) == a
    with pytest.raises(ValueError, match="duplicate"):
        unlabeled_rows([source(), source()], 2)


def test_pilot_streams_only_selected_prefix_and_ignores_labels(tmp_path):
    shard = tmp_path / "shard_00000.jsonl"
    shard.write_text(json.dumps(source() | {"label": 1}) + "\nnot JSON beyond selected prefix\n", encoding="utf-8-sig")
    assert selected_inputs(tmp_path, 1) == [source()]
    with pytest.raises(ValueError, match="invalid JSON"):
        selected_inputs(tmp_path, 2)


def test_fresh_question_only_run_generates_its_own_answer(tmp_path):
    question = {"id": "fresh", "question": "Which insects live in a vespiary?", "label": 1}
    rows = unlabeled_rows([question], 1)
    assert rows == [{"id": "fresh", "question": question["question"]}]
    report = run_pilot(tmp_path, rows, small_config(), {"model": "fake"}, FakeBackend, FakeEncoder)
    score = read_json(tmp_path / "examples" / "00000" / "scores.json")
    assert score["answer"] == "Wasps"
    assert score["answer_source"] == "generated_anchor"
    assert score["supplied_answer_score"] is not None
    assert report["supplied_anchor_disagreements"] == 0


def test_empty_outputs_in_question_only_run_are_unusable_not_an_error(tmp_path):
    raw = collect_example(tmp_path, {"question": "An example question?"}, FakeBackend(), small_config())
    raw["anchor"]["text"] = ""
    for unit in raw["units"]:
        unit["text"] = ""
    class NoEncoderNeeded:
        def encode(self, texts):
            raise AssertionError("there is no nonempty text to encode")
    score = score_example(raw, NoEncoderNeeded(), small_config())
    assert score["answer"] == ""
    assert score["scores"]["wording_corrected"] is None
    assert score["supplied_answer_score"] is None


def test_subspace_and_perturbations_have_matched_norms_and_dimensions():
    backend = FakeBackend()
    states = [backend.capture(p) for p in prompts(source()["question"])]
    basis = wording_basis(states, 3)
    control = random_basis(8, 3, 1)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3))
    np.testing.assert_allclose(control.T @ control, np.eye(3), atol=1e-12)
    for mode in MODES:
        delta = perturbation(mode, basis, control, 10, .02, 10)
        assert np.linalg.norm(delta) == pytest.approx(.2)
        np.testing.assert_array_equal(delta, perturbation(mode, basis, control, 10, .02, 10))
        if mode == "wording":
            np.testing.assert_allclose(delta, basis @ (basis.T @ delta), atol=1e-12)
        if mode == "orthogonal":
            np.testing.assert_allclose(basis.T @ delta, 0, atol=1e-12)
        if mode == "random_wording":
            np.testing.assert_allclose(delta, control @ (control.T @ delta), atol=1e-12)
        if mode == "random_orthogonal":
            np.testing.assert_allclose(control.T @ delta, 0, atol=1e-12)


def test_degenerate_views_fail_instead_of_claiming_a_wording_subspace():
    with pytest.raises(ValueError, match="no measurable displacement"):
        wording_basis(np.ones((len(INSTRUCTIONS), 8)), 3)
    for kwargs in ({"amplitudes": (0.,)}, {"amplitudes": (.1, .01)}, {"directions": 0}):
        with pytest.raises(ValueError):
            PilotConfig(**kwargs)


def test_semantic_summary_counts_paraphrases_and_marks_empty_outputs():
    texts = ["Wasps", "The insects are wasps", "Bees", ""]
    vectors = np.array([[1., 0], [1., 0], [0, 1.], [0, 0]])
    summary = semantic_summary(texts, vectors, vectors[0], vectors[0], .82)
    assert summary["cluster_count"] == 2
    assert summary["anchor_flip_rate"] == pytest.approx(1 / 3)
    assert summary["invalid_rate"] == .25
    assert summary["semantic_entropy"] == pytest.approx(-2 / 3 * np.log(2 / 3) - 1 / 3 * np.log(1 / 3))


def test_resume_never_repeats_completed_interventions(tmp_path):
    config = small_config()
    failed = FakeBackend(fail_at=5)
    with pytest.raises(RuntimeError, match="interrupted"):
        collect_example(tmp_path, source(), failed, config)
    assert len(failed.captures) == len(INSTRUCTIONS)
    resumed = FakeBackend()
    raw = collect_example(tmp_path, source(), resumed, config)
    expected = 1 + config.baseline_samples + len(MODES) * len(config.amplitudes) * config.directions
    assert len(resumed.calls) == expected - 4
    assert not resumed.captures
    for amplitude in config.amplitudes:
        for mode in MODES:
            units = [u for u in raw["units"] if u["mode"] == mode and u["amplitude"] == amplitude]
            assert len(units) == config.directions
            for unit in units:
                assert unit["delta_norm"] == pytest.approx(amplitude * raw["state"]["state_norm"])
    assert all(c["prompt"] == prompts(source()["question"])[0] for c in resumed.calls)


def test_scoring_keeps_truncation_and_invalidity_separate_from_semantic_flips(tmp_path):
    config = small_config()
    raw = collect_example(tmp_path, source(), FakeBackend(), config)
    score = score_example(raw, FakeEncoder(), config)
    assert score["scores"]["flip_auc"]["wording"] == 0
    assert score["scores"]["wording_corrected"] > 0
    assert score["first_tested_flip_amplitude"]["wording"] is None
    assert score["supplied_anchor_equivalent"]
    raw["units"][config.baseline_samples]["truncated"] = True  # First isotropic sample.
    score = score_example(raw, FakeEncoder(), config)
    assert not score["usable_conditions"]["isotropic"]
    assert score["scores"]["flip_auc"]["isotropic"] is None
    for unit in raw["units"]:
        if unit["mode"] == "orthogonal":
            unit["text"] = ""
    score = score_example(raw, FakeEncoder(), config)
    assert score["scores"]["wording_corrected"] is None
    assert score["conditions"]["orthogonal"][0]["invalid_rate"] == 1


def test_supplied_answer_disagreement_is_not_silently_scored_as_original_answer(tmp_path):
    config = small_config()
    raw = collect_example(tmp_path, source() | {"answer": "Bees"}, FakeBackend(), config)
    score = score_example(raw, FakeEncoder(), config)
    assert not score["supplied_anchor_equivalent"]
    assert score["supplied_answer_score"] is None
    assert score["scores"]["wording_corrected"] is not None  # Question fragility is still available.


def test_end_to_end_pilot_is_label_free_resumable_and_preserves_legacy(tmp_path):
    legacy = tmp_path / "existing.jsonl"
    legacy.write_text(json.dumps(source() | {"label": 0}) + "\n", encoding="utf-8")
    before = legacy.read_bytes()
    root = tmp_path / "new_pilot"
    rows = unlabeled_rows([json.loads(legacy.read_text())], 20)
    backend = FakeBackend()
    report = run_pilot(root, rows, small_config(), {"model": "fake"}, lambda: backend, FakeEncoder)
    assert report["examples"] == 1
    assert report["usable_examples"]["orthogonal"] == 1
    assert legacy.read_bytes() == before
    assert "label" not in read_json(root / "STABILITY_RUN.json")["inputs"][0]
    assert "label" not in read_json(root / "examples" / "00000" / "raw.json")["input"]
    def fail():
        raise AssertionError("complete pilot must not load models")
    again = run_pilot(root, rows, small_config(), {"model": "fake"}, fail, fail)
    assert again == report
    with pytest.raises(ValueError, match="settings changed"):
        run_pilot(root, rows, small_config(), {"model": "different"}, fail, fail)
    with pytest.raises(ValueError, match="must be empty"):
        prepare_run(tmp_path, {"new": True})


class Tensor(np.ndarray):
    """Tiny Torch-like tensor for testing the actual intervention/cache control flow."""
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def clone(self): return Tensor(self.copy())
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)
    def new_tensor(self, value): return Tensor(value)
    def new_ones(self, shape): return Tensor(np.ones(shape))
    def to(self, device): return self
    def exp(self): return Tensor(np.exp(self))
    def argmax(self, dim=None, keepdim=False):
        return Tensor(np.argmax(np.asarray(self), axis=dim, keepdims=keepdim))


def test_hook_preserves_tuple_cache_and_only_changes_last_prompt_position():
    hidden = Tensor(np.zeros((1, 3, 4)))
    sentinel = object()
    hook = LastPromptIntervention(delta=np.ones(4), capture=True)
    changed, retained = hook(None, None, (hidden, sentinel))
    assert retained is sentinel
    np.testing.assert_array_equal(hidden, 0)
    np.testing.assert_array_equal(changed[0, :2], 0)
    np.testing.assert_array_equal(changed[0, -1], 1)
    np.testing.assert_array_equal(hook.state, 0)
    with pytest.raises(RuntimeError, match="one prefill"):
        hook(None, None, hidden)


def test_real_backend_decode_loop_removes_intervention_before_cached_steps():
    class Block:
        hook = None
        def register_forward_hook(self, hook):
            self.hook = hook
            return SimpleNamespace(remove=lambda: setattr(self, "hook", None))
    block = Block()
    model_calls = []
    def model(**kw):
        prefill = "past_key_values" not in kw
        model_calls.append((prefill, block.hook is not None))
        hidden = Tensor(np.zeros((1, kw["input_ids"].shape[1], 4)))
        if block.hook:
            hidden = block.hook(block, (), hidden)
        logits = Tensor([[[-3., 3., -3.]]] if prefill else [[[-3., -3., 3.]]])
        return SimpleNamespace(logits=logits, past_key_values="cache")
    model.generation_config = SimpleNamespace(eos_token_id=2)
    model.config = SimpleNamespace(max_position_embeddings=100)
    torch = SimpleNamespace(manual_seed=lambda seed: None,
                            cuda=SimpleNamespace(is_available=lambda: False),
                            inference_mode=nullcontext,
                            log_softmax=lambda x, dim: Tensor(x - np.log(np.exp(x).sum(axis=dim, keepdims=True))),
                            cat=lambda xs, dim: Tensor(np.concatenate(xs, axis=dim)))
    backend = HFStabilityBackend.__new__(HFStabilityBackend)
    backend.torch, backend.model, backend.block, backend.device = torch, model, block, "cpu"
    backend.tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda tokens, **kw: "Wasps")
    backend.encode = lambda prompt: {"input_ids": Tensor([[1, 2, 3]]), "attention_mask": Tensor([[1, 1, 1]])}
    result = backend.generate("question", 3, 42, delta=np.ones(4))
    assert result["text"] == "Wasps"
    assert not result["truncated"]
    assert result["output_tokens"] == 2
    assert model_calls == [(True, True), (False, False)]
    assert block.hook is None


def test_first_token_metrics_detect_changes_without_a_greedy_token_flip():
    anchor, reference = first_token_diagnostics([3., 1., 0.])
    same, _ = first_token_diagnostics([3., 1., 0.], reference)
    moved, _ = first_token_diagnostics([2.8, 1.1, 0.], reference)
    assert same["js_divergence"] == pytest.approx(0, abs=1e-12)
    assert moved["js_divergence"] > 0
    assert not moved["top_token_changed"]
    assert moved["anchor_top_probability_drop"] > 0
    assert moved["top2_logit_margin"] < anchor["top2_logit_margin"]
    assert moved["kl_anchor_to_perturbed"] > 0
    flipped, _ = first_token_diagnostics([0., 4., 0.], reference)
    assert flipped["top_token_changed"]
    assert 0 <= flipped["js_divergence"] <= np.log(2)


def test_applied_norm_detects_perturbation_lost_to_low_precision():
    hidden = Tensor(np.full((1, 2, 4), 10000., dtype=np.float16))
    hook = LastPromptIntervention(delta=np.ones(4))
    edited = hook(None, None, hidden)
    np.testing.assert_array_equal(edited, hidden)
    assert hook.measurement["requested_norm"] == 2
    assert hook.measurement["applied_norm"] == 0
    assert hook.measurement["norm_retention"] == 0
    assert hook.measurement["nonzero_coordinates"] == 0


def test_diagnostic_run_records_continuous_changes_with_identical_answers(tmp_path):
    config = replace(small_config(), diagnostics=True)
    raw = collect_example(tmp_path, {"question": source()["question"]}, FakeBackend(), config)
    for unit in raw["units"]:
        unit["text"] = raw["anchor"]["text"]  # Same text, different first-token distributions.
    score = score_example(raw, FakeEncoder(), config)
    assert score["zero_control_passed"]
    assert score["scores"]["flip_auc"]["isotropic"] == 0
    assert score["scores"]["first_token_js_auc"]["isotropic"] > 0
    summary = score["conditions"]["isotropic"][0]["first_token"]
    assert summary["mean_applied_relative_norm"] == pytest.approx(config.amplitudes[0])
    assert summary["min_norm_retention"] == 1
    assert (tmp_path / "generations" / "anchor.first_logprobs.npy").exists()
    assert "_first_logprobs" not in read_json(tmp_path / "generations" / "anchor.json")
    zero = next(u for u in raw["units"] if u["mode"] == "zero_control")
    zero["first_token"]["js_divergence"] = .1
    assert not score_example(raw, FakeEncoder(), config)["zero_control_passed"]


def test_diagnostic_resume_reuses_anchor_distribution_and_zero_control(tmp_path):
    config = replace(small_config(), diagnostics=True)
    failing = FakeBackend(fail_at=3)
    with pytest.raises(RuntimeError, match="interrupted"):
        collect_example(tmp_path, {"question": "A new question?"}, failing, config)
    saved = (tmp_path / "generations" / "anchor.first_logprobs.npy").read_bytes()
    resumed = FakeBackend()
    raw = collect_example(tmp_path, {"question": "A new question?"}, resumed, config)
    assert saved == (tmp_path / "generations" / "anchor.first_logprobs.npy").read_bytes()
    assert len(resumed.calls) == config.baseline_samples + len(MODES) * len(config.amplitudes) * config.directions
    assert score_example(raw, FakeEncoder(), config)["zero_control_passed"]
