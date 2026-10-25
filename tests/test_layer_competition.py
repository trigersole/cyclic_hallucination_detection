from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qa_cycle_detector.layer_competition import (
    HFLayerCompetitionBackend, compare_questions, summarize_trajectories, validate_rows)
from qa_cycle_detector.layer_competition_cli import load_rows, prepare, run
from scripts.submit_layer_competition import make_plan


def rows():
    return validate_rows([{"id": "case", "question": "Original?", "label": 1,
        "answer": "Never consume me", "controls": [
            {"kind": "wording", "question": "Paraphrase?"},
            {"kind": "edited", "question": "Other detail?"},
            {"kind": "ablated", "question": "Unknown detail?"}]}])


def settings():
    return dict(samples=2, max_candidates=4, max_new_tokens=64, temperature=.9, seed=42)


def test_inputs_strip_correctness_fields_and_require_explicit_controls():
    assert set(rows()[0]) == {"id", "question", "controls"}
    with pytest.raises(ValueError, match="requires"):
        validate_rows([{"question": "Q", "controls": []}])
    with pytest.raises(ValueError, match="distinct"):
        validate_rows([{"question": "Q", "controls": [{"kind": "edited", "question": "Q"}]}])
    bundled = load_rows(Path(__file__).resolve().parents[1] / "data/layer_competition_questions.jsonl", 4)
    assert len(bundled) == 4
    assert all(len(r["controls"]) == 4 for r in bundled)


def test_trajectory_switches_ties_and_fixed_final_third():
    result = summarize_trajectories([[0, 1], [2, 1], [1, 2], [3, 1]], 0, [1, 2, 3, 4])
    assert result["winner_switches"] == 3
    assert result["stable_target_from_layer"] == 4
    assert result["target_win_fraction"] == .5
    assert result["late_target_margin"] == .5
    tied = summarize_trajectories([[1, 1]], 0, [1])
    assert tied["target_win_fraction"] == 0
    assert tied["stable_target_from_layer"] is None
    assert tied["winner_indices"] == [-1]
    assert not summarize_trajectories([[1]], 0, [1])["valid"]


def probe(values):
    return {"layer_ids": [1, 2, 3], "mean_logprob": values,
            "sum_logprob": values, "first_content_logprob": values,
            "final_readout_check_passed": True}


def test_specificity_subtracts_same_candidate_under_matched_questions():
    original = [probe([-1, -1, -1]), probe([-2, -2, -2])]
    wording = [probe([-1, -1, -1]), probe([-2, -2, -2])]
    edited = [probe([-5, -5, -5]), probe([-1, -1, -1])]
    result = compare_questions([original, wording, edited],
        [{"kind": "wording"}, {"kind": "edited"}], [0, 1], 0)
    metric = result["metrics"]["mean_logprob"]
    assert metric["edited_detail_specificity"]["target_margin"] == [5, 5, 5]
    assert metric["controls"][0]["original_minus_control"]["target_win_fraction"] == 0
    assert not compare_questions([original], [], [0], 0)["valid"]
    original[0]["final_readout_check_passed"] = False
    with pytest.raises(ValueError, match="validation"):
        compare_questions([original, edited], [{"kind": "edited"}], [0, 1], 0)


class Backend:
    metadata = {"resolved_commit": "frozen"}
    def __init__(self, fail=False, constant=False):
        self.calls, self.extracts, self.scores = 0, 0, 0
        self.fail, self.constant = fail, constant
    def generate(self, prompt, max_new_tokens, seed, temperature=None):
        self.calls += 1
        text = "A" if self.constant else ("B" if temperature else "A") if "Original?" in prompt else "C"
        return {"text": text, "truncated": False}
    def extract_answer(self, question, response, seed):
        self.extracts += 1
        return {"text": json.dumps({"span": response})}
    def score_candidate(self, question, candidate):
        self.scores += 1
        if self.fail and self.scores == 2:
            raise RuntimeError("interrupted")
        return probe([-1., -2., -3.])


def test_end_to_end_pool_provenance_labels_and_completed_resume(tmp_path):
    backend = Backend()
    report = run(tmp_path / "run", rows(), settings(), lambda: backend)
    assert backend.calls == 6 and backend.extracts == 6
    assert backend.scores == 12  # four question conditions, three unique candidates
    s = report["examples"][0]
    assert s["primary_candidate_count"] == 2 and s["expanded_candidate_count"] == 3
    assert s["original_pool"]["candidate_indices"] == [0, 1]
    assert s["expanded_pool"]["candidate_indices"] == [0, 1, 2]
    raw = json.loads((tmp_path / "run/examples/0000/raw.json").read_text())
    assert all(p["view"] > 0 for p in raw["candidates"][2]["sources"])
    assert "Never consume me" not in json.dumps(raw)
    from scripts.summarize_layer_competition import summarize
    compact = summarize(tmp_path / "run")
    assert "original_pool" in compact and "detail late margin=" in compact
    def fail():
        raise AssertionError("completed resume must not load model")
    assert run(tmp_path / "run", rows(), settings(), fail) == report
    with pytest.raises(ValueError, match="settings changed"):
        run(tmp_path / "run", rows(), {**settings(), "seed": 43}, fail)


def test_interrupted_resume_reuses_generations_extractions_and_probes(tmp_path):
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interrupted"):
        run(root, rows(), settings(), lambda: Backend(fail=True))
    assert not (root / ".stability.lock").exists()
    resumed = Backend()
    run(root, rows(), settings(), lambda: resumed)
    assert resumed.calls == resumed.extracts == 0
    assert resumed.scores == 11


def test_no_alternatives_is_missing_not_certain_and_old_runs_protected(tmp_path):
    backend = Backend(constant=True)
    report = run(tmp_path / "new", rows(), settings(), lambda: backend)
    assert report["valid_original_pools"] == report["valid_expanded_pools"] == 0
    assert backend.scores == 0
    old = tmp_path / "old"
    old.mkdir()
    (old / "STABILITY_RUN.json").write_text("{}")
    with pytest.raises(ValueError, match="separate"):
        prepare(old / "nested", {})
    assert (old / "STABILITY_RUN.json").read_text() == "{}"


def test_submission_is_separate_and_reuses_cache_environment(tmp_path):
    args = SimpleNamespace(python="python", model=None, run_root=None,
        input=Path("data/layer_competition_questions.jsonl"), examples=4,
        allow_downloads=False, revision="fixed", partition="MGPU-TC2", account=None)
    plan = make_plan(args, {"SCRATCH": str(tmp_path), "HF_HOME": "shared-cache", "MODEL_ID": "cached-model"})
    assert plan["root"].parent == tmp_path / "qa-layer-results"
    assert plan["env"]["HF_HOME"] == "shared-cache"
    assert "--local-files-only" in plan["command"]
    assert "cached-model" in plan["command"]
    assert "--job-name=layer-competition" in plan["submit"]
    assert not plan["root"].exists()


class Tensor(np.ndarray):
    def __new__(cls, data): return np.asarray(data).view(cls)
    @property
    def device(self): return "cpu"
    def detach(self): return self
    def cpu(self): return self
    def float(self): return self
    def numpy(self): return np.asarray(self)
    def to(self, device): return self
    def new_tensor(self, x): return Tensor(x)
    def gather(self, dim, ids): return Tensor(np.take_along_axis(self, ids, axis=dim))
    def squeeze(self, dim): return Tensor(np.squeeze(np.asarray(self), axis=dim))


def real_backend_fixture(fail=False):
    blocks = []
    for _ in range(3):
        block = SimpleNamespace(hook=None)
        def register(hook, b=block):
            b.hook = hook
            return SimpleNamespace(remove=lambda: setattr(b, "hook", None))
        block.register_forward_hook = register
        blocks.append(block)
    class Norm:
        def parameters(self): return iter([Tensor([1])])
        def __call__(self, state): return Tensor(state * 2)
    class Head:
        weight = Tensor([[1, 0], [0, 1], [1, 1]])
        def __call__(self, state): return Tensor(state @ self.weight.T)
    class Core:
        layers, norm = blocks, Norm()
        def __call__(self, **kw):
            ids = kw["input_ids"]
            assert ids.tolist() == [[9, 8, 0, 1]]
            hidden = Tensor(np.stack([ids, np.ones_like(ids)], axis=-1))
            for i, block in enumerate(blocks):
                hidden = Tensor(hidden + i + 1)
                block.hook(block, (), (hidden,))
                if fail:
                    raise RuntimeError("model failure")
            return SimpleNamespace(last_hidden_state=self.norm(hidden))
    backend = HFLayerCompetitionBackend.__new__(HFLayerCompetitionBackend)
    backend.model = SimpleNamespace(model=Core(), config=SimpleNamespace(max_position_embeddings=100),
                                    get_output_embeddings=lambda: Head())
    backend.torch = SimpleNamespace(inference_mode=nullcontext, ones_like=lambda x: Tensor(np.ones_like(x)),
        tensor=lambda x, device=None: Tensor(x), cat=lambda xs, dim: Tensor(np.concatenate(xs, axis=dim)),
        log_softmax=lambda x, dim: Tensor(x - np.logaddexp.reduce(x, axis=dim, keepdims=True)))
    backend.encode = lambda prompt: {"input_ids": Tensor([[9, 8]]), "attention_mask": Tensor([[1, 1]])}
    backend.tokenizer = SimpleNamespace(encode=lambda text, **kw: [0, 1, 2],
        decode=lambda ids, **kw: "".join({0: " ", 1: "Foo", 2: "bar"}[i] for i in ids))
    return backend, blocks


def test_actual_readout_path_positions_normalization_and_whitespace_exclusion():
    backend, blocks = real_backend_fixture()
    result = backend.score_candidate("Q", "Foobar")
    assert result["first_content_index"] == 1
    assert result["final_readout_max_error"] == 0
    assert result["layer_ids"] == [1, 2, 3]
    logps = np.asarray(result["token_logprobs"])
    np.testing.assert_allclose(result["mean_logprob"], logps[:, 1:].mean(axis=1))
    np.testing.assert_allclose(result["first_content_logprob"], logps[:, 1])
    assert all(b.hook is None for b in blocks)


def test_actual_readout_removes_all_hooks_on_model_error():
    backend, blocks = real_backend_fixture(fail=True)
    with pytest.raises(RuntimeError, match="model failure"):
        backend.score_candidate("Q", "Foobar")
    assert all(b.hook is None for b in blocks)
