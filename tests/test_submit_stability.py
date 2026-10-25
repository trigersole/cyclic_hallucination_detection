from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace
import sys

import pytest

from scripts import submit_stability


def test_no_input_argument_selects_fresh_unlabeled_questions():
    from qa_cycle_detector.stability_cli import selected_inputs
    args = submit_stability.parser().parse_args([])
    rows = selected_inputs(submit_stability.make_plan(args, {})["input"], 20)
    assert len(rows) == 20
    assert len({row["id"] for row in rows}) == 20
    assert all(set(row) == {"id", "question"} for row in rows)


def test_diagnostic_submission_uses_four_questions_and_preserves_preset_on_resume():
    args = submit_stability.parser().parse_args(["--diagnostics"])
    plan = submit_stability.make_plan(args, {})
    assert plan["examples"] == 4
    assert plan["environment"]["STABILITY_DIAGNOSTICS"] == "1"
    assert plan["input"].endswith("stability_diagnostic_questions.jsonl")
    assert "--diagnostics" in plan["validation"]
    assert plan["root"].name.startswith("diagnostics-4-")


def test_plan_reuses_environment_and_only_requires_input(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('{}\n', encoding="utf-8")
    args = submit_stability.parser().parse_args([str(source)])
    env = {"HALOSCOPE_PYTHON": sys.executable, "HF_HOME": "/existing/shared/cache",
           "SCRATCH": str(tmp_path), "SLURM_ACCOUNT": "existing-account", "MODEL_ID": "existing-model"}
    plan = submit_stability.make_plan(args, env)
    assert plan["python"] == sys.executable
    assert plan["environment"]["HF_HOME"] == "/existing/shared/cache"
    assert plan["environment"]["STABILITY_MODEL"] == "existing-model"
    assert plan["environment"]["STABILITY_MAX_EXAMPLES"] == "2"
    assert plan["environment"]["STABILITY_LOCAL_FILES_ONLY"] == "1"
    assert "--account=existing-account" in plan["command"]
    assert plan["root"].parent == tmp_path / "qa-stability-results"
    assert not plan["root"].exists()
    assert "STABILITY_INPUT" not in env  # Caller environment not mutated.


def test_dry_run_performs_only_input_validation_and_creates_no_directories(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    source.write_text('{}\n', encoding="utf-8")
    root = tmp_path / "new"
    args = submit_stability.parser().parse_args([str(source), "--run-root", str(root), "--dry-run"])
    calls = []
    monkeypatch.setattr(submit_stability.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    submit_stability.submit(args)
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"
    assert not root.exists()


def test_submission_passes_isolated_defaults_and_prints_job(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.jsonl"
    source.write_text('{}\n', encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(submit_stability, "REPOSITORY", repo)
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.setenv("STABILITY_PYTHON", sys.executable)
    monkeypatch.setattr(submit_stability.shutil, "which", lambda cmd: "/bin/sbatch")
    calls = []
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout="12345;cluster\n")
    monkeypatch.setattr(submit_stability.subprocess, "run", run)
    args = submit_stability.parser().parse_args([str(source), "--examples", "20", "--account", "research"])
    submit_stability.submit(args)
    assert len(calls) == 4  # input budget, imports, local snapshots, submission
    command, kwargs = calls[-1]
    assert command[0] == "sbatch" and "--account=research" in command
    assert kwargs["env"]["STABILITY_MAX_EXAMPLES"] == "20"
    assert kwargs["env"]["STABILITY_LOCAL_FILES_ONLY"] == "1"
    assert Path(kwargs["env"]["STABILITY_RUN_ROOT"]).is_dir()
    assert kwargs["cwd"] == repo
    assert "squeue -j 12345" in capsys.readouterr().out


def test_rejects_legacy_root_before_submission(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('{}\n', encoding="utf-8")
    args = submit_stability.parser().parse_args([str(source), "--run-root", str(tmp_path)])
    with pytest.raises(ValueError, match="non-pilot"):
        submit_stability.make_plan(args, {})


def test_rescore_submission_reads_saved_run_and_uses_new_output(tmp_path):
    source = tmp_path / "source"
    (source / "examples" / "00000").mkdir(parents=True)
    (source / "STABILITY_RUN.json").write_text(json.dumps({"settings": {"model": "source-model"}}), encoding="utf-8")
    (source / "examples" / "00000" / "raw.json").write_text('{}', encoding="utf-8")
    args = submit_stability.parser().parse_args(["--rescore-from", str(source)])
    plan = submit_stability.make_plan(args, {"SCRATCH": str(tmp_path)})
    assert plan["examples"] == 1
    assert plan["environment"]["STABILITY_MODEL"] == "source-model"
    assert plan["environment"]["STABILITY_RESCORE_FROM"] == str(source.resolve())
    assert plan["environment"]["STABILITY_DIAGNOSTICS"] == "0"
    assert "--rescore-from" in plan["validation"]
    assert plan["root"].name.startswith("rescore-")
    args.run_root = source
    with pytest.raises(ValueError, match="separate"):
        submit_stability.make_plan(args, {})
