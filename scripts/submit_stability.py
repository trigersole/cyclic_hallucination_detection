#!/usr/bin/env python3
"""Submit a fresh Slurm pilot: python scripts/submit_stability.py [INPUT]."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid


REPOSITORY = Path(__file__).resolve().parents[1]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("input", nargs="?", default=str(REPOSITORY / "data" / "stability_pilot_questions.jsonl"),
                        help="optional question-only/QA JSONL; defaults to the new bundled question set")
    result.add_argument("--examples", type=int, default=2, help="default: 2-example smoke test; use 20 for pilot")
    result.add_argument("--run-root", type=Path, help="optional explicit directory, including for resuming")
    result.add_argument("--model", help="default: STABILITY_MODEL, MODEL_ID, or existing Llama-2 model")
    result.add_argument("--python", dest="python_bin", help="override automatically selected model environment")
    result.add_argument("--account", help="optional Slurm account; defaults to SLURM_ACCOUNT")
    result.add_argument("--partition", help="optional override of launcher partition MGPU-TC2")
    result.add_argument("--allow-downloads", action="store_true", help="allow fetching models instead of using only cached files")
    result.add_argument("--dry-run", action="store_true", help="validate inputs and show submission without creating directories/jobs")
    return result


def make_plan(args, environ):
    if args.examples < 1:
        raise ValueError("--examples must be positive")
    source = Path(args.input).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"input does not exist: {source}")
    env = dict(environ)
    halo = Path(env.get("HALOSCOPE_DIR", str(Path.home() / "LLM_Haloscope")))
    candidate = halo / ".venv" / "bin" / "python"
    python_bin = (args.python_bin or env.get("STABILITY_PYTHON") or env.get("HALOSCOPE_PYTHON")
                  or (str(candidate) if candidate.is_file() else sys.executable))
    if not (Path(python_bin).is_file() or shutil.which(python_bin)):
        raise ValueError(f"Python not found: {python_bin}; use --python /path/to/existing/venv/bin/python")
    model = args.model or env.get("STABILITY_MODEL") or env.get("MODEL_ID") or "meta-llama/Llama-2-7b-chat-hf"
    automatic = args.run_root is None
    if automatic:
        base = Path(env.get("SCRATCH") or str(Path.home())) / "qa-stability-results"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = base / f"pilot-{args.examples}-{stamp}-{uuid.uuid4().hex[:8]}"
    else:
        root = args.run_root.expanduser().resolve()
        if root.exists() and any(root.iterdir()) and not (root / "STABILITY_RUN.json").is_file():
            raise ValueError("--run-root is an existing non-pilot directory; choose a new directory")
    for parent in (root, *root.parents):
        if (parent / "_RUN.json").exists() or (parent / "TRIPLET_RUN.json").exists():
            raise ValueError("choose a pilot directory outside legacy/triplet outputs")
    overlay = env.get("CYCLE_OVERLAY", str(Path(env.get("SCRATCH") or str(Path.home())) / "qa-cycle-overlay"))
    env.update(STABILITY_INPUT=str(source), STABILITY_RUN_ROOT=str(root), STABILITY_MODEL=model,
               STABILITY_MAX_EXAMPLES=str(args.examples), STABILITY_PYTHON=python_bin,
               STABILITY_LOCAL_FILES_ONLY="0" if args.allow_downloads else "1", CYCLE_OVERLAY=overlay)
    # Preserve HF_HOME/HF_HUB_CACHE and all existing settings; do not invent a new cache.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPOSITORY), overlay, env.get("PYTHONPATH")]))
    logs = REPOSITORY / "slurm" / "logs"
    command = ["sbatch", "--parsable", "--export=ALL",
               f"--output={logs / 'wording-stability-%j.out'}",
               f"--error={logs / 'wording-stability-%j.err'}"]
    account = args.account or env.get("SLURM_ACCOUNT")
    if account:
        command.append(f"--account={account}")
    if args.partition:
        command.append(f"--partition={args.partition}")
    command.append(str(REPOSITORY / "slurm" / "stability_pilot.sbatch"))
    validation = [python_bin, "-m", "qa_cycle_detector.stability_cli", "--input", str(source),
                  "--run-root", str(root), "--max-examples", str(args.examples), "--dry-run"]
    return {"environment": env, "command": command, "validation": validation,
            "python": python_bin, "root": root, "automatic": automatic, "logs": logs}


def submit(args):
    plan = make_plan(args, os.environ)
    env = plan["environment"]
    # Dry-run validation loads no models and needs no GPU/login-node CUDA access.
    subprocess.run(plan["validation"], cwd=REPOSITORY, env=env, check=True)
    if args.dry_run:
        print("Submission:", shlex.join(plan["command"]))
        print("Python:", plan["python"])
        print("Model:", env["STABILITY_MODEL"])
        print("Results:", plan["root"])
        print("Cache:", env.get("HF_HOME") or env.get("HF_HUB_CACHE") or "existing Hugging Face defaults")
        print("No job submitted; no output directories created.")
        return
    if not shutil.which("sbatch"):
        raise ValueError("sbatch is unavailable; run this command on your cluster login node")
    subprocess.run([plan["python"], "-c", "import torch, transformers, sentence_transformers; "
                    "print('Model environment OK (GPU checked inside the Slurm job).')"],
                   cwd=REPOSITORY, env=env, check=True)
    # Resolve required cached models on the login node, without loading their weights.
    if not args.allow_downloads:
        subprocess.run([plan["python"], "-c",
                        "from pathlib import Path; from huggingface_hub import snapshot_download; "
                        "import os; "
                        "[(snapshot_download(repo_id=m, revision=r, local_files_only=True) "
                        "if not Path(m).is_dir() else None) for m,r in "
                        "[(os.environ['STABILITY_MODEL'], os.environ.get('STABILITY_REVISION')), "
                        "(os.environ.get('STABILITY_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2'), None)]]; "
                        "print('Cached model snapshots found.')"],
                       cwd=REPOSITORY, env=env, check=True)
    if plan["automatic"]:
        plan["root"].mkdir(parents=True, exist_ok=False)
    plan["logs"].mkdir(parents=True, exist_ok=True)
    result = subprocess.run(plan["command"], cwd=REPOSITORY, env=env, check=True,
                            text=True, capture_output=True)
    job = result.stdout.strip().split(";")[0]
    print(f"Submitted {args.examples}-example stability pilot: {job}")
    print(f"Results: {plan['root']}")
    print(f"Monitor: squeue -j {job}")
    print(f"Log: tail -f {plan['logs'] / f'wording-stability-{job}.out'}")
    print(f"Report: {plan['root'] / 'report.json'}")
    print(f"Resume after interruption: python scripts/submit_stability.py {shlex.quote(args.input)} "
          f"--examples {args.examples} --run-root {shlex.quote(str(plan['root']))}")


def main():
    args = parser().parse_args()
    try:
        submit(args)
    except (ValueError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        print(f"Setup failed: {error}", file=sys.stderr)
        print("For a custom shared model cache, export your existing HF_HOME and retry. "
              "Use --python to select the existing model environment.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
