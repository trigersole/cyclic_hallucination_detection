#!/usr/bin/env python3
"""One-command Slurm submission for the separate layer-competition pilot."""
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


def make_plan(args, environ):
    env = dict(environ)
    home = Path.home()
    halo_python = Path(env.get("HALOSCOPE_DIR", str(home / "LLM_Haloscope"))) / ".venv/bin/python"
    python = args.python or env.get("STABILITY_PYTHON") or env.get("HALOSCOPE_PYTHON") or (
        str(halo_python) if halo_python.is_file() else sys.executable)
    model = args.model or env.get("STABILITY_MODEL") or env.get("MODEL_ID") or "meta-llama/Llama-2-7b-chat-hf"
    base = Path(env.get("SCRATCH") or home) / "qa-layer-results"
    root = (args.run_root.expanduser().resolve() if args.run_root else
            base / ("layers-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]))
    source = args.input.expanduser().resolve()
    overlay = env.get("CYCLE_OVERLAY", str(Path(env.get("SCRATCH") or home) / "qa-cycle-overlay"))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPOSITORY), overlay, env.get("PYTHONPATH")]))
    command = [python, "-m", "qa_cycle_detector.layer_competition_cli", "--input", str(source),
               "--run-root", str(root), "--model", model, "--examples", str(args.examples)]
    if not args.allow_downloads:
        command.append("--local-files-only")
    if args.revision:
        command.extend(["--revision", args.revision])
    logs = REPOSITORY / "slurm/logs"
    submit = ["sbatch", "--parsable", "--export=ALL", f"--partition={args.partition}",
              "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=32G", "--gres=gpu:1",
              "--time=01:30:00", "--job-name=layer-competition", f"--chdir={REPOSITORY}",
              f"--output={logs / 'layer-competition-%j.out'}", f"--error={logs / 'layer-competition-%j.err'}"]
    account = args.account or env.get("SLURM_ACCOUNT")
    if account:
        submit.append(f"--account={account}")
    gpu_check = shlex.join([python, "-c", "import torch; assert torch.cuda.is_available(), 'No allocated CUDA GPU'"])
    submit.extend(["--wrap", gpu_check + " && exec " + shlex.join(command)])
    return {"root": root, "env": env, "command": command, "submit": submit, "logs": logs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=REPOSITORY / "data/layer_competition_questions.jsonl")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--account")
    parser.add_argument("--partition", default="MGPU-TC2")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = make_plan(args, os.environ)
    subprocess.run(plan["command"] + ["--dry-run"], cwd=REPOSITORY, env=plan["env"], check=True)
    if args.dry_run:
        print("Submission:", shlex.join(plan["submit"]))
        print("Results:", plan["root"])
        print("No job submitted or directories created.")
        return
    if not shutil.which("sbatch"):
        parser.error("sbatch is unavailable; submit from the cluster login node")
    # Validate the output location before submitting; dry-run above checks inputs.
    # No run marker is written on the login node. The worker owns initialization.
    root = plan["root"]
    for parent in (root, *root.parents):
        if any((parent / marker).exists() for marker in ("STABILITY_RUN.json", "_RUN.json", "TRIPLET_RUN.json")):
            parser.error("choose an output folder outside existing runs")
    if root.exists() and any(root.iterdir()) and not (root / "LAYER_COMPETITION_RUN.json").exists():
        parser.error("output folder is not an existing competition run or empty folder")
    # Same model cache and Python overlay as the stability pilot; no installation.
    cache_check = ("import torch, transformers; from pathlib import Path; "
                   "from huggingface_hub import snapshot_download; import sys; "
                   "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2] or None, local_files_only=True) "
                   "if not Path(sys.argv[1]).is_dir() else None")
    if not args.allow_downloads:
        model = plan["command"][plan["command"].index("--model") + 1]
        subprocess.run([plan["command"][0], "-c", cache_check, model, args.revision or ""],
                       cwd=REPOSITORY, env=plan["env"], check=True)
    plan["logs"].mkdir(parents=True, exist_ok=True)
    result = subprocess.run(plan["submit"], cwd=REPOSITORY, env=plan["env"],
                            check=True, capture_output=True, text=True)
    print("Submitted job:", result.stdout.strip())
    print("Results:", root)
    print("Report:", root / "report.json")
    print("Resume with the same options plus --run-root", shlex.quote(str(root)))


if __name__ == "__main__":
    main()
