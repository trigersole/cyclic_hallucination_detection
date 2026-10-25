"""Small unlabeled pilot: python -m qa_cycle_detector.stability_cli --help."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .semantics import SentenceTransformerEncoder, similarities_to_target
from .stability import (HFStabilityBackend, INSTRUCTIONS, MODES, SCHEMA, PilotConfig,
                        perturbation, prompts, random_basis, semantic_summary, wording_basis)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unlabeled_rows(rows, maximum):
    if maximum < 1:
        raise ValueError("max-examples must be positive")
    selected, seen = [], set()
    for row in rows[:maximum]:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("each selected row needs a nonempty question string")
        if "answer" in row and (not isinstance(row["answer"], str) or not row["answer"].strip()):
            raise ValueError("optional supplied answer must be a nonempty string")
        key = (row["question"], row.get("answer"))
        if key in seen:
            raise ValueError("duplicate question/answer pair in pilot inputs")
        seen.add(key)
        # Whitelist inputs. Labels, grading results, and existing features never enter the pilot.
        selected.append({"question": row["question"]} |
                        ({"answer": row["answer"]} if "answer" in row else {}) |
                        ({"id": row["id"]} if "id" in row else {}))
    if not selected:
        raise ValueError("no pilot examples")
    return selected


def selected_inputs(path, maximum):
    """Stream only the pilot prefix, rather than loading an entire legacy run."""
    if maximum < 1:
        raise ValueError("max-examples must be positive")
    path = Path(path)
    paths = sorted(path.glob("shard_*.jsonl")) if path.is_dir() else [path]
    if not paths:
        raise ValueError("no shard_*.jsonl inputs found")
    rows = []
    for shard in paths:
        with shard.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON in {shard} at line {line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"expected an object in {shard} at line {line_number}")
                rows.append(row)
                if len(rows) == maximum:
                    return unlabeled_rows(rows, maximum)
    return unlabeled_rows(rows, maximum)


def prepare_run(root, signature):
    marker = root / "STABILITY_RUN.json"
    for parent in (root, *root.parents):
        if (parent / "_RUN.json").exists() or (parent / "TRIPLET_RUN.json").exists():
            raise ValueError("pilot root must be outside existing cycle/triplet outputs")
    if marker.exists():
        if read_json(marker) != signature:
            raise ValueError("pilot inputs or settings changed; choose a new run root")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("new pilot root must be empty; existing directories are never adopted")
        root.mkdir(parents=True, exist_ok=True)
        write_json(marker, signature)


@contextmanager
def locked(root):
    path = root / ".stability.lock"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError as error:
        raise ValueError("pilot is locked; after a hard kill remove .stability.lock only when no process is running") from error
    try:
        yield
    finally:
        path.unlink()


def cached_generation(path, backend, prompt, config, seed, delta=None, temperature=None):
    if path.exists():
        return read_json(path)
    start = time.perf_counter()
    value = backend.generate(prompt, config.max_new_tokens, seed, delta=delta, temperature=temperature)
    value.update(seconds=time.perf_counter() - start, seed=seed)
    write_json(path, value)
    return value


def collect_example(directory, row, backend, config):
    directory.mkdir(parents=True, exist_ok=True)
    views = prompts(row["question"])
    seed = (config.seed + int(digest({k: row[k] for k in ("question", "answer") if k in row})[:8], 16)) % (2 ** 31)
    state_path = directory / "states.json"
    if state_path.exists():
        state = read_json(state_path)
    else:
        states = []
        for i, view in enumerate(views):
            path = directory / "views" / f"{i}.json"
            if path.exists():
                value = read_json(path)
            else:
                started = time.perf_counter()
                value = {"prompt": view, "state": np.asarray(backend.capture(view)).tolist(),
                         "seconds": time.perf_counter() - started}
                write_json(path, value)
            states.append(value["state"])
        basis_started = time.perf_counter()
        basis = wording_basis(states, config.rank)
        control = random_basis(len(states[0]), basis.shape[1], seed)
        state = {"wording_basis": basis.tolist(), "random_basis": control.tolist(),
                 "state_norm": float(np.linalg.norm(states[0])), "actual_rank": basis.shape[1],
                 "hidden_dimension": len(states[0]),
                 "capture_seconds": sum(read_json(directory / "views" / f"{i}.json")["seconds"]
                                        for i in range(len(views))),
                 "basis_seconds": time.perf_counter() - basis_started,
                 "view_displacement_norms": np.linalg.norm(np.asarray(states) - states[0], axis=1).tolist()}
        write_json(state_path, state)
    basis = np.asarray(state["wording_basis"])
    control = np.asarray(state["random_basis"])
    anchor = cached_generation(directory / "generations" / "anchor.json", backend, views[0], config, seed)
    units = []
    for direction in range(config.baseline_samples):
        value = cached_generation(directory / "generations" / f"baseline_{direction}.json", backend,
                                  views[0], config, (seed + 1000 + direction) % (2 ** 31),
                                  temperature=config.baseline_temperature)
        units.append({"mode": "baseline", "direction": direction, "amplitude": None, **value})
    for aindex, amplitude in enumerate(config.amplitudes):
        for direction in range(config.directions):
            direction_seed = (seed + 2000 + direction) % (2 ** 31)
            for mode in MODES:
                delta = perturbation(mode, basis, control, direction_seed, amplitude, state["state_norm"])
                value = cached_generation(directory / "generations" / f"a{aindex}_d{direction}_{mode}.json",
                                          backend, views[0], config, direction_seed, delta=delta)
                units.append({"mode": mode, "direction": direction, "amplitude": amplitude,
                              "delta_norm": float(np.linalg.norm(delta)), **value})
        print(f"  completed perturbation amplitude {amplitude:g}", flush=True)
    result = {"schema": SCHEMA, "input": row, "anchor": anchor, "state": state, "units": units}
    write_json(directory / "raw.json", result)
    return result


def score_example(raw, encoder, config):
    started = time.perf_counter()
    answer = raw["input"].get("answer", raw["anchor"]["text"])
    answer_source = "supplied" if "answer" in raw["input"] else "generated_anchor"
    all_texts = [answer, raw["anchor"]["text"]] + [u["text"] for u in raw["units"]]
    unique = list(dict.fromkeys(text for text in all_texts if text.strip()))
    vectors = np.asarray(encoder.encode(unique), dtype=float) if unique else np.empty((0, 1))
    if (vectors.ndim != 2 or len(vectors) != len(unique) or not np.isfinite(vectors).all()
            or (np.linalg.norm(vectors, axis=1) < 1e-12).any()):
        raise ValueError("semantic encoder returned invalid vectors")
    lookup = dict(zip(unique, vectors))
    zero = np.zeros(vectors.shape[1])
    anchor = lookup.get(raw["anchor"]["text"], zero)
    supplied = lookup.get(answer, zero)

    def summarize(units):
        texts = [u["text"] for u in units]
        result = semantic_summary(texts, np.asarray([lookup.get(t, zero) for t in texts]),
                                  anchor, supplied, config.equivalence_threshold)
        result["truncated_rate"] = sum(u["truncated"] for u in units) / len(units)
        result["mean_token_entropy"] = float(np.mean([u["mean_token_entropy"] for u in units]))
        result["seconds"] = sum(u["seconds"] for u in units)
        return result

    baseline = summarize([u for u in raw["units"] if u["mode"] == "baseline"])
    conditions = {}
    for mode in MODES:
        conditions[mode] = []
        for amplitude in config.amplitudes:
            units = [u for u in raw["units"] if u["mode"] == mode and u["amplitude"] == amplitude]
            conditions[mode].append({"amplitude": amplitude, **summarize(units)})
    anchor_valid = bool(raw["anchor"]["text"].strip()) and not raw["anchor"]["truncated"]
    auc, first_flip, usable = {}, {}, {}
    for mode, values in conditions.items():
        usable[mode] = anchor_valid and all(v["invalid_rate"] == 0 and v["truncated_rate"] == 0 for v in values)
        rates = [0.] + [v["anchor_flip_rate"] for v in values]
        x = np.asarray([0.] + list(config.amplitudes))
        # Integrated flip fraction / tested amplitude span; not a probability of error.
        auc[mode] = (float(np.sum(np.diff(x) * (np.asarray(rates[1:]) + rates[:-1]) / 2) / x[-1])
                     if usable[mode] else None)
        first_flip[mode] = next((v["amplitude"] for v in values if v["anchor_flip_rate"] is not None
                                 and v["anchor_flip_rate"] > 0), None) if usable[mode] else None
    corrected = (auc["orthogonal"] - auc["wording"]
                 if usable["orthogonal"] and usable["wording"] else None)
    random_corrected = (auc["random_orthogonal"] - auc["random_wording"]
                        if usable["random_orthogonal"] and usable["random_wording"] else None)
    specificity = corrected - random_corrected if corrected is not None and random_corrected is not None else None
    generated = [raw["anchor"], *raw["units"]]
    anchor_similarity = float(similarities_to_target(np.asarray([anchor]), supplied)[0])
    anchor_agreement = anchor_valid and anchor_similarity >= config.equivalence_threshold
    return {"schema": SCHEMA, "input": raw["input"], "anchor": raw["anchor"]["text"],
            "answer": answer, "answer_source": answer_source,
            "supplied_anchor_similarity": anchor_similarity,
            "supplied_anchor_equivalent": anchor_agreement,
            "supplied_answer_score": corrected if anchor_agreement else None,
            "actual_rank": raw["state"]["actual_rank"], "baseline": baseline, "conditions": conditions,
            "usable_conditions": usable, "scores": {"flip_auc": auc,
                "wording_corrected": corrected, "random_corrected": random_corrected,
                "wording_specificity_contrast": specificity},
            "first_tested_flip_amplitude": first_flip,
            "note": "Null radius means no observed flip within the budget OR unusable samples; check usable_conditions. "
                    "Scores are uncalibrated sensitivity measures, not correctness probabilities.",
            "cost": {"generation_calls": len(generated),
                     "input_tokens": sum(u["input_tokens"] for u in generated),
                     "output_tokens": sum(u["output_tokens"] for u in generated),
                     "generation_seconds": sum(u["seconds"] for u in generated),
                     "capture_seconds": raw["state"]["capture_seconds"],
                     "semantic_scoring_seconds": time.perf_counter() - started}}


def aggregate(scores):
    ranking = sorted((s for s in scores if s["scores"]["wording_corrected"] is not None),
                     key=lambda s: s["scores"]["wording_corrected"], reverse=True)
    return {"schema": SCHEMA, "examples": len(scores), "label_usage": "none; correctness fields discarded",
            "supplied_anchor_disagreements": sum(not s["supplied_anchor_equivalent"] for s in scores),
            "usable_examples": {mode: sum(s["usable_conditions"][mode] for s in scores) for mode in MODES},
            "summary": {mode: {"mean_flip_auc": float(np.mean(values)) if values else None,
                                "examples": len(values)}
                        for mode in MODES
                        for values in [[s["scores"]["flip_auc"][mode] for s in scores
                                        if s["scores"]["flip_auc"][mode] is not None]]},
            "ranked_by_wording_corrected": [{"input": s["input"], "scores": s["scores"]} for s in ranking],
            "cost": {key: sum(s["cost"][key] for s in scores) for key in scores[0]["cost"]},
            "interpretation": "Diagnostic pilot only. No correctness labels, threshold fitting, or accuracy claims. "
                              "Compare wording correction against equal-rank random controls before attributing an effect."}


def run_pilot(root, rows, config, settings, backend_factory, encoder_factory):
    signature = {"schema": SCHEMA, "inputs": rows, "config": json.loads(json.dumps(config.to_dict())),
                 "settings": settings, "instructions": list(INSTRUCTIONS)}
    prepare_run(root, signature)
    with locked(root):
        backend, encoder = None, None
        scores = []
        for i, row in enumerate(rows):
            directory = root / "examples" / f"{i:05d}"
            path = directory / "scores.json"
            if path.exists():
                scores.append(read_json(path))
                continue
            raw_path = directory / "raw.json"
            if raw_path.exists():
                raw = read_json(raw_path)
            else:
                if backend is None:
                    backend = backend_factory()
                    metadata_path = root / "MODEL.json"
                    if metadata_path.exists() and read_json(metadata_path) != backend.metadata:
                        raise ValueError("resolved model/runtime changed; choose a new run root")
                    write_json(metadata_path, backend.metadata)
                print(f"example {i + 1}/{len(rows)}: collecting interventions", flush=True)
                raw = collect_example(directory, row, backend, config)
            if encoder is None:
                encoder = encoder_factory()
            score = score_example(raw, encoder, config)
            write_json(path, score)
            scores.append(score)
        report = aggregate(scores)
        write_json(root / "report.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(Path(__file__).resolve().parents[1] / "data" / "stability_pilot_questions.jsonl"),
                        help="question-only/QA JSONL or shard directory; defaults to fresh bundled questions")
    parser.add_argument("--run-root", required=True, help="new, separate output directory")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default=None, help="prefer an immutable model commit")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--layer", type=int, default=None, help="zero-based decoder block; default middle block")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[.005, .02, .05])
    parser.add_argument("--baseline-samples", type=int, default=4)
    parser.add_argument("--baseline-temperature", type=float, default=.8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--equivalence-threshold", type=float, default=.82)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print budget; no model loading or writes")
    args = parser.parse_args()
    config = PilotConfig(args.rank, args.directions, tuple(args.amplitudes), args.baseline_samples,
                         args.baseline_temperature, args.max_new_tokens, args.equivalence_threshold, args.seed)
    rows = selected_inputs(args.input, args.max_examples)
    settings = {key: getattr(args, key) for key in ("model", "revision", "embedding_model", "embedding_device",
                                                  "device_map", "torch_dtype", "local_files_only", "layer")}
    calls = 1 + config.baseline_samples + len(MODES) * len(config.amplitudes) * config.directions
    if args.dry_run:
        print(json.dumps({"examples": len(rows), "labels_used": False,
                          "generation_calls_per_example": calls, "capture_passes_per_example": len(INSTRUCTIONS),
                          "maximum_generated_tokens": len(rows) * calls * config.max_new_tokens,
                          "config": config.to_dict()}, indent=2))
        return
    report = run_pilot(Path(args.run_root).resolve(), rows, config, settings,
                       lambda: HFStabilityBackend(args.model, args.layer, args.device_map, args.torch_dtype,
                                                   args.local_files_only, args.revision),
                       lambda: SentenceTransformerEncoder(args.embedding_model, device=args.embedding_device,
                                                           local_files_only=args.local_files_only))
    print(json.dumps({"examples": report["examples"], "usable": report["usable_examples"],
                      "report": str(Path(args.run_root).resolve() / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
