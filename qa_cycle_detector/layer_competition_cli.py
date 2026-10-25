"""Separate, resumable pilot for question-specific answer competition across layers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .answer_content import parse_extraction
from .layer_competition import (SCHEMA, HFLayerCompetitionBackend, answer_prompt,
                                compare_questions, validate_rows)
from .stability_cli import locked, read_json, write_json


def load_rows(path, maximum):
    if maximum < 1:
        raise ValueError("examples must be positive")
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == maximum:
                    break
    return validate_rows(rows)


def prepare(root, signature):
    marker = root / "LAYER_COMPETITION_RUN.json"
    for parent in (root, *root.parents):
        if any((parent / name).exists() for name in ("_RUN.json", "TRIPLET_RUN.json", "STABILITY_RUN.json")):
            raise ValueError("choose a separate output folder outside previous runs")
        if parent != root and (parent / marker.name).exists():
            raise ValueError("do not nest competition runs")
    if marker.exists():
        if read_json(marker) != signature:
            raise ValueError("run settings changed; choose a new folder")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("output folder must be empty or a matching competition run")
        root.mkdir(parents=True, exist_ok=True)
        write_json(marker, signature)


def run(root, rows, settings, backend_factory):
    rows = validate_rows(rows)
    prepare(root, {"schema": SCHEMA, "rows": rows, "settings": settings})
    backend = None
    def get_backend():
        nonlocal backend
        if backend is None:
            backend = backend_factory()
            metadata_path = root / "backend.json"
            if metadata_path.exists() and read_json(metadata_path) != backend.metadata:
                raise ValueError("resolved model/runtime changed; choose a new run folder")
            write_json(metadata_path, backend.metadata)
        return backend
    def cached(path, compute):
        if path.exists():
            return read_json(path)
        started = time.perf_counter()
        value = compute()
        value["seconds"] = time.perf_counter() - started
        write_json(path, value)
        return value
    examples = []
    with locked(root):
        for index, row in enumerate(rows):
            directory = root / "examples" / f"{index:04d}"
            score_path = directory / "scores.json"
            if score_path.exists():
                examples.append(read_json(score_path))
                continue
            views = [{"kind": "original", "question": row["question"]}, *row["controls"]]
            generations, pool, seen, anchor = [], [], {}, None
            serial = 0
            # Original greedy + stochastic samples, then one greedy per control.
            # Counterfactual answers expand a secondary pool, never the primary pool.
            for view_index, view in enumerate(views):
                count = 1 + settings["samples"] if view_index == 0 else 1
                for sample in range(count):
                    number, seed = serial, settings["seed"] + index * 10000 + serial * 2
                    serial += 1
                    generated = cached(directory / "generations" / f"{number}.json",
                        lambda v=view, s=sample, seed=seed: get_backend().generate(
                            answer_prompt(v["question"]), settings["max_new_tokens"], seed,
                            temperature=settings["temperature"] if s else None))
                    item = {"view": view_index, "sample": sample, "generation": generated}
                    if not generated["truncated"] and generated["text"].strip():
                        extraction = cached(directory / "extractions" / f"{number}.json",
                            lambda v=view, g=generated, seed=seed: get_backend().extract_answer(
                                v["question"], g["text"], seed + 1))
                        parsed = parse_extraction(extraction["text"], generated["text"])
                        item["extraction"] = parsed
                        if parsed["valid"]:
                            key = parsed["canonical"]
                            if key not in seen and len(pool) < settings["max_candidates"]:
                                seen[key] = len(pool)
                                pool.append({"text": parsed["span"].strip().rstrip("."),
                                             "canonical": key, "sources": []})
                            if key in seen:
                                ci = seen[key]
                                pool[ci]["sources"].append({"view": view_index, "sample": sample})
                                item["candidate_index"] = ci
                                if view_index == 0 and sample == 0:
                                    anchor = ci
                            else:
                                item["omitted_reason"] = "candidate budget exhausted"
                    generations.append(item)
            raw = {"input": row, "views": views, "anchor_index": anchor,
                   "candidates": pool, "generations": generations}
            write_json(directory / "raw.json", raw)
            primary = [i for i, c in enumerate(pool) if any(s["view"] == 0 for s in c["sources"])]
            summary = {"id": row["id"], "question": row["question"],
                       "anchor": pool[anchor]["text"] if anchor is not None else None,
                       "candidates": [c["text"] for c in pool],
                       "primary_candidate_count": len(primary), "expanded_candidate_count": len(pool),
                       "invalid_or_truncated_generations": sum("candidate_index" not in g for g in generations)}
            if anchor is None or len(pool) < 2:
                summary.update(original_pool={"valid": False, "reason": "missing anchor or alternatives"},
                               expanded_pool={"valid": False, "reason": "missing anchor or alternatives"})
            else:
                probes = []
                for vi, view in enumerate(views):
                    probes.append([cached(directory / "probes" / f"{vi}-{ci}.json",
                        lambda q=view["question"], c=candidate["text"]: get_backend().score_candidate(q, c))
                        for ci, candidate in enumerate(pool)])
                summary["original_pool"] = compare_questions(probes, row["controls"], primary, anchor)
                summary["expanded_pool"] = compare_questions(probes, row["controls"], list(range(len(pool))), anchor)
            write_json(score_path, summary)
            examples.append(summary)
            print(f"Completed {row['id']}: original pool={len(primary)}, expanded={len(pool)}", flush=True)
        report = {"schema": SCHEMA, "label_usage": "none; only questions and explicit controls consumed",
                  "examples": examples,
                  "valid_original_pools": sum(s["original_pool"]["valid"] for s in examples),
                  "valid_expanded_pools": sum(s["expanded_pool"]["valid"] for s in examples),
                  "interpretation": "Exploratory support diagnostics, not calibrated hallucination probabilities. "
                      "Expanded pool includes answers to edited questions and is a secondary control. "
                      "Intermediate logit-lens values are uncalibrated. No novelty or accuracy claim."}
        write_json(root / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "data/layer_competition_questions.jsonl")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--revision")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--samples", type=int, default=4, help="stochastic original-question samples, plus one greedy")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.max_candidates < 2 or args.max_new_tokens < 1 or not 0 < args.temperature < 10:
        parser.error("invalid sample count, candidate budget, token limit, or temperature")
    rows = load_rows(args.input, args.examples)
    settings = {key: value for key, value in vars(args).items()
                if key not in {"input", "run_root", "examples", "dry_run"}}
    if args.dry_run:
        print(json.dumps({"schema": SCHEMA, "examples": len(rows),
            "generation_requests": sum(1 + args.samples + len(r["controls"]) for r in rows),
            "max_scoring_requests": sum((1 + len(r["controls"])) * args.max_candidates for r in rows),
            "extraction": "one request per nonempty, untruncated generated answer",
            "output": str(args.run_root), "settings": settings}, indent=2))
        return
    factory = lambda: HFLayerCompetitionBackend(args.model, revision=args.revision,
                                                local_files_only=args.local_files_only)
    run(args.run_root.resolve(), rows, settings, factory)


if __name__ == "__main__":
    main()
