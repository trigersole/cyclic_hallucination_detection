"""Small unlabeled pilot: python -m qa_cycle_detector.stability_cli --help."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .semantics import SentenceTransformerEncoder, similarities_to_target
from .answer_content import parse_extraction, span_summary
from .stability import (DIAGNOSTIC_AMPLITUDES, HFStabilityBackend, INSTRUCTIONS, MODES, SCHEMA, PilotConfig,
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


def cached_generation(path, backend, prompt, config, seed, delta=None, temperature=None,
                      reference_logprobs=None, store_reference=False):
    if path.exists():
        return read_json(path)
    start = time.perf_counter()
    kwargs = {"delta": delta, "temperature": temperature}
    if config.diagnostics:
        kwargs.update(diagnostics=True, reference_logprobs=reference_logprobs)
    value = backend.generate(prompt, config.max_new_tokens, seed, **kwargs)
    first = value.pop("_first_logprobs", None)
    if store_reference:
        if first is None:
            raise ValueError("diagnostic backend did not return anchor vocabulary probabilities")
        destination = path.with_suffix(".first_logprobs.npy")
        temporary = destination.with_suffix(".npy.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            np.save(handle, first, allow_pickle=False)
        os.replace(temporary, destination)
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
    anchor_path = directory / "generations" / "anchor.json"
    anchor = cached_generation(anchor_path, backend, views[0], config, seed,
                               store_reference=config.diagnostics)
    reference = None
    if config.diagnostics:
        reference = np.load(anchor_path.with_suffix(".first_logprobs.npy"), allow_pickle=False)
    units = []
    if config.diagnostics:
        value = cached_generation(directory / "generations" / "zero_control.json", backend,
                                  views[0], config, seed, delta=np.zeros(basis.shape[0]),
                                  reference_logprobs=reference)
        units.append({"mode": "zero_control", "direction": 0, "amplitude": 0., **value})
    for direction in range(config.baseline_samples):
        value = cached_generation(directory / "generations" / f"baseline_{direction}.json", backend,
                                  views[0], config, (seed + 1000 + direction) % (2 ** 31),
                                  temperature=config.baseline_temperature, reference_logprobs=reference)
        units.append({"mode": "baseline", "direction": direction, "amplitude": None, **value})
    for aindex, amplitude in enumerate(config.amplitudes):
        for direction in range(config.directions):
            direction_seed = (seed + 2000 + direction) % (2 ** 31)
            for mode in MODES:
                delta = perturbation(mode, basis, control, direction_seed, amplitude, state["state_norm"])
                value = cached_generation(directory / "generations" / f"a{aindex}_d{direction}_{mode}.json",
                                          backend, views[0], config, direction_seed, delta=delta,
                                          reference_logprobs=reference)
                units.append({"mode": mode, "direction": direction, "amplitude": amplitude,
                              "delta_norm": float(np.linalg.norm(delta)), **value})
        print(f"  completed perturbation amplitude {amplitude:g}", flush=True)
    result = {"schema": SCHEMA, "input": row, "anchor": anchor, "state": state, "units": units}
    write_json(directory / "raw.json", result)
    return result


def diagnostic_summary(units, anchor):
    first = [u["first_token"] for u in units]
    measurements = [u.get("intervention_measurement") for u in units]
    valid_measurements = [m for m in measurements if m is not None]
    retention = [m["norm_retention"] for m in valid_measurements if m["norm_retention"] is not None]
    return {"mean_js_divergence": float(np.mean([v["js_divergence"] for v in first])),
            "max_js_divergence": max(v["js_divergence"] for v in first),
            "mean_anchor_top_probability_drop": float(np.mean([v["anchor_top_probability_drop"] for v in first])),
            "mean_top2_logit_margin_change": float(np.mean([v["top2_logit_margin"] - anchor["top2_logit_margin"]
                                                             for v in first])),
            "top_token_flip_rate": float(np.mean([v["top_token_changed"] for v in first])),
            "mean_requested_relative_norm": float(np.mean([m["requested_relative_norm"] for m in valid_measurements]))
                if valid_measurements else None,
            "mean_applied_relative_norm": float(np.mean([m["applied_relative_norm"] for m in valid_measurements]))
                if valid_measurements else None,
            "min_norm_retention": min(retention) if retention else None,
            "max_norm_retention": max(retention) if retention else None,
            "zero_effective_perturbations": sum(m["applied_norm"] == 0 and m["requested_norm"] > 0
                                                 for m in valid_measurements)}


def collect_content(directory, raw, backend, config):
    """Checkpoint span extraction and fixed-prefix probes separately from generation."""
    question = raw["input"]["question"]
    texts = list(dict.fromkeys([raw["anchor"]["text"], raw["input"].get("answer", "")] +
                              [u["text"] for u in raw["units"]]))
    entries, extraction_usage = {}, []
    for text in texts:
        if not text.strip():
            entries[text] = parse_extraction('{"span": null}', text)
            continue
        key = digest({"question": question, "response": text})
        path = directory / "answer_spans" / f"{key}.json"
        if path.exists():
            value = read_json(path)
        else:
            started = time.perf_counter()
            generated = backend.extract_answer(question, text, (config.seed + int(key[:8], 16)) % (2 ** 31))
            value = {"response": text, "parsed": parse_extraction(generated["text"], text),
                     "input_tokens": generated["input_tokens"], "output_tokens": generated["output_tokens"],
                     "seconds": time.perf_counter() - started}
            write_json(path, value)
        entries[text] = value["parsed"]
        extraction_usage.append(value)
    raw["answer_extractions"] = {"entries": entries, "generation_calls": len(extraction_usage),
        "input_tokens": sum(v["input_tokens"] for v in extraction_usage),
        "output_tokens": sum(v["output_tokens"] for v in extraction_usage),
        "seconds": sum(v["seconds"] for v in extraction_usage)}
    if config.diagnostics:
        extraction = entries[raw["anchor"]["text"]]
        reference_path = directory / "answer_probes" / "anchor.npy"

        def probe(path, delta=None, reference=None, store=False):
            if path.exists():
                return read_json(path)
            started = time.perf_counter()
            requests = 0
            if not extraction["valid"]:
                result = {"valid": False, "error": "anchor answer span is invalid"}
            else:
                try:
                    requests = 1
                    result = backend.score_answer_tokens(prompts(question)[0], raw["anchor"], extraction,
                                                          delta=delta, reference_logprobs=reference)
                    logprobs = result.pop("_answer_logprobs")
                    if store:
                        reference_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = reference_path.with_suffix(".npy.tmp")
                        with temporary.open("wb") as handle:
                            np.save(handle, logprobs, allow_pickle=False)
                        os.replace(temporary, reference_path)
                except ValueError as error:
                    result = {"valid": False, "error": str(error)}
            result.update(seconds=time.perf_counter() - started, probe_requests=requests)
            write_json(path, result)
            return result

        anchor_probe = probe(directory / "answer_probes" / "anchor.json", store=True)
        raw["answer_probe_reference"] = anchor_probe
        if anchor_probe["valid"]:
            reference = np.load(reference_path, allow_pickle=False)
            basis = np.asarray(raw["state"]["wording_basis"])
            control = np.asarray(raw["state"]["random_basis"])
            for u in raw["units"]:
                if u["mode"] == "baseline":
                    continue
                delta = (np.zeros(basis.shape[0]) if u["mode"] == "zero_control" else
                         perturbation(u["mode"], basis, control, u["seed"], u["amplitude"], raw["state"]["state_norm"]))
                path = directory / "answer_probes" / f"{u['mode']}_{u['amplitude']}_{u['direction']}.json"
                u["answer_token_diagnostics"] = probe(path, delta, reference)
    raw["content_complete"] = True
    write_json(directory / "raw.json", raw)
    return raw


def score_example(raw, encoder, config):
    started = time.perf_counter()
    answer = raw["input"].get("answer", raw["anchor"]["text"])
    answer_source = "supplied" if "answer" in raw["input"] else "generated_anchor"
    spans = raw.get("answer_extractions", {}).get("entries")
    all_texts = [answer, raw["anchor"]["text"]] + [u["text"] for u in raw["units"]]
    unique = list(dict.fromkeys(text for text in all_texts if text.strip()))
    vectors = (np.eye(len(unique)) if spans is not None else
               np.asarray(encoder.encode(unique), dtype=float)) if unique else np.empty((0, 1))
    if (vectors.ndim != 2 or len(vectors) != len(unique) or not np.isfinite(vectors).all()
            or (np.linalg.norm(vectors, axis=1) < 1e-12).any()):
        raise ValueError("semantic encoder returned invalid vectors")
    lookup = dict(zip(unique, vectors))
    zero = np.zeros(vectors.shape[1])
    anchor = lookup.get(raw["anchor"]["text"], zero)
    supplied = lookup.get(answer, zero)

    def summarize(units):
        texts = [u["text"] for u in units]
        result = (span_summary([spans[t] for t in texts], spans[raw["anchor"]["text"]], spans[answer])
                  if spans is not None else
                  semantic_summary(texts, np.asarray([lookup.get(t, zero) for t in texts]),
                                   anchor, supplied, config.equivalence_threshold))
        result["truncated_rate"] = sum(u["truncated"] for u in units) / len(units)
        result["mean_token_entropy"] = float(np.mean([u["mean_token_entropy"] for u in units]))
        result["seconds"] = sum(u["seconds"] for u in units)
        if config.diagnostics:
            result["first_token"] = diagnostic_summary(units, raw["anchor"]["first_token"])
            probes = [u.get("answer_token_diagnostics") for u in units]
            valid_probes = [p for p in probes if p and p["valid"]]
            result["answer_tokens"] = {"valid_samples": len(valid_probes),
                "mean_js_divergence": float(np.mean([p["mean_js_divergence"] for p in valid_probes]))
                    if len(valid_probes) == len(units) else None,
                "mean_target_logprob_change": float(np.mean([p["mean_target_logprob_change"] for p in valid_probes]))
                    if len(valid_probes) == len(units) else None}
        return result

    baseline = summarize([u for u in raw["units"] if u["mode"] == "baseline"])
    conditions = {}
    for mode in MODES:
        conditions[mode] = []
        for amplitude in config.amplitudes:
            units = [u for u in raw["units"] if u["mode"] == mode and u["amplitude"] == amplitude]
            conditions[mode].append({"amplitude": amplitude, **summarize(units)})
    anchor_valid = (bool(raw["anchor"]["text"].strip()) and not raw["anchor"]["truncated"]
                    and (spans is None or spans[raw["anchor"]["text"]]["valid"]))
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
    if spans is not None:
        comparable = spans[raw["anchor"]["text"]]["valid"] and spans[answer]["valid"]
        anchor_similarity = float(spans[raw["anchor"]["text"]]["canonical"] == spans[answer]["canonical"]) if comparable else None
    else:
        anchor_similarity = float(similarities_to_target(np.asarray([anchor]), supplied)[0])
    anchor_agreement = anchor_valid and anchor_similarity is not None and (
        anchor_similarity == 1. if spans is not None else anchor_similarity >= config.equivalence_threshold)
    result = {"schema": SCHEMA, "input": raw["input"], "anchor": raw["anchor"]["text"],
            "anchor_usable": anchor_valid,
            "anchor_output_valid": bool(raw["anchor"]["text"].strip()) and not raw["anchor"]["truncated"],
            "anchor_span_valid": spans[raw["anchor"]["text"]]["valid"] if spans is not None else None,
            "answer": answer, "answer_source": answer_source,
            "answer_comparison": "question_conditioned_extractive_span_exact" if spans is not None else "legacy_sentence_cosine",
            "answer_span": spans[raw["anchor"]["text"]] if spans is not None else None,
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
    extraction_cost = raw.get("answer_extractions", {})
    result["cost"].update(answer_extraction_calls=extraction_cost.get("generation_calls", 0),
                          answer_extraction_seconds=extraction_cost.get("seconds", 0),
                          answer_extraction_input_tokens=extraction_cost.get("input_tokens", 0),
                          answer_extraction_output_tokens=extraction_cost.get("output_tokens", 0))
    probes = [raw.get("answer_probe_reference", {})] + [u.get("answer_token_diagnostics", {}) for u in raw["units"]]
    result["cost"].update(answer_probe_requests=sum(p.get("probe_requests", 0) for p in probes),
                          answer_probe_seconds=sum(p.get("seconds", 0) for p in probes))
    if config.diagnostics:
        zero = next(u for u in raw["units"] if u["mode"] == "zero_control")
        passed = (zero["text"] == raw["anchor"]["text"] and zero["first_token"]["js_divergence"] <= 1e-8
                  and zero["intervention_measurement"]["applied_norm"] == 0)
        js_auc = {}
        x = np.asarray([0.] + list(config.amplitudes))
        for mode, values in conditions.items():
            y = np.asarray([0.] + [v["first_token"]["mean_js_divergence"] for v in values])
            js_auc[mode] = float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / x[-1])
        result["zero_control_passed"] = passed
        result["zero_control"] = {"text": zero["text"], "first_token": zero["first_token"],
                                  "intervention_measurement": zero["intervention_measurement"]}
        result["anchor_first_token"] = raw["anchor"]["first_token"]
        result["scores"]["first_token_js_auc"] = js_auc
        result["scores"]["first_token_wording_corrected"] = js_auc["orthogonal"] - js_auc["wording"]
        if "answer_probe_reference" in raw:
            result["answer_content_reference"] = raw["answer_probe_reference"]
            zero_probe = zero.get("answer_token_diagnostics", {})
            result["answer_zero_control_passed"] = (zero_probe["mean_js_divergence"] <= 1e-8
                                                     if zero_probe.get("valid") else None)
            answer_auc = {}
            for mode, values in conditions.items():
                y = [0.] + [v["answer_tokens"]["mean_js_divergence"] for v in values]
                answer_auc[mode] = (float(np.sum(np.diff(x) * (np.asarray(y[1:]) + y[:-1]) / 2) / x[-1])
                                    if all(v is not None for v in y) else None)
            result["scores"]["answer_token_js_auc"] = answer_auc
            result["scores"]["answer_token_wording_corrected"] = (answer_auc["orthogonal"] - answer_auc["wording"]
                if answer_auc["orthogonal"] is not None and answer_auc["wording"] is not None else None)
        if not passed:
            result["diagnostic_warning"] = "Zero control failed: inspect numerical nondeterminism or intervention integrity."
    return result


def aggregate(scores):
    ranking = sorted((s for s in scores if s["scores"]["wording_corrected"] is not None),
                     key=lambda s: s["scores"]["wording_corrected"], reverse=True)
    return {"schema": SCHEMA, "examples": len(scores), "label_usage": "none; correctness fields discarded",
            "empty_or_truncated_anchors": sum(not s["anchor_output_valid"] for s in scores),
            "invalid_anchor_extractions": sum(s["anchor_span_valid"] is False for s in scores),
            "invalid_answer_probe_references": sum(not s["answer_content_reference"]["valid"] for s in scores
                                                    if "answer_content_reference" in s),
            "supplied_anchor_disagreements": sum(s["answer_source"] == "supplied" and s["anchor_usable"]
                                                  and not s["supplied_anchor_equivalent"] for s in scores),
            "zero_control_failures": sum(s.get("zero_control_passed") is False for s in scores),
            "answer_zero_control_failures": sum(s.get("answer_zero_control_passed") is False for s in scores),
            "usable_examples": {mode: sum(s["usable_conditions"][mode] for s in scores) for mode in MODES},
            "summary": {mode: {"mean_flip_auc": float(np.mean(values)) if values else None,
                                "examples": len(values),
                                "mean_format_first_token_js_auc": float(np.mean(js_values)) if js_values else None,
                                "mean_answer_token_js_auc": float(np.mean(answer_values)) if answer_values else None}
                        for mode in MODES
                        for values in [[s["scores"]["flip_auc"][mode] for s in scores
                                        if s["scores"]["flip_auc"][mode] is not None]]
                        for js_values in [[s["scores"]["first_token_js_auc"][mode] for s in scores
                                           if "first_token_js_auc" in s["scores"]]]
                        for answer_values in [[s["scores"]["answer_token_js_auc"][mode] for s in scores
                                               if s["scores"].get("answer_token_js_auc", {}).get(mode) is not None]]},
            "ranked_by_wording_corrected": [{"input": s["input"], "scores": s["scores"]} for s in ranking],
            "ranked_by_answer_token_wording_corrected": [
                {"input": s["input"], "scores": s["scores"]} for s in sorted(
                    [s for s in scores if s["scores"].get("answer_token_wording_corrected") is not None],
                    key=lambda s: s["scores"]["answer_token_wording_corrected"], reverse=True)],
            "cost": {key: sum(s["cost"][key] for s in scores) for key in scores[0]["cost"]},
            "interpretation": "Diagnostic pilot only. Extracted answer-span changes are candidate changes, not proven "
                              "factual errors; aliases may be overcounted. No correctness-label training or threshold fitting. "
                              "Compare against equal-rank random controls. Formatting-token JS is not the primary score."}


def run_pilot(root, rows, config, settings, backend_factory, encoder_factory):
    signature = {"schema": SCHEMA, "inputs": rows, "config": json.loads(json.dumps(config.to_dict())),
                 "settings": settings, "instructions": list(INSTRUCTIONS)}
    prepare_run(root, signature)
    with locked(root):
        backend, encoder = None, None
        def get_backend():
            nonlocal backend
            if backend is None:
                backend = backend_factory()
                metadata_path = root / "MODEL.json"
                if metadata_path.exists() and read_json(metadata_path) != backend.metadata:
                    raise ValueError("resolved model/runtime changed; choose a new run root")
                write_json(metadata_path, backend.metadata)
            return backend
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
                print(f"example {i + 1}/{len(rows)}: collecting interventions", flush=True)
                raw = collect_example(directory, row, get_backend(), config)
            if not raw.get("content_complete"):
                print(f"example {i + 1}/{len(rows)}: extracting answer content", flush=True)
                raw = collect_content(directory, raw, get_backend(), config)
            score = score_example(raw, encoder, config)
            write_json(path, score)
            scores.append(score)
        report = aggregate(scores)
        write_json(root / "report.json", report)
        return report


def saved_run(source, maximum=None):
    source = Path(source).resolve()
    marker = read_json(source / "STABILITY_RUN.json")
    if not str(marker.get("schema", "")).startswith("wording-stability-pilot-v"):
        raise ValueError("source is not a stability pilot")
    paths = sorted(source.glob("examples/*/raw.json"))
    if maximum is not None:
        if maximum < 1:
            raise ValueError("max-examples must be positive")
        paths = paths[:maximum]
    if not paths:
        raise ValueError("source has no completed raw answers")
    config = dict(marker["config"])
    config["amplitudes"] = tuple(config["amplitudes"])
    config["diagnostics"] = False  # No new answer generation or probability probes.
    return marker, [read_json(p) for p in paths], PilotConfig(**config)


def rescore_saved(root, source, maximum, settings, backend_factory):
    source = Path(source).resolve()
    if root == source or source in root.parents:
        raise ValueError("rescore output must be separate from the source run")
    marker, raws, config = saved_run(source, maximum)
    prepare_run(root, {"schema": SCHEMA, "mode": "saved_answer_text_rescore",
                       "source": str(source), "source_digest": digest(raws), "settings": settings,
                       "config": json.loads(json.dumps(config.to_dict()))})
    scores, backend = [], None
    with locked(root):
        for i, original in enumerate(raws):
            directory = root / "examples" / f"{i:05d}"
            if (directory / "scores.json").exists():
                scores.append(read_json(directory / "scores.json"))
                continue
            if backend is None:
                backend = backend_factory()
                metadata_path = root / "MODEL.json"
                if metadata_path.exists() and read_json(metadata_path) != backend.metadata:
                    raise ValueError("extraction model/runtime changed on resume")
                write_json(metadata_path, backend.metadata)
            raw = copy.deepcopy(original)
            raw["input"] = unlabeled_rows([raw["input"]], 1)[0]
            raw.pop("answer_probe_reference", None)
            for unit in raw["units"]:
                unit.pop("answer_token_diagnostics", None)
            raw = collect_content(directory, raw, backend, config)
            score = score_example(raw, None, config)
            score["reused_generation_cost"] = {k: score["cost"][k] for k in (
                "generation_calls", "input_tokens", "output_tokens", "generation_seconds", "capture_seconds")}
            for key in score["reused_generation_cost"]:
                score["cost"][key] = 0
            write_json(directory / "scores.json", score)
            scores.append(score)
        report = aggregate(scores)
        report.update(mode="saved_answer_text_rescore", source=str(source),
                      note="Only extraction is rerun. Original perturbation answers and source files are unchanged.")
        write_json(root / "report.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None,
                        help="question-only/QA JSONL or shard directory; defaults to fresh bundled questions")
    parser.add_argument("--run-root", required=True, help="new, separate output directory")
    parser.add_argument("--rescore-from", help="rescore saved answers in a completed v1/v2/v3 run without regenerating them")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--diagnostics", action="store_true", help="first-token metrics, zero control, wider sweep, four diagnostic questions")
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=None, help="prefer an immutable model commit")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--layer", type=int, default=None, help="zero-based decoder block; default middle block")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=None)
    parser.add_argument("--baseline-samples", type=int, default=4)
    parser.add_argument("--baseline-temperature", type=float, default=.8)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--equivalence-threshold", type=float, default=.82)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print budget; no model loading or writes")
    args = parser.parse_args()
    if args.rescore_from:
        source_marker, source_rows, _ = saved_run(args.rescore_from, args.max_examples)
        args.model = args.model or source_marker["settings"]["model"]
        if args.dry_run:
            print(json.dumps({"mode": "saved_answer_text_rescore", "examples": len(source_rows),
                              "new_perturbation_generations": 0, "probability_probe_passes": 0,
                              "answer_extraction": "one request per distinct question/response; no factuality labels"}, indent=2))
            return
        report = rescore_saved(Path(args.run_root).resolve(), args.rescore_from, args.max_examples,
            {"model": args.model, "revision": args.revision, "local_files_only": args.local_files_only},
            lambda: HFStabilityBackend(args.model, args.layer, args.device_map, args.torch_dtype,
                                       args.local_files_only, args.revision))
        print(json.dumps({"examples": report["examples"], "report": str(Path(args.run_root).resolve() / "report.json")}, indent=2))
        return
    args.model = args.model or "Qwen/Qwen2.5-3B-Instruct"
    if args.input is None:
        filename = "stability_diagnostic_questions.jsonl" if args.diagnostics else "stability_pilot_questions.jsonl"
        args.input = str(Path(__file__).resolve().parents[1] / "data" / filename)
    if args.max_examples is None:
        args.max_examples = 4 if args.diagnostics else 20
    amplitudes = args.amplitudes if args.amplitudes is not None else (DIAGNOSTIC_AMPLITUDES if args.diagnostics else (.005, .02, .05))
    tokens = args.max_new_tokens if args.max_new_tokens is not None else (64 if args.diagnostics else 24)
    config = PilotConfig(args.rank, args.directions, tuple(amplitudes), args.baseline_samples,
                         args.baseline_temperature, tokens, args.equivalence_threshold, args.seed, args.diagnostics)
    rows = selected_inputs(args.input, args.max_examples)
    settings = {key: getattr(args, key) for key in ("model", "revision", "embedding_model", "embedding_device",
                                                  "device_map", "torch_dtype", "local_files_only", "layer")}
    calls = 1 + int(config.diagnostics) + config.baseline_samples + len(MODES) * len(config.amplitudes) * config.directions
    if args.dry_run:
        print(json.dumps({"examples": len(rows), "labels_used": False,
                          "generation_calls_per_example": calls, "capture_passes_per_example": len(INSTRUCTIONS),
                          "maximum_generated_tokens": len(rows) * calls * config.max_new_tokens,
                          "answer_extraction": "additional cached request per unique response, up to 96 generated tokens each",
                          "answer_probe_passes_per_example": 2 + len(MODES) * len(config.amplitudes) * config.directions if config.diagnostics else 0,
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
