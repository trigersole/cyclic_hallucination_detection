"""Label-free question-conditioned candidate trajectories; independent of stability runs."""
from __future__ import annotations

import numpy as np

from .stability import HFStabilityBackend

SCHEMA = "question-layer-competition-v1"


def answer_prompt(question):
    return ("Answer the factual question briefly. Return only the direct answer, "
            "without explanation.\n\nQuestion: " + question + "\nAnswer:")


def validate_rows(rows):
    """Explicit controls are required; never infer key details or consume truth labels."""
    selected, seen = [], set()
    for index, row in enumerate(rows):
        question = row.get("question")
        if not isinstance(question, str) or not question.strip() or question in seen:
            raise ValueError("questions must be nonempty and unique")
        seen.add(question)
        controls, kinds, questions = [], set(), {question}
        for control in row.get("controls", []):
            kind, text = control.get("kind"), control.get("question")
            if kind not in {"wording", "edited", "ablated"}:
                raise ValueError("control kind must be wording, edited, or ablated")
            if not isinstance(text, str) or not text.strip() or text in questions:
                raise ValueError("control questions must be nonempty and distinct")
            questions.add(text)
            kinds.add(kind)
            controls.append({"kind": kind, "question": text})
        if kinds != {"wording", "edited", "ablated"}:
            raise ValueError("each question requires wording, edited, and ablated controls")
        selected.append({"id": str(row.get("id", index)), "question": question, "controls": controls})
    if not selected or len({r["id"] for r in selected}) != len(selected):
        raise ValueError("provide at least one example with unique IDs")
    return selected


def summarize_trajectories(matrix, target, layer_ids):
    """Scores are rankings of length-normalized likelihood, not answer probabilities."""
    values = np.asarray(matrix, dtype=float)
    if (values.ndim != 2 or not np.isfinite(values).all()
            or values.shape[0] != len(layer_ids) or values.shape[1] < 1
            or not 0 <= target < values.shape[1]):
        raise ValueError("invalid layer-by-candidate matrix")
    if values.shape[1] < 2:
        return {"valid": False, "reason": "fewer than two distinct candidates"}
    winners = values.argmax(axis=1)
    tied = np.sum(values == values.max(axis=1, keepdims=True), axis=1) > 1
    winners[tied] = -1  # Unresolved ties must not silently pick candidate zero.
    margins = values[:, target] - np.max(np.delete(values, target, axis=1), axis=1)
    # Ties are not wins. No confidence threshold or layer selection is fitted.
    late = slice(2 * len(values) // 3, None)
    stable = next((i for i in range(len(values)) if np.all(margins[i:] > 0)), None)
    return {"valid": True, "winner_indices": winners.tolist(),
            "target_margin": margins.tolist(),
            "winner_switches": int(np.count_nonzero(np.diff(winners))),
            "target_win_fraction": float(np.mean(margins > 0)),
            "late_target_margin": float(margins[late].mean()),
            "stable_target_from_layer": layer_ids[stable] if stable is not None else None}


def compare_questions(probes, controls, candidate_indices, target):
    """Original-minus-control support removes each candidate's shared continuation bias."""
    if target is None or target not in candidate_indices or len(candidate_indices) < 2:
        return {"valid": False, "reason": "anchor missing or fewer than two distinct candidates"}
    local_target = candidate_indices.index(target)
    layer_ids = probes[0][0]["layer_ids"]
    for view in probes:
        for p in view:
            if p["layer_ids"] != layer_ids or not p["final_readout_check_passed"]:
                raise ValueError("layer alignment or final readout validation failed")
    output = {"valid": True, "layer_ids": layer_ids, "candidate_indices": candidate_indices,
              "target_index": target, "metrics": {}}
    for metric in ("mean_logprob", "sum_logprob", "first_content_logprob"):
        matrices = [np.asarray([v[c][metric] for c in candidate_indices]).T for v in probes]
        original = matrices[0]
        values = {"original": summarize_trajectories(original, local_target, layer_ids),
                  "controls": []}
        for i, control in enumerate(controls, 1):
            gain = original - matrices[i]
            values["controls"].append({"kind": control["kind"],
                "control_ranking": summarize_trajectories(matrices[i], local_target, layer_ids),
                "original_minus_control": summarize_trajectories(gain, local_target, layer_ids),
                "target_logprob_gain": gain[:, local_target].tolist()})
        changed = [i for i, c in enumerate(controls, 1) if c["kind"] == "edited"]
        specificity = original - np.mean([matrices[i] for i in changed], axis=0)
        values["edited_detail_specificity"] = summarize_trajectories(specificity, local_target, layer_ids)
        output["metrics"][metric] = values
    return output


class HFLayerCompetitionBackend(HFStabilityBackend):
    """Llama/Qwen logit lens: final norm + output head, with no learned probe."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model.config.model_type not in {"llama", "qwen2", "qwen3"}:
            raise ValueError("layer pilot currently supports Llama/Qwen2/Qwen3 only")
        if not hasattr(self.model.model, "norm"):
            raise ValueError("decoder must expose its final normalization")
        self.metadata.pop("intervention", None)
        self.metadata.update(readout="block output -> final norm -> output head; no interventions",
                             lens="untrained logit lens; intermediate values are uncalibrated")

    def score_candidate(self, question, candidate):
        torch, core = self.torch, self.model.model
        inputs = self.encode(answer_prompt(question))
        prompt_length = inputs["input_ids"].shape[1]
        # Identical candidate token IDs in every question condition. No retokenized
        # question/answer concatenation or independently generated answer prefixes.
        ids = self.tokenizer.encode(" " + candidate.strip(), add_special_tokens=False)
        if not ids:
            raise ValueError("candidate has no tokens")
        decode = lambda x: self.tokenizer.decode(x, skip_special_tokens=True)
        first = next((i for i in range(len(ids)) if decode(ids[:i + 1]).strip()), None)
        if first is None or decode(ids).strip() != candidate.strip():
            raise ValueError("candidate tokenizer does not round-trip")
        suffix = inputs["input_ids"].new_tensor([ids[:-1]])
        inputs["input_ids"] = torch.cat([inputs["input_ids"], suffix], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and inputs["input_ids"].shape[1] > limit:
            raise ValueError("candidate exceeds context window; no silent truncation")
        positions = list(range(prompt_length - 1, prompt_length - 1 + len(ids)))
        states, handles = {}, []
        def capture(index):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                states[index] = hidden[0, positions].detach().cpu()
            return hook
        try:
            for index, block in enumerate(core.layers):
                handles.append(block.register_forward_hook(capture(index)))
            with torch.inference_mode():
                # Avoid materializing vocabulary logits for every prompt position.
                result = core(**inputs, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if len(states) != len(core.layers):
            raise RuntimeError("not all decoder blocks were observed")
        head, norm = self.model.get_output_embeddings(), core.norm
        head_device, norm_device = head.weight.device, next(norm.parameters()).device
        targets = torch.tensor(ids, device=head_device)
        def project(state, normalize):
            with torch.inference_mode():
                if normalize:
                    state = norm(state.to(norm_device))
                logp = torch.log_softmax(head(state.to(head_device)).float(), dim=-1)
                return logp.gather(1, targets[:, None]).squeeze(1).cpu().numpy()
        logps = np.asarray([project(states[i], True) for i in range(len(core.layers))])
        reference = project(result.last_hidden_state[0, positions], False)
        error = float(np.max(np.abs(logps[-1] - reference)))
        if not np.isfinite(logps).all() or error > 1e-3:
            raise ValueError(f"final layer readout does not match decoder output: {error}")
        return {"layer_ids": list(range(1, len(core.layers) + 1)),
                "token_ids": ids, "token_texts": [decode([t]) for t in ids],
                "first_content_index": first, "token_logprobs": logps.tolist(),
                "mean_logprob": logps[:, first:].mean(axis=1).tolist(),
                "sum_logprob": logps[:, first:].sum(axis=1).tolist(),
                "first_content_logprob": logps[:, first].tolist(),
                "final_readout_max_error": error, "final_readout_check_passed": True}
