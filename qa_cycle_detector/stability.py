"""Label-free pilot: semantic sensitivity inside/outside prompt-wording directions.

This module does not import Torch until the real-model backend is constructed.
The geometry and scoring routines can be exercised without model downloads.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .semantics import cluster_embeddings, similarities_to_target


SCHEMA = "wording-stability-pilot-v1"
INSTRUCTIONS = (
    "Answer the factual question briefly. Return only the direct answer.",
    "Give a brief direct answer to the factual question, without explanation.",
    "Respond to the factual question with only a concise answer.",
    "Provide only the answer to the factual question. Be brief.",
    "Keep your answer to the factual question short and direct.",
    "State the answer to the factual question concisely, without commentary.",
)
MODES = ("isotropic", "wording", "orthogonal", "random_wording", "random_orthogonal")


@dataclass(frozen=True)
class PilotConfig:
    rank: int = 3
    directions: int = 4
    amplitudes: tuple[float, ...] = (.005, .02, .05)
    baseline_samples: int = 4
    baseline_temperature: float = .8
    max_new_tokens: int = 24
    equivalence_threshold: float = .82
    seed: int = 42

    def __post_init__(self):
        if min(self.rank, self.directions, self.baseline_samples, self.max_new_tokens) < 1:
            raise ValueError("rank, directions, sample count, and token limit must be positive")
        if (not self.amplitudes or any(not np.isfinite(a) or a <= 0 for a in self.amplitudes)
                or tuple(sorted(set(self.amplitudes))) != self.amplitudes):
            raise ValueError("amplitudes must be unique, positive, finite, and increasing")
        if not np.isfinite(self.baseline_temperature) or self.baseline_temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not -1 <= self.equivalence_threshold <= 1:
            raise ValueError("equivalence threshold must be in [-1, 1]")

    def to_dict(self):
        return asdict(self)


def prompts(question: str) -> list[str]:
    # The factual question is byte-for-byte identical in every view.
    return [f"{instruction}\n\nQuestion: {question}\nAnswer:" for instruction in INSTRUCTIONS]


def wording_basis(states: Any, rank: int) -> np.ndarray:
    x = np.asarray(states, dtype=float)
    if (x.ndim != 2 or len(x) != len(INSTRUCTIONS) or x.shape[1] < 2
            or not np.isfinite(x).all() or rank < 1):
        raise ValueError("expected finite last-prompt-token states for all wording views")
    differences = x[1:] - x[0]
    # Anchor-relative displacements: centering again would remove one useful direction.
    _, singular, vt = np.linalg.svd(differences, full_matrices=False)
    available = int((singular > max(float(singular[0]) * 1e-5, 1e-10)).sum())
    actual = min(rank, available, x.shape[1] - 1)
    if actual == 0:
        raise ValueError("wording views have no measurable displacement; cannot identify a subspace")
    return vt[:actual].T


def random_basis(dimension: int, rank: int, seed: int) -> np.ndarray:
    if not 0 < rank < dimension:
        raise ValueError("random-control rank must be between zero and hidden dimension")
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, rank)))
    return q


def perturbation(mode: str, basis: np.ndarray, control: np.ndarray,
                 seed: int, amplitude: float, state_norm: float) -> np.ndarray:
    if mode not in MODES:
        raise ValueError("unknown perturbation condition")
    if not np.isfinite(amplitude) or amplitude < 0 or not np.isfinite(state_norm) or state_norm <= 0:
        raise ValueError("invalid relative perturbation magnitude")
    if basis.shape != control.shape or not 0 < basis.shape[1] < basis.shape[0]:
        raise ValueError("wording and random controls need identical nonzero rank")
    # One base draw per direction, reused across conditions and amplitudes.
    vector = np.random.default_rng(seed).normal(size=basis.shape[0])
    if mode != "isotropic":
        u = control if mode.startswith("random_") else basis
        projected = u @ (u.T @ vector)
        vector = projected if mode in ("wording", "random_wording") else vector - projected
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("degenerate projected noise draw")
    return vector * (amplitude * state_norm / norm)


def semantic_summary(texts: list[str], embeddings: np.ndarray, anchor_embedding: np.ndarray,
                     supplied_embedding: np.ndarray, threshold: float) -> dict[str, Any]:
    valid = [i for i, text in enumerate(texts) if text.strip()]
    result = {"samples": len(texts), "valid_samples": len(valid),
              "invalid_rate": 1 - len(valid) / len(texts),
              "semantic_entropy": None, "anchor_flip_rate": None,
              "supplied_flip_rate": None, "mean_anchor_similarity": None}
    if not valid:
        return result
    vectors = embeddings[valid]
    clusters = cluster_embeddings(vectors, threshold)
    p = np.asarray([len(c) / len(valid) for c in clusters])
    anchor = similarities_to_target(vectors, anchor_embedding)
    supplied = similarities_to_target(vectors, supplied_embedding)
    return {**result, "semantic_entropy": float(-(p * np.log(p)).sum()),
            "cluster_count": len(clusters),
            "anchor_flip_rate": float((anchor < threshold).mean()),
            "supplied_flip_rate": float((supplied < threshold).mean()),
            "mean_anchor_similarity": float(anchor.mean())}


class LastPromptIntervention:
    """One hook on one decoder block's output at the last prompt-token position.

    Register only during prefill. No edits to weights or subsequent decoding states.
    """

    def __init__(self, delta=None, capture=False):
        self.delta = delta
        self.capture = capture
        self.state = None
        self.calls = 0

    def __call__(self, module, inputs, output):
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("intervention must be installed only during one prefill")
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("pilot requires a single unpadded prompt")
        if self.capture:
            self.state = hidden[0, -1].detach().float().cpu().numpy().copy()
        if self.delta is None:
            return output
        edited = hidden.clone()
        # Accelerate may place this block on another device; construct noise here.
        edited[0, -1] = edited[0, -1] + hidden.new_tensor(self.delta)
        return (edited, *output[1:]) if isinstance(output, tuple) else edited


class HFStabilityBackend:
    """Llama/Qwen-style decoder backend using explicit cached autoregressive decoding."""

    def __init__(self, model_name: str, layer: int | None = None, device_map: str = "auto",
                 torch_dtype: str = "auto", local_files_only: bool = False,
                 revision: str | None = None):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision,
                                                       local_files_only=local_files_only)
        dtype = torch_dtype if torch_dtype == "auto" else getattr(torch, torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, device_map=device_map, torch_dtype=dtype,
            local_files_only=local_files_only)
        self.model.eval()
        core = getattr(self.model, "model", None)
        layers = getattr(core, "layers", None)
        if layers is None:
            raise ValueError("pilot supports decoder models exposing model.model.layers (Llama/Qwen)")
        self.layer = len(layers) // 2 if layer is None else layer
        if not 0 <= self.layer < len(layers):
            raise ValueError("layer is a zero-based decoder block index outside the model")
        self.block = layers[self.layer]
        self.device = self.model.get_input_embeddings().weight.device
        self.metadata = {"model": model_name, "layer": self.layer, "layers": len(layers),
                         "resolved_commit": getattr(self.model.config, "_commit_hash", None),
                         "torch": torch.__version__, "transformers": transformers.__version__,
                         "intervention": "decoder-block output, final prompt token, prefill only"}

    def encode(self, prompt):
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                      tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and inputs["input_ids"].shape[1] >= limit:
            raise ValueError("prompt exceeds context window; pilot never silently truncates")
        return inputs.to(self.device)

    def capture(self, prompt):
        hook = LastPromptIntervention(capture=True)
        handle = self.block.register_forward_hook(hook)
        try:
            with self.torch.inference_mode():
                self.model(**self.encode(prompt), use_cache=False)
        finally:
            handle.remove()
        if hook.state is None:
            raise RuntimeError("decoder hook was not invoked")
        return hook.state

    def generate(self, prompt, max_new_tokens, seed, delta=None, temperature=None):
        torch = self.torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        inputs = self.encode(prompt)
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and inputs["input_ids"].shape[1] + max_new_tokens > limit:
            raise ValueError("prompt plus generation exceeds context window")
        hook = LastPromptIntervention(delta=delta)
        handle = self.block.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=True)
        finally:
            handle.remove()
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        tokens, entropies, chosen_logprobs = [], [], []
        attention_mask = inputs["attention_mask"]
        for step in range(max_new_tokens):
            logits = output.logits[:, -1].float()
            logp = torch.log_softmax(logits, dim=-1)
            entropies.append(float(-(logp.exp() * logp).sum().item()))
            if temperature is None:
                token = logits.argmax(dim=-1, keepdim=True)
            else:
                token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
            token_id = int(token.item())
            chosen_logprobs.append(float(logp[0, token_id].item()))
            tokens.append(token_id)
            if token_id in eos or step == max_new_tokens - 1:
                break
            if output.past_key_values is None:
                raise ValueError("model must support cached decoding")
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=-1)
            with torch.inference_mode():
                output = self.model(input_ids=token.to(self.device),
                                    attention_mask=attention_mask, past_key_values=output.past_key_values,
                                    use_cache=True)
        return {"text": self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": len(tokens),
                "mean_token_entropy": float(np.mean(entropies)),
                "mean_token_logprob": float(np.mean(chosen_logprobs)),
                "truncated": tokens[-1] not in eos, "intervention_calls": hook.calls}
