from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .config import GenerationConfig


class TextGenerator(Protocol):
    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> list[list[str]]:
        """Return one list of sampled completions per prompt."""


@dataclass
class HuggingFaceGenerator:
    """A small chat/text generation adapter with batched sampling.

    Heavy ML imports and model loading happen only when this class is created, so
    feature math and unit tests remain lightweight.
    """

    model_name: str
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    seed: int = 42
    local_files_only: bool = False

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = self.torch_dtype
        self.reseed(self.seed)
        if dtype != "auto":
            dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def reseed(self, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _format(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> list[list[str]]:
        import torch

        formatted = [self._format(prompt) for prompt in prompts]
        encoded = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                num_return_sequences=config.samples,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decoder-only generation returns the full padded input prefix for every
        # sequence. Slice at the padded width, not at each unpadded token count.
        input_width = encoded["input_ids"].shape[1]
        decoded: list[list[str]] = []
        cursor = 0
        for _ in formatted:
            group: list[str] = []
            for _ in range(config.samples):
                tokens = output[cursor, input_width:]
                group.append(self.tokenizer.decode(tokens, skip_special_tokens=True).strip())
                cursor += 1
            decoded.append(group)
        return decoded


ANSWER_TYPES = {
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "DATE",
    "NUMBER",
    "WORK",
    "EVENT",
    "OTHER",
}


def _normalized_value(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE))


def parse_structured_answer(
    text: str, expected_answer: str | None = None
) -> dict[str, Any]:
    """Parse ``ANSWER || TYPE || QUALIFIER`` without silently changing A0."""

    result: dict[str, Any] = {
        "valid": False,
        "raw": text,
        "value": "",
        "type": "",
        "qualifier": "",
        "failure_reason": "missing_delimiters",
    }
    candidates = [line.strip(" `\t") for line in text.splitlines() if "||" in line]
    if not candidates:
        return result
    for candidate in candidates:
        parts = [part.strip() for part in candidate.split("||")]
        if len(parts) != 3 or not all(parts):
            result["failure_reason"] = "expected_three_nonempty_fields"
            continue
        value, answer_type, qualifier = parts
        value = re.sub(
            r"^(?:answer|value)\s*:\s*", "", value, flags=re.IGNORECASE
        )
        answer_type = answer_type.upper()
        result.update(value=value, type=answer_type, qualifier=qualifier)
        if answer_type not in ANSWER_TYPES:
            result["failure_reason"] = "invalid_type"
            continue
        if expected_answer is not None and _normalized_value(
            value
        ) != _normalized_value(expected_answer):
            result["failure_reason"] = "answer_value_changed"
            continue
        if len(qualifier.split()) > 8:
            result["failure_reason"] = "qualifier_too_long"
            continue
        result["valid"] = True
        result["failure_reason"] = None
        break
    return result


def answer_prompt(question: str, answer_format: str = "sentence") -> str:
    if answer_format == "sentence":
        return f"Answer the question concisely in one sentence.\n\nQuestion: {question}"
    if answer_format == "structured":
        return (
            "Answer the factual question using exactly one line in this format:\n"
            "ANSWER || TYPE || QUALIFIER\n\n"
            "TYPE must be one of PERSON, LOCATION, ORGANIZATION, DATE, NUMBER, "
            "WORK, EVENT, OTHER. QUALIFIER must contain at most eight words and "
            "identify the answer itself. Do not explain, repeat, or paraphrase the "
            "question.\n\n"
            f"Question: {question}\nAnswer:"
        )
    if answer_format != "short":
        raise ValueError("answer_format must be 'short', 'sentence', or 'structured'")
    return (
        "Answer the factual question directly and concisely. Return only the answer, "
        f"without commentary.\n\nQuestion: {question}\nAnswer:"
    )


def canonicalization_prompt(question: str, answer: str) -> str:
    """Ask for a structured representation while requiring A0 to stay unchanged."""

    return (
        "Reformat the supplied answer using exactly one line:\n"
        "ANSWER || TYPE || QUALIFIER\n\n"
        "Copy ANSWER exactly from SUPPLIED ANSWER. TYPE must be one of PERSON, "
        "LOCATION, ORGANIZATION, DATE, NUMBER, WORK, EVENT, OTHER. QUALIFIER must "
        "contain at most eight words and identify the answer itself. Do not correct "
        "the answer. Do not explain, repeat, or paraphrase the question.\n\n"
        f"Question: {question}\nSUPPLIED ANSWER: {answer}\nFormatted answer:"
    )


def inverse_prompt(answer: str) -> str:
    # The original question must never be included: doing so would leak the target.
    return (
        "Generate one concise factual question for which the text below would be a "
        "correct direct answer. Return only the question.\n\n"
        f"Answer: {answer}\nQuestion:"
    )
