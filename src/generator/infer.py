"""
Phase 14A — SQL Generator inference

Loads the fine-tuned Qwen2.5-Coder-7B-Instruct LoRA adapter and generates
SQL queries given question + schema + key_fields + SAR examples.

Used by the end-to-end pipeline (Phase 16) and POSG (Phase 15A).
"""

from __future__ import annotations

from typing import List

_SYSTEM = (
    "You are an expert SQL query writer. "
    "Given a database schema, key fields, and similar example queries, "
    "generate the correct SQL query for the question."
)


def _build_user_prompt(question: str, schema: str,
                        key_fields: List[str], sar_examples: List[dict]) -> str:
    kf_str = ", ".join(key_fields) if key_fields else "N/A"
    ex_parts = []
    for i, ex in enumerate(sar_examples, 1):
        ex_parts.append(f"Example {i}:\nQ: {ex['question']}\nSQL: {ex['sql']}")
    examples_str = "\n\n".join(ex_parts) if ex_parts else "N/A"
    return (
        f"## Database Schema\n{schema}\n\n"
        f"## Key Fields\n{kf_str}\n\n"
        f"## Similar Examples\n{examples_str}\n\n"
        f"## Question\n{question}\n\n"
        f"Generate the SQL query:"
    )


class GeneratorInfer:
    """
    SQL Generator inference wrapper.

    Args:
        checkpoint_path: Path to the fine-tuned LoRA adapter directory.
        n_candidates:    Number of SQL candidates to generate (for POSG, use 5).
        temperature:     Sampling temperature (0.0 = greedy, 0.8 for diverse k>1).
        max_new_tokens:  Maximum tokens to generate.
    """

    def __init__(
        self,
        checkpoint_path: str,
        n_candidates: int = 1,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ):
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        from src.device import get_device

        self.device      = get_device()
        self.n_candidates = n_candidates
        self.temperature  = temperature
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        question: str,
        schema: str,
        key_fields: List[str],
        sar_examples: List[dict],
    ) -> List[str]:
        """
        Generate SQL candidates.

        Returns:
            List of SQL strings, length == n_candidates.
        """
        import torch

        user_content = _build_user_prompt(question, schema, key_fields, sar_examples)
        messages = [
            {"role": "system",    "content": _SYSTEM},
            {"role": "user",      "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        do_sample = self.temperature > 0 and self.n_candidates > 1
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            num_return_sequences=self.n_candidates,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        results = []
        for ids in output_ids:
            decoded = self.tokenizer.decode(
                ids[input_len:], skip_special_tokens=True
            ).strip()
            results.append(decoded)
        return results
