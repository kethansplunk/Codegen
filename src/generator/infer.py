"""
Phase 14A — SQL Generator inference

Loads the fine-tuned Qwen2.5-Coder-7B-Instruct LoRA adapter and generates
SQL queries given question + schema + key_fields + SAR examples.

Used by the end-to-end pipeline (Phase 16) and POSG (Phase 15A).
"""

from __future__ import annotations

import importlib
import sys

from typing import List


def _disable_peft_torchao():
    """
    Stop peft's LoRA dispatcher raising on Colab's old torchao (0.10.0).

    peft's is_torchao_available() raises on any torchao < 0.16.0, and its LoRA
    dispatch_torchao() calls it while injecting the adapter. Masking torchao in
    sys.modules is fragile — transformers re-imports the real torchao during
    model load — so we stub the checker to return False in the modules that use
    it. dispatch_torchao() then short-circuits and standard LoRA is used. We do
    not use torchao (A100 = bf16, T4 = bitsandbytes). Call before loading the
    adapter, after peft is importable.
    """
    sys.modules["torchao"] = None
    for mod_name in ("peft.import_utils", "peft.tuners.lora.torchao"):
        try:
            importlib.import_module(mod_name).is_torchao_available = lambda: False
        except Exception:
            pass

# Must match the prompts built in scripts/build_generator_training_data.py
_SYSTEM = {
    "sql": (
        "You are an expert SQL query writer. "
        "Given a database schema, key fields, and similar example queries, "
        "generate the correct SQL query for the question."
    ),
    "nosql": (
        "You are an expert MongoDB query writer. "
        "Given a database schema, key fields, and similar example pipelines, "
        "generate the correct MongoDB aggregation query for the question as a "
        'JSON object: {"collection": <name>, "pipeline": [<stages>]}.'
    ),
}


def _example_query(ex: dict, track: str) -> str:
    """Render a retrieved SAR example's query, matching the training format."""
    if track == "sql":
        return f"SQL: {ex['sql']}"
    import json
    mql = {"collection": ex.get("mql_collection", ""),
           "pipeline":   ex.get("mql_pipeline", [])}
    return f"MQL: {json.dumps(mql, ensure_ascii=False)}"


def _build_user_prompt(question: str, schema: str, key_fields: List[str],
                       sar_examples: List[dict], track: str) -> str:
    kf_str = ", ".join(key_fields) if key_fields else "N/A"
    ex_parts = [
        f"Example {i}:\nQ: {ex['question']}\n{_example_query(ex, track)}"
        for i, ex in enumerate(sar_examples, 1)
    ]
    examples_str = "\n\n".join(ex_parts) if ex_parts else "N/A"
    verb = "SQL query" if track == "sql" else "MongoDB query"
    return (
        f"## Database Schema\n{schema}\n\n"
        f"## Key Fields\n{kf_str}\n\n"
        f"## Similar Examples\n{examples_str}\n\n"
        f"## Question\n{question}\n\n"
        f"Generate the {verb}:"
    )


class GeneratorInfer:
    """
    Generator inference wrapper (SQL or NoSQL).

    Args:
        checkpoint_path: Path to the fine-tuned LoRA adapter directory.
        track:           "sql" (default) or "nosql" — selects the prompt format
                         and system message, which must match how the adapter
                         was trained.
        n_candidates:    Number of candidates to generate (for POSG, use 5).
        temperature:     Sampling temperature (0.0 = greedy, 0.8 for diverse k>1).
        max_new_tokens:  Maximum tokens to generate.
        seed:            If set, pins torch's RNG once at construction time so
                         a run's full sequence of generate() calls is
                         reproducible. Unset by default -- candidate sampling
                         (do_sample=True, temperature>0) is otherwise
                         unseeded, so identical inputs can produce different
                         candidates and different EX numbers between runs
                         (confirmed varying by several points at n=30 in
                         docs/phase15_posg_findings.md).
    """

    def __init__(
        self,
        checkpoint_path: str,
        track: str = "sql",
        n_candidates: int = 1,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        seed: int | None = None,
    ):
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        from src.device import get_device

        _disable_peft_torchao()   # must run before AutoPeftModelForCausalLM loads the adapter

        if seed is not None:
            torch.manual_seed(seed)

        self.track       = track
        self.device      = get_device()
        self.n_candidates = n_candidates
        self.temperature  = temperature
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if self.device == "cuda":
            gpu0_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if gpu0_gb < 24:
                # T4-class GPU (16GB): bf16 (~14GB) plus SAR retriever + KV cache
                # for n_candidates parallel sequences doesn't fit, so
                # device_map="auto" silently offloads layers to CPU/disk instead
                # of erroring. CPU-side generation for a 7B model then takes
                # minutes per candidate instead of seconds -- looks hung, isn't.
                # int8 quantization (~7GB) fits with room to spare, so the whole
                # model stays on GPU. Requires the `bitsandbytes` package.
                from transformers import BitsAndBytesConfig
                load_kwargs.pop("torch_dtype")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["device_map"] = "auto"
            else:
                # A100-class GPU (40GB+): full bf16 fits without offload or
                # quantization. device_map="auto" + offload_folder is kept as a
                # safety net (raises instead of erroring if it somehow doesn't
                # fit) -- offload_folder uses mkdtemp (not a fixed /tmp path)
                # since /tmp is world-writable and a predictable path is a
                # symlink-attack risk.
                import tempfile
                offload_dir = tempfile.mkdtemp(prefix="generator_offload_")
                load_kwargs["device_map"] = "auto"
                load_kwargs["offload_folder"] = offload_dir

                # On multi-GPU boxes (e.g. Kaggle's T4 x2 -- both under 24GB, so
                # they'd hit the quantized branch above instead, but a mixed or
                # multi-A100 host could still land here), accelerate's "auto"
                # balancer only looks at static weight size when placing layers
                # -- since the ~13GB model technically fits on one card, it
                # happily crams everything onto GPU 0 and leaves the rest empty,
                # then OOMs later once the KV cache for n_candidates parallel
                # sequences grows during actual generation. Capping max_memory
                # per GPU forces a real multi-GPU split instead.
                n_gpus = torch.cuda.device_count()
                if n_gpus > 1:
                    max_memory = {}
                    for i in range(n_gpus):
                        total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                        max_memory[i] = f"{total_gb * 0.6:.1f}GiB"   # leave headroom for KV cache
                    load_kwargs["max_memory"] = max_memory

        self.model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path, **load_kwargs)
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
        Generate query candidates (SQL strings, or MQL JSON strings for nosql).

        Returns:
            List of query strings, length == n_candidates.
        """
        import torch

        user_content = _build_user_prompt(question, schema, key_fields, sar_examples, self.track)
        messages = [
            {"role": "system",    "content": _SYSTEM[self.track]},
            {"role": "user",      "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Greedy decoding (do_sample=False) cannot produce more than one
        # distinct sequence — transformers raises if num_return_sequences>1
        # without sampling. So sample whenever multiple candidates are asked
        # for, falling back to a small temperature if the caller passed 0.
        if self.n_candidates > 1:
            do_sample   = True
            temperature = self.temperature if self.temperature > 0 else 0.8
        else:
            do_sample   = self.temperature > 0
            temperature = self.temperature if do_sample else None

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=self.n_candidates,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        if output_ids is None:
            # model.generate() is documented to always return a tensor or
            # raise -- this should be unreachable. If it fires, something
            # (an accelerate hook, a quantization edge case) is swallowing a
            # real error upstream; surface that here instead of letting an
            # empty/None result silently propagate into a confusing
            # 'NoneType is not iterable' three calls downstream in POSG.
            raise RuntimeError(
                "model.generate() returned None instead of raising or "
                "returning a tensor -- this indicates a suppressed error "
                "during generation. Re-run with CUDA_LAUNCH_BLOCKING=1 for a "
                "synchronous (attributable) stack trace."
            )

        input_len = inputs["input_ids"].shape[1]
        results = []
        for ids in output_ids:
            decoded = self.tokenizer.decode(
                ids[input_len:], skip_special_tokens=True
            ).strip()
            results.append(decoded)
        return results
