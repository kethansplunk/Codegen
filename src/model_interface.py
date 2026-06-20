"""
Local Qwen model interface for inference.
Adapted from SchemaRAG llm_local.py:
- Replaced modelscope with transformers (standard HuggingFace)
- Added MPS / CUDA / CPU device detection via src/device.py
- enable_thinking flag preserved for Qwen3 compatibility
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.device import get_device


class ModelInterface:
    def __init__(self, model_path: str, max_new_tokens: int = 32768):
        self.model_path    = model_path
        self.max_new_tokens = max_new_tokens
        self.device        = get_device()
        self.tokenizer     = None
        self.model         = None
        self._load()

    def _load(self):
        print(f"Loading model from {self.model_path} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded.")

    def generate(
        self,
        instruct: str,
        prompt: str,
        n: int = 1,
        num_beams: int = 1,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate n responses.

        Args:
            instruct: System message.
            prompt: User message.
            n: Number of sequences to return (requires num_beams >= n).
            num_beams: Beam width. Set to n for beam search in POSG.
            enable_thinking: Passes enable_thinking to Qwen3 chat template.
        """
        messages = [
            {"role": "system", "content": instruct},
            {"role": "user",   "content": prompt},
        ]

        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking:
            template_kwargs["enable_thinking"] = True

        text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        gen_params: Dict[str, Any] = {
            "max_new_tokens":      kwargs.get("max_new_tokens", self.max_new_tokens),
            "num_beams":           max(num_beams, n),
            "num_return_sequences": n,
            "early_stopping":      kwargs.get("early_stopping", True),
            "temperature":         kwargs.get("temperature", 1.0),
            "repetition_penalty":  kwargs.get("repetition_penalty", 1.1),
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_params)

        input_len = inputs.input_ids.shape[1]
        results = []
        for out in generated_ids:
            decoded = self.tokenizer.decode(
                out[input_len:], skip_special_tokens=True
            ).strip()
            results.append(decoded)
        return results
