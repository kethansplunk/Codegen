"""
Unified SchemaLinker interface — switchable between DeepSeek API and trained model.

mode="api"   → Calls DeepSeek API (primary). No local GPU required.
              Set DEEPSEEK_API_KEY in environment or .env file.

mode="model" → Loads a trained PEFT adapter (Stage 1/2/3 checkpoint).
              Requires GPU and a completed training run.

Usage:
    from src.schema_linker.linker import get_schema_linker

    linker = get_schema_linker(config["schema_linker"])
    key_fields = linker.link(question="...", schema="...")
    # returns e.g. ["singer.Singer_ID", "concert.Concert_ID"]

Config keys (configs/config.yaml → schema_linker):
    mode:           "api" or "model"
    api_model:      DeepSeek model name  (api mode only)
    api_key_env:    env-var name holding the API key  (api mode only)
    sql_checkpoint: path to trained SQL adapter  (model mode only)
    nosql_checkpoint: path to trained NoSQL adapter  (model mode only)
    max_retries:    retry count on parse failure  (both modes)
"""

from __future__ import annotations

import os
import re
import time
from typing import List


# ---------------------------------------------------------------------------
# Shared output parser (identical to src/schema_linker/infer.py)
# ---------------------------------------------------------------------------

def _extract_key_fields(text: str) -> List[str] | None:
    """Return list of 'table.col' strings, or None if parsing fails."""
    last = text.rfind("The key field")
    if last == -1:
        return None
    remaining = text[last:]
    start = remaining.find("[")
    end   = remaining.find("]")
    if start == -1 or end == -1 or end <= start:
        return None
    items = [s.strip().strip('"').strip("'") for s in remaining[start + 1:end].split(",")]
    result = [s for s in items if s]
    return result if result else None


# ---------------------------------------------------------------------------
# API backend (DeepSeek)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = "You are a Schema Linking Expert. Identify which tables and columns are needed to answer a question."

_USER_TEMPLATE = """\
# Question: {question}
# Database: {schema}

Think step by step inside <think>...</think> tags, then identify the key fields.

IMPORTANT: End your response with EXACTLY this line (keep the square brackets):
The key field matching the question is: [table.column1, table.column2]"""


class ApiSchemaLinker:
    """Schema linking via DeepSeek API. No GPU or trained model required."""

    def __init__(self, api_model: str = "deepseek-chat", api_key_env: str = "DEEPSEEK_API_KEY", max_retries: int = 3):
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"DeepSeek API key not found. Set {api_key_env} in your environment or .env file."
            )
        self.client     = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.api_model  = api_model
        self.max_retries = max_retries

    def link(self, question: str, schema: str) -> List[str]:
        """
        Return key_fields for the given question + schema.

        Retries up to max_retries times if the response cannot be parsed.
        Returns [] if all retries fail.
        """
        prompt = _USER_TEMPLATE.format(question=question, schema=schema)

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=1024,
                )
                text   = resp.choices[0].message.content or ""
                fields = _extract_key_fields(text)
                if fields is not None:
                    return fields
                print(f"[SchemaLinker API] Parse failed on attempt {attempt}/{self.max_retries}")
            except Exception as e:
                print(f"[SchemaLinker API] Error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        print("[SchemaLinker API] All retries failed — returning empty key_fields")
        return []


# ---------------------------------------------------------------------------
# Model backend (trained PEFT adapter)
# ---------------------------------------------------------------------------

class ModelSchemaLinker:
    """
    Schema linking via a fine-tuned PEFT adapter.
    Wraps ModelInterface + the existing run_schema_linker logic.
    Requires a completed Stage 1/2/3 training run and a GPU.
    """

    def __init__(self, checkpoint_path: str, max_retries: int = 3):
        # Lazy import — only pulls in torch/transformers when model mode is selected
        from src.model_interface import ModelInterface
        self.model       = ModelInterface(checkpoint_path)
        self.max_retries = max_retries

    def link(self, question: str, schema: str) -> List[str]:
        system = "You are a Schema Linking Expert"
        prompt = f"# Question: {question}\n# Database: \"{schema}\""

        for attempt in range(1, self.max_retries + 1):
            outputs = self.model.generate(system, prompt, n=1)
            text    = outputs[0] if outputs else ""
            fields  = _extract_key_fields(text)
            if fields is not None:
                return fields
            print(f"[SchemaLinker Model] Parse failed on attempt {attempt}/{self.max_retries}")

        print("[SchemaLinker Model] All retries failed — returning empty key_fields")
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_schema_linker(sl_config: dict, track: str = "sql"):
    """
    Build and return the appropriate SchemaLinker from config.

    Args:
        sl_config: The schema_linker section of configs/config.yaml.
        track:     "sql" or "nosql" — selects which checkpoint when mode=model.

    Returns:
        ApiSchemaLinker  when sl_config["mode"] == "api"
        ModelSchemaLinker when sl_config["mode"] == "model"
    """
    mode = sl_config.get("mode", "api")
    max_retries = sl_config.get("max_retries", 3)

    if mode == "api":
        return ApiSchemaLinker(
            api_model=sl_config.get("api_model", "deepseek-chat"),
            api_key_env=sl_config.get("api_key_env", "DEEPSEEK_API_KEY"),
            max_retries=max_retries,
        )

    if mode == "model":
        key = "sql_checkpoint" if track == "sql" else "nosql_checkpoint"
        checkpoint = sl_config.get(key)
        if not checkpoint:
            raise ValueError(f"schema_linker.{key} not set in config.yaml")
        return ModelSchemaLinker(checkpoint_path=checkpoint, max_retries=max_retries)

    raise ValueError(f"Unknown schema_linker.mode '{mode}'. Must be 'api' or 'model'.")
