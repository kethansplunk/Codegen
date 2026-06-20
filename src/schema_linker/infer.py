"""
SchemaLinker inference.
Adapted from SchemaRAG use_SchemaLinker.py:
- Replaced modelscope with transformers.
- Uses ModelInterface from src/model_interface.py.
- evidence parameter removed (Spider has no evidence; use "" for BIRD).
- Retry loop preserved: keeps generating until a valid key-field list is
  extracted (matches SchemaRAG's while schema_links == 1 pattern).
- Saves think_pre (reasoning trace) alongside predictions.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from src.model_interface import ModelInterface


# ---------------------------------------------------------------------------
# Output parsers  (ported directly from SchemaRAG use_SchemaLinker.py)
# ---------------------------------------------------------------------------

def extract_key_fields(text: str) -> List[str] | int:
    """
    Extract the predicted key fields from model output.
    Returns a list of 'table.col' strings, or 1 if parsing failed.
    """
    last_key = text.rfind("The key field")
    if last_key == -1:
        last_key = text.rfind("The key fields")
        if last_key == -1:
            return 1

    remaining = text[last_key:]
    start     = remaining.find("[")
    end       = remaining.find("]")

    if start == -1 or end == -1 or end < start:
        return 1

    content = remaining[start + 1 : end]
    result  = [item.strip() for item in content.split(",") if item.strip()]
    return result if result else 1


def extract_think_content(text: str) -> str:
    matches = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    return matches[-1].strip() if matches else ""


def extract_answer_content(text: str) -> str:
    match = re.search(r"</think>(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_schema_linker(
    model: ModelInterface,
    items: list,
    output_path: str,
    max_retries: int = 3,
) -> list:
    """
    Run the trained SchemaLinker on a list of items.

    Each item must have:
        question (str)
        database (str)  — schema text from format_schema_text()

    Adds to each item:
        schema_links_pred (list[str])
        think_pre         (str)
        answer_pre        (str)

    Args:
        model:       Loaded ModelInterface pointing to the trained SchemaLinker.
        items:       List of dicts with 'question' and 'database'.
        output_path: Where to save the enriched items (JSON).
        max_retries: How many times to retry on invalid output.
    """
    system = "You are a Schema Linking Expert"

    for idx, item in enumerate(items):
        print(f"Processing {idx + 1}/{len(items)}")

        prompt = (
            f"# Question: {item['question']}\n"
            f"# Database: \"{item['schema']}\""
        )

        schema_links = 1
        tried        = 0
        response     = ""

        while schema_links == 1 and tried < max_retries:
            tried += 1
            outputs  = model.generate(system, prompt, n=1)
            response = outputs[0] if outputs else ""
            schema_links = extract_key_fields(response)

        if schema_links == 1:
            schema_links = []

        item["schema_links_pred"] = schema_links
        item["think_pre"]         = extract_think_content(response)
        item["answer_pre"]        = extract_answer_content(response)

        if (idx + 1) % 5 == 0 or idx == len(items) - 1:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=4)

    print(f"SchemaLinker inference complete. Saved to {output_path}")
    return items
