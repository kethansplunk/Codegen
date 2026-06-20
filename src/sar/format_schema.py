"""
Schema text parser for SAR training data preparation.
Adapted from SchemaRAG SAR_train/format_schema.py:
- parse_database_schema: parses our schema text into {tables, columns} dict.
- Used to enrich each CoT entry with a structured 'schema' field that SAR
  training consumes for embedding lookup.
- Also used during SAR inference to map schema text → table/column lists.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def parse_database_schema(schema_text: str) -> Dict[str, Any]:
    """
    Parse schema text into structured form.

    Input format (from build_cot_data.format_schema_text):
        # Table: actor
        [
        (actor_id:INT, Primary Key, Examples: [1, 2]),
        (name:TEXT, Examples: [Tom Hanks]),
        ]

    Returns:
        {"tables": ["actor", ...], "columns": {"actor": ["actor_id", "name"], ...}}
    """
    schema: Dict[str, Any] = {"tables": [], "columns": {}}
    table_start_pattern = r"# Table: (\w+)\s*\["
    table_starts = [
        {"name": m.group(1), "start": m.start(), "bracket_start": m.end() - 1}
        for m in re.finditer(table_start_pattern, schema_text)
    ]

    for i, current in enumerate(table_starts):
        search_start = current["bracket_start"] + 1
        search_end   = table_starts[i + 1]["start"] if i + 1 < len(table_starts) else len(schema_text)
        search_text  = schema_text[search_start:search_end]

        end_match = re.search(r"^\s*\]", search_text, re.MULTILINE)
        if not end_match:
            continue

        table_content = schema_text[current["bracket_start"] + 1 : search_start + end_match.start()]
        columns = re.findall(r"\((\w+):", table_content)

        schema["tables"].append(current["name"])
        schema["columns"][current["name"]] = columns

    return schema


def enrich_with_schema(items: List[Dict]) -> List[Dict]:
    """
    Add a structured 'schema' field to each item that has a 'database' text field.
    Used to prepare CoT / RAG data for SAR training.
    """
    for item in items:
        db_text = item.get("schema") or item.get("database", "")
        if db_text:
            item["parsed_schema"] = parse_database_schema(db_text)
    return items


def process_json_file(input_file: str, output_file: str | None = None):
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list of dicts")

    data = enrich_with_schema(data)

    out = output_file or input_file
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Enriched {len(data)} items → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    process_json_file(args.input, args.output)
