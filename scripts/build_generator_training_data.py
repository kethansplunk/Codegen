"""
Phase 14A — Build SQL Generator Training Data

Combines sql_cot_train.json (question + schema + key_fields + sql) with
top-3 SAR-retrieved similar examples from the ChromaDB index.

Output: Data/generator_data/sql_generator_train.jsonl
Each line: {"text": "<Qwen instruct format>"}  — ready for SFTTrainer.

Usage (local Mac, after Phase 13 ChromaDB build):
    python -m scripts.build_generator_training_data

Usage (Colab, indexes on Drive):
    python -m scripts.build_generator_training_data \
        --chroma_dir /content/drive/MyDrive/codegen/indexes/chroma_sql \
        --sar_model  /content/drive/MyDrive/codegen/checkpoints/sar_sql/sar_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_SYSTEM = (
    "You are an expert SQL query writer. "
    "Given a database schema, key fields, and similar example queries, "
    "generate the correct SQL query for the question."
)

_TOP_K = 4   # retrieve 4; drop self-match → keep 3


def _format_sar_examples(examples: list) -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:\nQ: {ex['question']}\nSQL: {ex['sql']}")
    return "\n\n".join(parts)


def _format_entry(question: str, schema: str, key_fields: list,
                  sar_examples: list, sql: str) -> str:
    kf_str = ", ".join(key_fields) if key_fields else "N/A"
    user = (
        f"## Database Schema\n{schema}\n\n"
        f"## Key Fields\n{kf_str}\n\n"
        f"## Similar Examples\n{_format_sar_examples(sar_examples)}\n\n"
        f"## Question\n{question}\n\n"
        f"Generate the SQL query:"
    )
    return (
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{sql}<|im_end|>"
    )


def build(
    cot_path: str,
    chroma_dir: str,
    sar_model_path: str,
    out_path: str,
    bge_model: str = "BAAI/bge-large-en-v1.5",
    embed_dim: int = 1024,
    checkpoint_every: int = 100,
):
    from src.sar.infer import ChromaSARRetriever

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(cot_path, encoding="utf-8") as f:
        cot_data = json.load(f)
    print(f"CoT entries: {len(cot_data)}")

    print("Loading ChromaSARRetriever ...")
    retriever = ChromaSARRetriever(
        model_path=sar_model_path,
        chroma_dir=chroma_dir,
        collection_name=os.path.basename(chroma_dir.rstrip("/")).replace("chroma_", "sar_"),
        bge_model=bge_model,
        embed_dim=embed_dim,
    )

    # Resume: each entry writes exactly one line, so the number of lines
    # already in the output file is exactly the number of entries processed.
    # Using the line count (not a separate checkpoint counter) keeps the
    # resume point aligned with what was actually written, so an interrupted
    # run never re-appends the entries between the last checkpoint and the crash.
    start_idx = 0
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        print(f"Resuming: {start_idx} entries already written")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, entry in enumerate(cot_data):
            if i < start_idx:
                continue

            question    = entry["question"]
            sql         = entry["sql"]
            schema      = entry.get("schema", "")
            key_fields  = entry.get("key_fields", [])

            # Retrieve top-4, drop self-match
            candidates = retriever.retrieve(question, top_k=_TOP_K)
            sar_examples = [
                c for c in candidates
                if c["question"].strip() != question.strip()
            ][:3]

            text = _format_entry(question, schema, key_fields, sar_examples, sql)
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            fout.flush()

            if (i + 1) % checkpoint_every == 0:
                print(f"  {i + 1}/{len(cot_data)} entries written")

    with open(out_path, encoding="utf-8") as f:
        total = sum(1 for _ in f)
    print(f"\nDone. {total} entries → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot",        default="Data/cot_data/sql_cot_train.json")
    parser.add_argument("--chroma_dir", default="indexes/chroma_sql")
    parser.add_argument("--sar_model",  default="models/sar_sql/sar_model.pt")
    parser.add_argument("--out",        default="Data/generator_data/sql_generator_train.jsonl")
    parser.add_argument("--bge",        default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()

    build(
        cot_path=args.cot,
        chroma_dir=args.chroma_dir,
        sar_model_path=args.sar_model,
        out_path=args.out,
        bge_model=args.bge,
    )


if __name__ == "__main__":
    main()
