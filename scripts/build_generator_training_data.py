"""
Phase 14A/14B — Build Generator Training Data (SQL or NoSQL)

Combines the CoT data (question + schema + key_fields + target query) with
top-3 SAR-retrieved similar examples from the ChromaDB index, and writes a
Qwen-instruct JSONL ready for src/generator/train.py.

    --track sql   → label is the SQL string;      examples shown as "SQL: ..."
    --track nosql → label is the MQL JSON dict;    examples shown as "MQL: ..."

Usage (SQL, Phase 14A):
    python -m scripts.build_generator_training_data --track sql \
        --cot        Data/cot_data/sql_cot_train.json \
        --chroma_dir indexes/chroma_sql \
        --sar_model  models/sar_sql/sar_model.pt \
        --out        Data/generator_data/sql_generator_train.jsonl

Usage (NoSQL, Phase 14B):
    python -m scripts.build_generator_training_data --track nosql \
        --cot        Data/cot_data/nosql_cot_train.json \
        --chroma_dir indexes/chroma_nosql \
        --sar_model  models/sar_nosql/sar_model.pt \
        --out        Data/generator_data/nosql_generator_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

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

_TOP_K = 4   # retrieve 4; drop self-match → keep 3


# ---------------------------------------------------------------------------
# Track-specific formatting
# ---------------------------------------------------------------------------

def _example_query(ex: dict, track: str) -> str:
    """Render a retrieved SAR example's query for the prompt."""
    if track == "sql":
        return f"SQL: {ex['sql']}"
    mql = {"collection": ex.get("mql_collection", ""),
           "pipeline":   ex.get("mql_pipeline", [])}
    return f"MQL: {json.dumps(mql, ensure_ascii=False)}"


def _target_query(entry: dict, track: str) -> str:
    """The label the model must produce."""
    if track == "sql":
        return entry["sql"]
    return json.dumps(entry["mql"], ensure_ascii=False)


def _format_entry(question: str, schema: str, key_fields: list,
                  sar_examples: list, target: str, track: str) -> str:
    kf_str = ", ".join(key_fields) if key_fields else "N/A"
    ex_parts = [
        f"Example {i}:\nQ: {ex['question']}\n{_example_query(ex, track)}"
        for i, ex in enumerate(sar_examples, 1)
    ]
    examples_str = "\n\n".join(ex_parts) if ex_parts else "N/A"
    verb = "SQL query" if track == "sql" else "MongoDB query"
    user = (
        f"## Database Schema\n{schema}\n\n"
        f"## Key Fields\n{kf_str}\n\n"
        f"## Similar Examples\n{examples_str}\n\n"
        f"## Question\n{question}\n\n"
        f"Generate the {verb}:"
    )
    return (
        f"<|im_start|>system\n{_SYSTEM[track]}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(
    cot_path: str,
    chroma_dir: str,
    sar_model_path: str,
    out_path: str,
    track: str,
    bge_model: str = "BAAI/bge-large-en-v1.5",
    embed_dim: int = 1024,
    checkpoint_every: int = 100,
):
    from src.sar.infer import ChromaSARRetriever

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(cot_path, encoding="utf-8") as f:
        cot_data = json.load(f)
    print(f"CoT entries: {len(cot_data)}  (track={track})")

    print("Loading ChromaSARRetriever ...")
    retriever = ChromaSARRetriever(
        model_path=sar_model_path,
        chroma_dir=chroma_dir,
        collection_name=os.path.basename(chroma_dir.rstrip("/")).replace("chroma_", "sar_"),
        bge_model=bge_model,
        embed_dim=embed_dim,
    )

    # Resume: each entry writes exactly one line, so the number of lines already
    # in the output file is exactly the number of entries processed. Using the
    # line count (not a separate checkpoint counter) keeps the resume point
    # aligned with what was actually written.
    start_idx = 0
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        print(f"Resuming: {start_idx} entries already written")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, entry in enumerate(cot_data):
            if i < start_idx:
                continue

            question   = entry["question"]
            schema     = entry.get("schema", "")
            key_fields = entry.get("key_fields", [])
            target     = _target_query(entry, track)

            # Retrieve top-4, drop self-match
            candidates = retriever.retrieve(question, top_k=_TOP_K)
            sar_examples = [
                c for c in candidates
                if c["question"].strip() != question.strip()
            ][:3]

            text = _format_entry(question, schema, key_fields, sar_examples, target, track)
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            fout.flush()

            if (i + 1) % checkpoint_every == 0:
                print(f"  {i + 1}/{len(cot_data)} entries written")

    with open(out_path, encoding="utf-8") as f:
        total = sum(1 for _ in f)
    print(f"\nDone. {total} entries → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["sql", "nosql"], default="sql")
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
        track=args.track,
        bge_model=args.bge,
    )


if __name__ == "__main__":
    main()
