"""
Phase 15 — dev-set prep for POSG evaluation.

The run_posg_sql.py smoke test can't tell us whether POSG actually helps,
because it runs on Data/cot_data/sql_cot_train.json — data the Generator
was fine-tuned on. Results there were 100% exact-match with unique_candidates
often == 1 even on multi-join/subquery/GROUP BY questions: the model isn't
reasoning, it's recalling, so there's no genuine uncertainty for POSG to
arbitrate. Spider's dev split uses a disjoint set of databases the Generator
has never trained on, so it's the only place real model uncertainty (and
therefore POSG's actual value) can show up.

This script builds the (schema, key_fields) inputs for dev.json the same
way Phase 8A built them for train, with one deliberate difference:
key_fields come from the REAL SchemaLinker (a live prediction), not a
ground-truth/oracle label like the train CoT data has — because at real
inference time there is no oracle, and this file is meant to mirror that.

Usage:
    python -m scripts.build_dev_eval_set --out Data/cot_data/sql_dev_eval.json

    # Quick pass on the first 30 dev questions only:
    python -m scripts.build_dev_eval_set --limit 30 --out Data/cot_data/sql_dev_eval_sample.json

Requires:
    Data/Spider/dev.json                   (question, query, db_id per entry)
    Data/fk_graphs/<db_name>.json          (already built, Phase 5A — all 166 DBs)
    Data/prompt_schema/sql/<db_name>.json  (already built, Phase 6 — all 166 DBs)
    DEEPSEEK_API_KEY in .env (schema_linker.mode: api, the current config default)

Note: schema formatting reads only the two cached JSON files above — it does
NOT need Data/Spider/database/*.sqlite. Getting real EX scores afterward
(via run_posg_sql.py) does need those .sqlite files; without them you still
get exact-match + POSG-vs-greedy divergence stats.
"""

from __future__ import annotations

import argparse
import json
import os

import yaml


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build(
    dev_path: str,
    out_path: str,
    config: dict,
    limit: int | None,
    checkpoint_every: int = 20,
):
    from scripts.build_cot_data import format_schema_text
    from src.schema_linker.linker import get_schema_linker

    with open(dev_path, encoding="utf-8") as f:
        dev = json.load(f)
    if limit:
        dev = dev[:limit]
    print(f"Dev entries: {len(dev)}")

    linker = get_schema_linker(config["schema_linker"], track="sql")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Resume: the output file holds exactly what's been written so far, so
    # its length is the true progress marker (matches build_generator_training_data.py).
    out = []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            out = json.load(f)
        print(f"Resuming: {len(out)} entries already written")

    done_keys = {(e["question"], e["db_name"]) for e in out}
    skipped_no_schema = 0

    for i, entry in enumerate(dev):
        question = entry["question"]
        sql      = entry["query"]
        db_name  = entry["db_id"]

        if (question, db_name) in done_keys:
            continue

        schema_text = format_schema_text(db_name)
        if not schema_text:
            skipped_no_schema += 1
            continue

        key_fields = linker.link(question=question, schema=schema_text)

        out.append({
            "question":   question,
            "sql":        sql,
            "db_name":    db_name,
            "schema":     schema_text,
            "key_fields": key_fields,
        })

        if (i + 1) % checkpoint_every == 0 or i == len(dev) - 1:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"  {i + 1}/{len(dev)} processed "
                  f"({len(out)} written, {skipped_no_schema} skipped: no cached schema)")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(out)} dev entries -> {out_path}  "
          f"({skipped_no_schema} skipped: schema not cached locally)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dev",    default="Data/Spider/dev.json")
    parser.add_argument("--out",    default="Data/cot_data/sql_dev_eval.json")
    parser.add_argument("--limit",  type=int, default=None,
                         help="Only process the first N dev entries (useful for a quick pass).")
    args = parser.parse_args()

    config = _load_config(args.config)
    build(dev_path=args.dev, out_path=args.out, config=config, limit=args.limit)


if __name__ == "__main__":
    main()
