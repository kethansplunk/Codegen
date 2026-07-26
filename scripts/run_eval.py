"""
Phase 18 — evaluation driver + SchemaRAG Table 5 ablation.

    # Full pipeline only, 100 held-out Spider dev questions
    python -m scripts.run_eval --track sql --data Data/cot_data/sql_dev_eval_full.json --n 100

    # Full Table 5 ablation (4 configurations, 3 generation passes per question)
    python -m scripts.run_eval --track sql --data Data/cot_data/sql_dev_eval_full.json \\
        --n 100 --ablation all

    # NoSQL track (needs a live mongod with the databases loaded)
    python -m scripts.run_eval --track nosql --data Data/cot_data/nosql_cot_train.json --n 100

Data file: a JSON list of entries with question / db_name / schema / key_fields
and a gold query (`sql` for the SQL track, `mql` or mql_collection+mql_pipeline
for NoSQL). scripts/build_dev_eval_set.py produces exactly this for SQL.

**Use a held-out set.** The Generator was fine-tuned on Data/cot_data/*_cot_train.json,
so evaluating there measures memorization, not generalization — Phase 15 saw
100% exact-match on train with only one unique candidate per question. Spider's
dev split uses databases the Generator has never seen.

Reports EX (the headline metric, targets: >82% SQL / >60% NoSQL) plus
exact-match, and writes the full per-question report to --out for Phase 19
error analysis.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime

import yaml

from src.eval.harness import ABLATIONS, EvalHarness, format_summary, save_report


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_hard(entry: dict, track: str) -> bool:
    """Multi-table / aggregation questions — where the components should matter most."""
    if track == "sql":
        s = entry.get("sql", "").upper()
        return (" JOIN " in s) or (s.count("SELECT") > 1) or ("GROUP BY" in s)
    pipeline = entry.get("mql", {}).get("pipeline") or entry.get("mql_pipeline") or []
    stages = {k for stage in pipeline for k in stage}
    return bool({"$lookup", "$group"} & stages) or len(pipeline) > 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--track", required=True, choices=["sql", "nosql"])
    parser.add_argument("--data", required=True,
                        help="JSON file of entries with question/db_name/schema/"
                             "key_fields + gold query. Use a HELD-OUT set.")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of questions to sample (default 100). "
                             "Use --n 0 for the whole file.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds both which questions are sampled and the "
                             "Generator's candidate sampling. Ablation deltas "
                             "compare separate generation passes, so an unseeded "
                             "run mixes each component's real effect with "
                             "sampling noise.")
    parser.add_argument("--hard", action="store_true",
                        help="Only multi-join/subquery/GROUP BY (SQL) or "
                             "$lookup/$group/3+-stage (NoSQL) questions.")
    parser.add_argument("--ablation", default="full",
                        help=f"Comma-separated subset of {ABLATIONS}, or 'all' "
                             f"for the full SchemaRAG Table 5 sweep.")
    parser.add_argument("--strategy", default="balanced",
                        choices=["balanced", "schema_priority", "example_priority"])
    parser.add_argument("--out", default=None,
                        help="Where to write the JSON report "
                             "(default evaluation/results/phase18_<track>_<ts>.json).")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    ablations = ABLATIONS if args.ablation == "all" else [
        a.strip() for a in args.ablation.split(",") if a.strip()
    ]
    for a in ablations:
        if a not in ABLATIONS:
            parser.error(f"unknown ablation {a!r}; expected a subset of {ABLATIONS} or 'all'")

    with open(args.data, encoding="utf-8") as f:
        entries = json.load(f)

    if args.hard:
        before = len(entries)
        entries = [e for e in entries if _is_hard(e, args.track)]
        print(f"--hard: filtered {before} -> {len(entries)} entries")

    if args.n and args.n < len(entries):
        random.seed(args.seed)
        entries = random.sample(entries, args.n)

    if not entries:
        parser.error("no entries left to evaluate after filtering")

    n_passes = len({(a != "no_schema_linker", a != "no_sar") for a in ablations})
    print(f"Evaluating {len(entries)} questions x {len(ablations)} ablation(s) "
          f"= {len(entries) * n_passes} generation passes "
          f"({n_passes} per question — configurations sharing generator inputs are batched).")
    if "train" in args.data:
        print("[!] --data path contains 'train'. The Generator was fine-tuned on the "
              "train CoT files, so EX there reflects memorization, not generalization.")

    config  = _load_config(args.config)
    harness = EvalHarness(track=args.track, config=config, strategy=args.strategy,
                          seed=args.seed, verbose=not args.quiet)
    try:
        report = harness.evaluate(entries, ablations=ablations)
    finally:
        harness.close()

    print(format_summary(report))

    out = args.out or (f"evaluation/results/phase18_{args.track}_"
                       f"{datetime.now():%Y%m%d_%H%M%S}.json")
    print(f"\nFull per-question report -> {save_report(report, out)}")


if __name__ == "__main__":
    main()
