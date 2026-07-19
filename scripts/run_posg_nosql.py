"""
Phase 15B — POSG driver for NoSQL (MQL).

CLI driver around the pipeline wiring in src/pipeline_nosql.py (Phase 16).
Mirrors scripts/run_posg_sql.py's structure and validation methodology, with
the NoSQL-specific pieces swapped in:

    question --> SchemaLinker (key_fields, skipped in --smoke_test)
             --> SAR (top-3 retrieved examples, track="nosql")
             --> Generator (n candidates, track="nosql", each a JSON string
                 {"collection": ..., "pipeline": [...]})
             --> ParetoOptimalMQL (POSG selection, src/posg/posg_nosql.py)
             --> [optional] result-set comparison against gold MQL

Requires a LIVE MongoDB server, unlike the SQL track (which only needs static
.sqlite files). ParetoOptimalMQL.evaluate_executability() connects to Mongo
and actually runs the aggregation pipeline. Without mongod running and the
target databases loaded (see src/mongodb_converter.py's MongoDBConverter),
every candidate will look "executable" in a misleading way: MongoDB does not
raise an error when you aggregate against a collection that doesn't exist or
isn't populated -- it just returns an empty result. So executability=1.0
does NOT by itself mean the target database was actually loaded; only the
gold-comparison EX score (which compares actual result sets) will reveal
that a database's data is missing.

Usage (smoke test -- reuses schema + key_fields already baked into the CoT
data, so it exercises SAR -> Generator -> POSG wiring without needing
SchemaLinker calls):

    python -m scripts.run_posg_nosql --smoke_test --n 10

Usage (single question -- calls SchemaLinker per configs/config.yaml):

    python -m scripts.run_posg_nosql --question "How many singers are there?" \\
        --db_name concert_singer
"""

from __future__ import annotations

import argparse
import json
import random

import yaml

from src.pipeline_nosql import run_one


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_hard(mql: dict) -> bool:
    """Rough complexity heuristic: $lookup (join), $group, or a multi-stage pipeline."""
    stages = mql.get("pipeline", [])
    stage_types = {list(s.keys())[0] for s in stages if s}
    return ("$lookup" in stage_types) or ("$group" in stage_types) or (len(stages) > 2)


def smoke_test(config: dict, n: int, strategy: str, seed: int, hard: bool, data_path: str):
    from src.generator.infer import GeneratorInfer
    from src.sar.infer import get_sar_retriever

    print(f"Data source: {data_path}")
    with open(data_path, encoding="utf-8") as f:
        cot = json.load(f)

    if hard:
        pool = [e for e in cot if _is_hard(e["mql"])]
        print(f"--hard: filtered {len(cot)} -> {len(pool)} $lookup/$group/multi-stage entries")
    else:
        pool = cot

    random.seed(seed)
    sample = random.sample(pool, min(n, len(pool)))

    mongo_uri = f"mongodb://{config['mongodb']['host']}:{config['mongodb']['port']}"
    print(f"Mongo URI: {mongo_uri}  (make sure mongod is running and target DBs are loaded)")

    print("Loading SAR retriever ...")
    sar = get_sar_retriever(config["sar"], track="nosql")

    print("Loading Generator ...")
    generator = GeneratorInfer(
        checkpoint_path=config["generator"]["nosql_checkpoint"],
        track="nosql",
        n_candidates=config["generator"]["n_candidates"],
        temperature=config["generator"]["temperature"],
        seed=seed,
    )

    results = []
    for i, entry in enumerate(sample, 1):
        print(f"\n[{i}/{len(sample)}] {entry['question']}")
        r = run_one(
            question=entry["question"],
            schema=entry["schema"],
            key_fields=entry.get("key_fields", []),
            db_name=entry["db_name"],
            sar=sar,
            generator=generator,
            mongo_uri=mongo_uri,
            strategy=strategy,
            gold_mql=entry["mql"],
        )
        print(f"  gold:     {json.dumps(r['gold'], ensure_ascii=False)}")
        print(f"  greedy:   {json.dumps(r['greedy'], ensure_ascii=False)}")
        print(f"  selected: {json.dumps(r['selected'], ensure_ascii=False)}"
              + ("  [POSG DIVERGED FROM GREEDY]" if r["posg_diverged"] else "  [same as greedy]"))
        print(f"  pareto_front_size={r['pareto_front_size']}  unique_candidates={r['n_unique_candidates']}")
        if "ex" in r:
            print(f"  exact_match: posg={r['exact_match']} greedy={r['greedy_exact_match']}"
                  f"   |  EX: posg={r['ex']} greedy={r['greedy_ex']}  (gold_rows={r['gold_row_count']})")
        elif r.get("gold_row_count") == 0:
            print(f"  exact_match: posg={r['exact_match']} greedy={r['greedy_exact_match']}"
                  f"   |  EX SKIPPED — gold query returned 0 rows, almost certainly means "
                  f"'{entry['db_name']}' isn't loaded into MongoDB (not a real empty answer)")
        else:
            print(f"  exact_match: posg={r['exact_match']} greedy={r['greedy_exact_match']}"
                  f"   |  EX skipped (gold pipeline itself failed to execute -- "
                  f"is '{entry['db_name']}' loaded into MongoDB?)")
        if r["n_unique_candidates"] > 1:
            print("  per-candidate scores:")
            seen = set()
            for j, s in enumerate(r["scores"]):
                key = json.dumps({"collection": s["collection"], "pipeline": s["pipeline"]}, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                print(f"    [{j}] exec={s['executability']} schema_conformity={s['schema_conformity']}"
                      f" example_consistency={s['example_consistency']}"
                      f"  mql={json.dumps({'collection': s['collection'], 'pipeline': s['pipeline']}, ensure_ascii=False)[:100]}")
        results.append(r)

    scored           = [r for r in results if "ex" in r]
    empty_gold       = [r for r in results if r.get("gold_row_count") == 0]
    exec_failed      = [r for r in results if r.get("gold_row_count") is None]
    n_exact          = sum(r["exact_match"] for r in results)
    n_greedy_exact   = sum(r["greedy_exact_match"] for r in results)
    n_diverged       = sum(r["posg_diverged"] for r in results)
    n_multi_front    = sum(r["pareto_front_size"] > 1 for r in results)
    ex_scores        = [r["ex"] for r in scored]
    greedy_ex_scores = [r["greedy_ex"] for r in scored]

    print(f"\n=== Smoke test: {len(results)} examples (hard={hard}) ===")
    print(f"exact-match  — POSG: {n_exact}/{len(results)} ({n_exact / len(results):.1%})"
          f"   greedy: {n_greedy_exact}/{len(results)} ({n_greedy_exact / len(results):.1%})")
    if ex_scores:
        print(f"EX           — POSG: {sum(ex_scores)}/{len(ex_scores)} ({sum(ex_scores) / len(ex_scores):.1%})"
              f"   greedy: {sum(greedy_ex_scores)}/{len(greedy_ex_scores)} ({sum(greedy_ex_scores) / len(greedy_ex_scores):.1%})"
              f"   (scored on {len(scored)}/{len(results)} genuinely non-empty gold results)")
    else:
        print("EX:            skipped for all examples -- no gold pipeline executed with any rows.")
    if empty_gold:
        dbs = sorted({r["db_name"] for r in empty_gold})
        print(f"[!] {len(empty_gold)}/{len(results)} examples had a gold query return 0 rows -- "
              f"excluded from EX above (likely these databases aren't loaded into MongoDB): {dbs}")
    if exec_failed:
        print(f"[!] {len(exec_failed)}/{len(results)} examples had the gold pipeline error out entirely.")
    print(f"POSG diverged from greedy top-1 on {n_diverged}/{len(results)} examples"
          f"; Pareto front had >1 candidate on {n_multi_front}/{len(results)}")


def single_question(config: dict, question: str, db_name: str, strategy: str, schema: str | None):
    from src.pipeline_nosql import run_pipeline

    r = run_pipeline(question, db_name, config, strategy=strategy, schema=schema)
    print("\ncandidates:")
    for c in r["candidates"]:
        print(f"  - {json.dumps(c, ensure_ascii=False)}")
    print(f"\nselected: {json.dumps(r['selected'], ensure_ascii=False)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--strategy", default="balanced",
                         choices=["balanced", "schema_priority", "example_priority"])

    parser.add_argument("--smoke_test", action="store_true",
                         help="Sample --n entries from --data (schema + key_fields "
                              "already present in the file) and run them through "
                              "SAR -> Generator -> POSG.")
    parser.add_argument("--data", default="Data/cot_data/nosql_cot_train.json",
                         help="JSON file of entries with question/mql/db_name/schema/"
                              "key_fields. Defaults to the train CoT data (Generator "
                              "has memorized this -- expect the same memorization "
                              "pitfall seen on the SQL track).")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard", action="store_true",
                         help="Only sample questions whose gold MQL pipeline has a "
                              "$lookup, $group, or more than 2 stages.")

    parser.add_argument("--question")
    parser.add_argument("--db_name")
    parser.add_argument("--schema", default=None,
                         help="Raw schema text. If omitted, built from --db_name "
                              "via scripts.build_nosql_cot_data.format_schema_text.")

    args = parser.parse_args()
    config = _load_config(args.config)

    if args.smoke_test:
        smoke_test(config, n=args.n, strategy=args.strategy, seed=args.seed,
                   hard=args.hard, data_path=args.data)
        return

    if not args.question or not args.db_name:
        parser.error("--question and --db_name are required unless --smoke_test is set")

    single_question(config, args.question, args.db_name, args.strategy, args.schema)


if __name__ == "__main__":
    main()
