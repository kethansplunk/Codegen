"""
Phase 15A — POSG driver for SQL.

Wires the standalone Phase 12-14 pieces into the flow the architecture doc
describes for Phase 15 (nothing previously called these together):

    question --> SchemaLinker (key_fields, skipped in --smoke_test)
             --> SAR (top-3 retrieved examples, src/sar/infer.py)
             --> Generator (n candidates, src/generator/infer.py)
             --> ParetoOptimal (POSG selection, src/posg/posg_sql.py)
             --> [optional] EX comparison against gold SQL, src/eval/exec_eval.py

Two modes:

  Smoke test — reuses the schema + key_fields already baked into the CoT data
  (no SchemaLinker/API calls), so it only exercises the new SAR -> Generator
  -> POSG wiring and is fast/free to run repeatedly:

      python -m scripts.run_posg_sql --smoke_test --n 10

  Single question — calls SchemaLinker per configs/config.yaml (mode: api
  needs DEEPSEEK_API_KEY, mode: model needs a trained checkpoint):

      python -m scripts.run_posg_sql --question "How many singers are there?" \\
          --db_name concert_singer

EX comparison against gold requires Data/Spider/database/<db_id>/<db_id>.sqlite
to exist locally. If it's missing, POSG selection still runs end-to-end
(executability falls back to a parse-only check) and this script reports
exact-string-match instead of EX.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import yaml


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _schema_links_set(key_fields: list) -> set:
    """
    'table.column' strings -> {"table", "column"} lowercase, matching how
    posg_sql._extract_schema_from_sql tokenizes identifiers out of the SQL.
    """
    links = set()
    for kf in key_fields:
        for part in kf.split("."):
            part = part.strip().lower()
            if part:
                links.add(part)
    return links


def _is_hard(sql: str) -> bool:
    """Rough complexity heuristic: multi-table join, subquery, or aggregation grouping."""
    s = sql.upper()
    return (" JOIN " in s) or (s.count("SELECT") > 1) or ("GROUP BY" in s)


def run_one(
    question: str,
    schema: str,
    key_fields: list,
    db_name: str,
    sar,
    generator,
    db_dir: str,
    strategy: str,
    gold_sql: str | None = None,
) -> dict:
    from src.posg.posg_sql import ParetoOptimal

    retrieved = sar.retrieve(question, top_k=4)
    sar_examples = [ex for ex in retrieved if ex["question"].strip() != question.strip()][:3]

    candidates = generator.generate(question, schema, key_fields, sar_examples)

    db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
    pareto  = ParetoOptimal(db_path=db_path if os.path.exists(db_path) else None)

    schema_links = _schema_links_set(key_fields)
    example_sqls = [ex["sql"] for ex in sar_examples if "sql" in ex]

    # Compute the same evaluation POSG uses internally so we can report *why*
    # it picked what it picked, not just the final answer — select_final_sql()
    # only returns the winning SQL string, so this mirrors its first two steps.
    evaluated    = pareto.evaluate_candidates(candidates, schema_links, example_sqls)
    pareto_front = pareto.find_pareto_optimal(evaluated)
    selected     = pareto.select_final_sql(candidates, schema_links, example_sqls, strategy=strategy)

    greedy = candidates[0] if candidates else ""

    result = {
        "question": question,
        "db_name": db_name,
        "candidates": candidates,
        "scores": [
            {"sql": c.sql, "executability": s.executability,
             "schema_conformity": round(s.schema_conformity, 3),
             "example_consistency": round(s.example_consistency, 3)}
            for c, s in evaluated
        ],
        "pareto_front_size": len(pareto_front),
        "n_unique_candidates": len(set(candidates)),
        "greedy": greedy,
        "selected": selected,
        "posg_diverged": selected.strip() != greedy.strip(),
        "gold": gold_sql,
    }

    if gold_sql is not None:
        norm = lambda s: s.strip().rstrip(";").lower()
        result["exact_match"] = norm(selected) == norm(gold_sql)
        result["greedy_exact_match"] = norm(greedy) == norm(gold_sql)
        if os.path.exists(db_path):
            from src.eval.exec_eval import evaluate_ex
            result["ex"]        = evaluate_ex([selected], [gold_sql], db_dir, [db_name])
            result["greedy_ex"] = evaluate_ex([greedy], [gold_sql], db_dir, [db_name])

    return result


def smoke_test(config: dict, n: int, strategy: str, seed: int, hard: bool, data_path: str):
    from src.generator.infer import GeneratorInfer
    from src.sar.infer import get_sar_retriever

    print(f"Data source: {data_path}")
    with open(data_path, encoding="utf-8") as f:
        cot = json.load(f)

    if hard:
        pool = [e for e in cot if _is_hard(e["sql"])]
        print(f"--hard: filtered {len(cot)} -> {len(pool)} multi-join/subquery/group-by entries")
    else:
        pool = cot

    random.seed(seed)
    sample = random.sample(pool, min(n, len(pool)))

    db_dir = config["dataset"]["db_path"]
    if not os.path.isdir(db_dir):
        print(f"[warn] {db_dir} not found — EX comparison will be skipped, "
              f"falling back to exact-string-match only.\n")

    print("Loading SAR retriever ...")
    sar = get_sar_retriever(config["sar"], track="sql")

    print("Loading Generator ...")
    generator = GeneratorInfer(
        checkpoint_path=config["generator"]["sql_checkpoint"],
        track="sql",
        n_candidates=config["generator"]["n_candidates"],
        temperature=config["generator"]["temperature"],
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
            db_dir=db_dir,
            strategy=strategy,
            gold_sql=entry["sql"],
        )
        print(f"  gold:     {r['gold']}")
        print(f"  greedy:   {r['greedy']}")
        print(f"  selected: {r['selected']}"
              + ("  [POSG DIVERGED FROM GREEDY]" if r["posg_diverged"] else "  [same as greedy]"))
        print(f"  pareto_front_size={r['pareto_front_size']}  unique_candidates={r['n_unique_candidates']}")
        print(f"  exact_match: posg={r['exact_match']} greedy={r['greedy_exact_match']}"
              + (f"   |  EX: posg={r['ex']} greedy={r['greedy_ex']}" if "ex" in r
                 else "   |  EX skipped (no local Spider DB)"))
        if r["n_unique_candidates"] > 1:
            print("  per-candidate scores:")
            seen = set()
            for j, s in enumerate(r["scores"]):
                if s["sql"] in seen:
                    continue
                seen.add(s["sql"])
                print(f"    [{j}] exec={s['executability']} schema_conformity={s['schema_conformity']}"
                      f" example_consistency={s['example_consistency']}  sql={s['sql'][:90]}")
        results.append(r)

    n_exact        = sum(r["exact_match"] for r in results)
    n_greedy_exact = sum(r["greedy_exact_match"] for r in results)
    n_diverged     = sum(r["posg_diverged"] for r in results)
    n_multi_front  = sum(r["pareto_front_size"] > 1 for r in results)
    ex_scores        = [r["ex"] for r in results if "ex" in r]
    greedy_ex_scores = [r["greedy_ex"] for r in results if "greedy_ex" in r]

    print(f"\n=== Smoke test: {len(results)} examples (hard={hard}) ===")
    print(f"exact-match  — POSG: {n_exact}/{len(results)} ({n_exact / len(results):.1%})"
          f"   greedy: {n_greedy_exact}/{len(results)} ({n_greedy_exact / len(results):.1%})")
    if ex_scores:
        print(f"EX           — POSG: {sum(ex_scores)}/{len(ex_scores)} ({sum(ex_scores) / len(ex_scores):.1%})"
              f"   greedy: {sum(greedy_ex_scores)}/{len(greedy_ex_scores)} ({sum(greedy_ex_scores) / len(greedy_ex_scores):.1%})")
    else:
        print("EX:            skipped (Data/Spider/database not found locally)")
    print(f"POSG diverged from greedy top-1 on {n_diverged}/{len(results)} examples"
          f"; Pareto front had >1 candidate on {n_multi_front}/{len(results)}")


def single_question(config: dict, question: str, db_name: str, strategy: str, schema: str | None):
    from scripts.build_cot_data import format_schema_text
    from src.generator.infer import GeneratorInfer
    from src.sar.infer import get_sar_retriever
    from src.schema_linker.linker import get_schema_linker

    schema = schema or format_schema_text(db_name)

    print("Running SchemaLinker ...")
    linker     = get_schema_linker(config["schema_linker"], track="sql")
    key_fields = linker.link(question=question, schema=schema)
    print(f"  key_fields: {key_fields}")

    print("Loading SAR retriever ...")
    sar = get_sar_retriever(config["sar"], track="sql")

    print("Loading Generator ...")
    generator = GeneratorInfer(
        checkpoint_path=config["generator"]["sql_checkpoint"],
        track="sql",
        n_candidates=config["generator"]["n_candidates"],
        temperature=config["generator"]["temperature"],
    )

    r = run_one(
        question=question,
        schema=schema,
        key_fields=key_fields,
        db_name=db_name,
        sar=sar,
        generator=generator,
        db_dir=config["dataset"]["db_path"],
        strategy=strategy,
    )
    print(f"\ncandidates:")
    for c in r["candidates"]:
        print(f"  - {c}")
    print(f"\nselected: {r['selected']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--strategy", default="balanced",
                         choices=["balanced", "schema_priority", "example_priority"])

    parser.add_argument("--smoke_test", action="store_true",
                         help="Sample --n entries from --data (schema + key_fields "
                              "already present in the file) and run them through "
                              "SAR -> Generator -> POSG.")
    parser.add_argument("--data", default="Data/cot_data/sql_cot_train.json",
                         help="JSON file of entries with question/sql/db_name/schema/"
                              "key_fields. Defaults to the train CoT data (Generator "
                              "has memorized this — use scripts/build_dev_eval_set.py "
                              "output here instead for a real generalization test).")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard", action="store_true",
                         help="Only sample questions whose gold SQL has a JOIN, "
                              "subquery, or GROUP BY — the cases POSG is more "
                              "likely to actually matter for.")

    parser.add_argument("--question")
    parser.add_argument("--db_name")
    parser.add_argument("--schema", default=None,
                         help="Raw schema text. If omitted, built from --db_name "
                              "via scripts.build_cot_data.format_schema_text.")

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
