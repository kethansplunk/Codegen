"""
Phase 16 — SQL pipeline library.

Promotes the SchemaLinker -> SAR -> Generator -> POSG wiring validated in
Phase 15A (scripts/run_posg_sql.py) into importable library code:

    question --> SchemaLinker (key_fields)
             --> SAR (top-3 retrieved examples, src/sar/infer.py)
             --> Generator (n candidates, src/generator/infer.py)
             --> ParetoOptimal (POSG selection, src/posg/posg_sql.py)
             --> [optional] EX comparison against gold SQL, src/eval/exec_eval.py

`run_one()` runs a single question through the chain given already-built
SAR/Generator instances (used by scripts/run_posg_sql.py's batch smoke test,
which reuses one instance across many questions). `run_pipeline()` is the
higher-level entry point that also builds the schema text and instantiates
SchemaLinker/SAR/Generator from config -- this is what Phase 17's LangGraph
router should call.
"""

from __future__ import annotations

import os


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


def run_pipeline(
    question: str,
    db_name: str,
    config: dict,
    strategy: str = "balanced",
    schema: str | None = None,
    gold_sql: str | None = None,
) -> dict:
    """
    Full SchemaLinker -> SAR -> Generator -> POSG pipeline for one question.
    Builds the schema text and instantiates every component from `config`
    (the loaded configs/config.yaml dict) if not already supplied.
    """
    from scripts.build_cot_data import format_schema_text
    from src.generator.infer import GeneratorInfer
    from src.sar.infer import get_sar_retriever
    from src.schema_linker.linker import get_schema_linker

    schema = schema or format_schema_text(db_name)

    linker     = get_schema_linker(config["schema_linker"], track="sql")
    key_fields = linker.link(question=question, schema=schema)

    sar = get_sar_retriever(config["sar"], track="sql")

    generator = GeneratorInfer(
        checkpoint_path=config["generator"]["sql_checkpoint"],
        track="sql",
        n_candidates=config["generator"]["n_candidates"],
        temperature=config["generator"]["temperature"],
    )

    return run_one(
        question=question,
        schema=schema,
        key_fields=key_fields,
        db_name=db_name,
        sar=sar,
        generator=generator,
        db_dir=config["dataset"]["db_path"],
        strategy=strategy,
        gold_sql=gold_sql,
    )
