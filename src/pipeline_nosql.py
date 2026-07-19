"""
Phase 16 — NoSQL pipeline library.

Promotes the SchemaLinker -> SAR -> Generator -> POSG wiring validated in
Phase 15B (scripts/run_posg_nosql.py) into importable library code:

    question --> SchemaLinker (key_fields)
             --> SAR (top-3 retrieved examples, track="nosql")
             --> Generator (n candidates, track="nosql", each a JSON string
                 {"collection": ..., "pipeline": [...]})
             --> ParetoOptimalMQL (POSG selection, src/posg/posg_nosql.py)
             --> [optional] result-set comparison against gold MQL

Requires a LIVE MongoDB server, unlike the SQL track (which only needs
static .sqlite files) -- ParetoOptimalMQL.evaluate_executability() connects
to Mongo and actually runs the aggregation pipeline. See
scripts/run_posg_nosql.py's module docstring for the executability caveat
this implies for interpreting scores.

`run_one()` runs a single question through the chain given already-built
SAR/Generator instances. `run_pipeline()` is the higher-level entry point
that also builds the schema text and instantiates SchemaLinker/SAR/Generator
from config -- this is what Phase 17's LangGraph router should call.
"""

from __future__ import annotations

import json
import re


def _collection_links_set(key_fields: list) -> set:
    """
    'collection.field' strings -> {"collection", ...} lowercase. Unlike SQL's
    schema_conformity (token-level Jaccard), posg_nosql's schema_conformity
    only cares about which COLLECTIONS a pipeline touches (including $lookup
    targets), not individual field names -- so only the part before the dot
    matters here.
    """
    links = set()
    for kf in key_fields:
        collection = kf.split(".")[0].strip().lower()
        if collection:
            links.add(collection)
    return links


def _parse_candidate(text: str) -> dict:
    """Parse a Generator MQL candidate string into {'collection':..., 'pipeline':...}."""
    text = text.strip()
    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"collection": "", "pipeline": []}
        try:
            obj = json.loads(match.group(0))
        except Exception:
            return {"collection": "", "pipeline": []}
    return {"collection": obj.get("collection", ""), "pipeline": obj.get("pipeline", [])}


def _mql_result_eq(a: list, b: list) -> bool:
    """Order-insensitive result comparison, ignoring MongoDB's auto-generated _id."""
    def norm(doc):
        return tuple(sorted((k, str(v)) for k, v in doc.items() if k != "_id"))
    return sorted(norm(d) for d in a) == sorted(norm(d) for d in b)


def _execute_mql(pareto, collection: str, pipeline: list):
    try:
        return list(pareto.db[collection].aggregate(pipeline, maxTimeMS=3000))
    except Exception:
        return None


def run_one(
    question: str,
    schema: str,
    key_fields: list,
    db_name: str,
    sar,
    generator,
    mongo_uri: str,
    strategy: str,
    gold_mql: dict | None = None,
) -> dict:
    from src.posg.posg_nosql import ParetoOptimalMQL

    retrieved = sar.retrieve(question, top_k=4)
    sar_examples = [ex for ex in retrieved if ex["question"].strip() != question.strip()][:3]

    raw_candidates = generator.generate(question, schema, key_fields, sar_examples)
    if not raw_candidates:
        raise RuntimeError(
            f"Generator.generate() returned {raw_candidates!r} (expected a "
            f"non-empty list of query strings) for question={question!r} "
            f"db_name={db_name!r}. This should be structurally impossible -- "
            f"see the matching check in src/generator/infer.py's generate()."
        )
    candidates = [_parse_candidate(c) for c in raw_candidates]

    schema_links = _collection_links_set(key_fields)

    with ParetoOptimalMQL(db_name=db_name, mongo_uri=mongo_uri) as pareto:
        evaluated    = pareto.evaluate_candidates(candidates, schema_links, sar_examples)
        pareto_front = pareto.find_pareto_optimal(evaluated)
        selected     = pareto.select_final_mql(candidates, schema_links, sar_examples, strategy=strategy)

        greedy = candidates[0] if candidates else {"collection": "", "pipeline": []}

        result = {
            "question": question,
            "db_name": db_name,
            "candidates": candidates,
            "scores": [
                {"collection": c.collection, "pipeline": c.pipeline,
                 "executability": s.executability,
                 "schema_conformity": round(s.schema_conformity, 3),
                 "example_consistency": round(s.example_consistency, 3)}
                for c, s in evaluated
            ],
            "pareto_front_size": len(pareto_front),
            "n_unique_candidates": len({json.dumps(c, sort_keys=True) for c in candidates}),
            "greedy": greedy,
            "selected": selected,
            "posg_diverged": selected != greedy,
            "gold": gold_mql,
        }

        if gold_mql is not None:
            result["exact_match"]        = (selected == gold_mql)
            result["greedy_exact_match"] = (greedy == gold_mql)

            gold_result = _execute_mql(pareto, gold_mql.get("collection", ""), gold_mql.get("pipeline", []))
            # A gold query returning [] almost always means the target database
            # was never loaded into Mongo (aggregate() doesn't raise on a missing/
            # empty collection) rather than a legitimately empty answer -- an
            # empty-vs-empty match here would be a false positive, not a real EX
            # hit, so we flag it via gold_row_count and exclude it from EX below.
            result["gold_row_count"] = len(gold_result) if gold_result is not None else None
            if gold_result is not None and len(gold_result) > 0:
                sel_result = _execute_mql(pareto, selected.get("collection", ""), selected.get("pipeline", []))
                gre_result = _execute_mql(pareto, greedy.get("collection", ""), greedy.get("pipeline", []))
                result["ex"]        = 1.0 if (sel_result is not None and _mql_result_eq(sel_result, gold_result)) else 0.0
                result["greedy_ex"] = 1.0 if (gre_result is not None and _mql_result_eq(gre_result, gold_result)) else 0.0

    return result


def run_pipeline(
    question: str,
    db_name: str,
    config: dict,
    strategy: str = "balanced",
    schema: str | None = None,
    gold_mql: dict | None = None,
) -> dict:
    """
    Full SchemaLinker -> SAR -> Generator -> POSG pipeline for one question.
    Builds the schema text and instantiates every component from `config`
    (the loaded configs/config.yaml dict) if not already supplied.
    """
    from scripts.build_nosql_cot_data import format_schema_text
    from src.generator.infer import GeneratorInfer
    from src.sar.infer import get_sar_retriever
    from src.schema_linker.linker import get_schema_linker

    schema = schema or format_schema_text(db_name)

    linker     = get_schema_linker(config["schema_linker"], track="nosql")
    key_fields = linker.link(question=question, schema=schema)

    sar = get_sar_retriever(config["sar"], track="nosql")

    generator = GeneratorInfer(
        checkpoint_path=config["generator"]["nosql_checkpoint"],
        track="nosql",
        n_candidates=config["generator"]["n_candidates"],
        temperature=config["generator"]["temperature"],
    )

    mongo_uri = f"mongodb://{config['mongodb']['host']}:{config['mongodb']['port']}"
    return run_one(
        question=question,
        schema=schema,
        key_fields=key_fields,
        db_name=db_name,
        sar=sar,
        generator=generator,
        mongo_uri=mongo_uri,
        strategy=strategy,
        gold_mql=gold_mql,
    )
