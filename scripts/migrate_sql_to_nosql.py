"""
Phase 20C -- SQL-to-NoSQL migration utility.

Migrates one Spider SQL database (schema + data) to MongoDB, reusing
MongoDBConverter (Phase 5B) directly -- the same converter build_all() uses
for all 166 Spider databases in a loop, just pointed at a single db_name.
Requires the target db_name to already have an FK graph
(Data/fk_graphs/<db>.json, Phase 5A) and NoSQL PromptSchema
(Data/prompt_schema/nosql/<db>.json, Phase 6); both already exist for all
166 Spider databases, so migrating any of them needs no extra setup.

Once migrated, the database is immediately queryable through the existing
NoSQL pipeline (src/router/langgraph_router.py's Router, app.py, api.py) --
none of those care how the data got into Mongo. --verify runs one real
question through the Router end-to-end (Phase 14B's trained NoSQL
Generator), confirming the migration produced a genuinely working database
rather than just rows sitting in Mongo unconnected to anything.

    python -m scripts.migrate_sql_to_nosql --db_name concert_singer
    python -m scripts.migrate_sql_to_nosql --db_name concert_singer \\
        --verify --question "How many singers are there?"
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

from src.mongodb_converter import MongoDBConverter


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def migrate(db_name: str, config: dict, converter: MongoDBConverter | None = None) -> dict:
    """
    Convert db_name to MongoDB and write the same schema-cache file
    convert_all() writes, so a later full Phase 5B re-run treats this
    database as already done instead of reconverting it.

    `converter` is injectable so tests can pass a stub instead of needing a
    real (or mocked) MongoDB connection -- same rationale as Router's
    injectable linker/sar/generator in src/router/langgraph_router.py.
    """
    db_root = config["dataset"]["db_path"]
    db_file = os.path.join(db_root, db_name, f"{db_name}.sqlite")
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"no local SQLite database at {db_file}")

    fk_graph_dir = config["fk_graph"]["cache_path"]
    if converter is None:
        mongo_uri = f"mongodb://{config['mongodb']['host']}:{config['mongodb']['port']}"
        converter = MongoDBConverter(mongo_uri)

    schema = converter.convert_database(db_root, db_name, fk_graph_dir)

    schema_cache_dir = config["mongodb"]["schema_cache"]
    os.makedirs(schema_cache_dir, exist_ok=True)
    with open(os.path.join(schema_cache_dir, f"{db_name}_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    return schema


def format_summary(db_name: str, schema: dict) -> str:
    lines = [f"=== Migrated {db_name} -> MongoDB ==="]
    for name, coll in schema["collections"].items():
        pk = ", ".join(coll["primary_key"]) or "none"
        lines.append(f"  {name:<30} {coll['row_count']:>6} docs   pk=({pk})")
    total = sum(c["row_count"] for c in schema["collections"].values())
    lines.append(f"  {'total':<30} {total:>6} docs across {len(schema['collections'])} collections")
    return "\n".join(lines)


def verify(db_name: str, config: dict, question: str) -> dict:
    """One real question through the NoSQL Router. Needs real checkpoints,
    DEEPSEEK_API_KEY, and a live mongod -- same requirements as app.py/api.py."""
    from src.router.langgraph_router import Router

    router = Router(track="nosql", config=config, verbose=True)
    try:
        return router.run(question, db_name)
    finally:
        router.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--db_name", required=True,
                         help="Spider SQL database to migrate. Needs an existing "
                              "FK graph (Data/fk_graphs/) and NoSQL PromptSchema "
                              "(Data/prompt_schema/nosql/) -- both already exist "
                              "for all 166 Spider databases.")
    parser.add_argument("--verify", action="store_true",
                         help="Run one smoke-test question through the NoSQL "
                              "Router after migrating (needs real checkpoints, "
                              "DEEPSEEK_API_KEY, and a live mongod).")
    parser.add_argument("--question", default=None,
                         help="Question to verify with. Required with --verify.")
    args = parser.parse_args()

    if args.verify and not args.question:
        parser.error("--question is required with --verify")

    config = _load_config(args.config)
    schema = migrate(args.db_name, config)
    print(format_summary(args.db_name, schema))

    if args.verify:
        print(f"\n=== Verifying via NoSQL Router: {args.question!r} ===")
        result = verify(args.db_name, config, args.question)
        print(f"status: {result['status']}")
        print(f"query:  {json.dumps(result['query'], default=str)}")
        if result["rows"] is not None:
            print(f"rows:   {len(result['rows'])}")


if __name__ == "__main__":
    main()
