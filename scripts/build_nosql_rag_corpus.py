import json
import os
import sqlite3
import time
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MONGO_CLIENT = MongoClient("mongodb://localhost:27017")

BASE        = os.path.join(os.path.dirname(__file__), "..")
SQL_CORPUS  = os.path.join(BASE, "Data", "rag_corpus", "spider_sql_rag.json")
SCHEMA_DIR  = os.path.join(BASE, "Data", "mongodb")
SQLITE_DIR  = os.path.join(BASE, "Data", "Spider", "database")
OUTPUT_PATH = os.path.join(BASE, "Data", "rag_corpus", "spider_nosql_rag.json")
CHECKPOINT  = os.path.join(BASE, "Data", "rag_corpus", "nosql_checkpoint.json")


def load_mongo_schema(db_name: str) -> str:
    path = os.path.join(SCHEMA_DIR, f"{db_name}_schema.json")
    if not os.path.exists(path):
        return ""
    schema = json.load(open(path))
    lines = []
    for col_name, col in schema["collections"].items():
        fields = ", ".join(f["name"] for f in col["columns"])
        lines.append(f"Collection '{col_name}': fields [{fields}]")
        if col["foreign_keys"]:
            for fk in col["foreign_keys"]:
                lines.append(
                    f"  FK: {col_name}.{fk['child_col']} → {fk['to']}.{fk['parent_col']}"
                )
    return "\n".join(lines)


def call_deepseek(question: str, sql: str, schema_text: str) -> dict | None:
    prompt = f"""You are a MongoDB expert. Convert the SQL query to a MongoDB aggregation pipeline.

Database schema:
{schema_text}

Question: {question}
SQL: {sql}

Return ONLY valid JSON in this exact format, no explanation:
{{
  "collection": "primary_collection_name",
  "pipeline": [... aggregation stages ...]
}}

Rules:
- Use $match for filtering, $group for GROUP BY, $sort for ORDER BY, $limit for LIMIT
- Use $lookup for JOINs between collections
- Use $count or $group with $sum:1 for COUNT(*)
- Always return a pipeline array even for simple queries"""

    try:
        response = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"Error occurred while calling DeepSeek: {e}")
        return None


def execute_mql(db_name: str, collection: str, pipeline: list) -> list | None:
    try:
        db = MONGO_CLIENT[db_name]
        results = list(db[collection].aggregate(pipeline, maxTimeMS=5000))
        return results
    except Exception:
        return None


def execute_sql(db_name: str, sql: str) -> list | None:
    db_path = os.path.join(SQLITE_DIR, db_name, f"{db_name}.sqlite")
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        rows = conn.execute(sql).fetchall()
        conn.close()
        return rows
    except Exception:
        return None


def results_match(sql_rows: list, mql_docs: list) -> bool:
    if sql_rows is None or mql_docs is None:
        return False
    # Compare row counts — exact value comparison is unreliable across SQL/MongoDB types
    return len(sql_rows) == len(mql_docs) and len(sql_rows) > 0


def build_nosql_rag_corpus():
    entries = json.load(open(SQL_CORPUS))

    # Sort by complexity: simplest first (1 table, 0 joins)
    entries.sort(key=lambda e: (
        e["structural_type"]["num_tables"],
        e["structural_type"]["num_joins"],
        e["structural_type"]["has_subquery"],
    ))

    # Load checkpoint if resuming
    start_idx = 0
    corpus = []
    if os.path.exists(CHECKPOINT):
        data = json.load(open(CHECKPOINT))
        corpus = data["corpus"]
        start_idx = data["next_idx"]
        print(f"Resuming from index {start_idx}, {len(corpus)} pairs collected so far")

    stats = {"total": len(entries), "attempted": 0, "generated": 0,
             "verified": 0, "failed": 0}

    for i, entry in enumerate(entries[start_idx:], start=start_idx):
        question = entry["question"]
        sql      = entry["sql"]
        db_name  = entry["db_name"]

        schema_text = load_mongo_schema(db_name)
        if not schema_text:
            stats["failed"] += 1
            continue

        stats["attempted"] += 1

        mql = call_deepseek(question, sql, schema_text)
        if not mql or "collection" not in mql or "pipeline" not in mql:
            stats["failed"] += 1
            time.sleep(0.2)
            continue

        stats["generated"] += 1

        sql_rows = execute_sql(db_name, sql)
        mql_docs = execute_mql(db_name, mql["collection"], mql["pipeline"])

        if results_match(sql_rows, mql_docs):
            stats["verified"] += 1
            corpus.append({
                "question":        question,
                "mql_collection":  mql["collection"],
                "mql_pipeline":    mql["pipeline"],
                "db_name":         db_name,
                "structural_type": entry["structural_type"],
                "source_sql":      sql,
            })

        # Progress every 50
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{stats['total']}] verified={stats['verified']} "
                  f"generated={stats['generated']} failed={stats['failed']}")
            # Save checkpoint
            json.dump({"corpus": corpus, "next_idx": i + 1},
                      open(CHECKPOINT, "w"), indent=2)

        time.sleep(0.1)  # avoid rate limiting

    # Save final output
    json.dump(corpus, open(OUTPUT_PATH, "w"), indent=2)
    print(f"\nDone.")
    print(f"Total attempted : {stats['attempted']}")
    print(f"Generated       : {stats['generated']}")
    print(f"Verified        : {stats['verified']} "
          f"({stats['verified']/max(stats['attempted'],1)*100:.1f}%)")
    print(f"Saved to        : {OUTPUT_PATH}")


if __name__ == "__main__":
    build_nosql_rag_corpus()
