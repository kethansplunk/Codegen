import sqlite3, json, os
import bm25s
from pymongo import MongoClient


class PromptSchemaBuilder:

    def build_sql(self, db_root: str, db_name: str) -> dict:
        db_file = os.path.join(db_root, db_name, f"{db_name}.sqlite")
        conn = sqlite3.connect(db_file)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")

        result = {}
        try:
            for (table_name,) in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                safe = table_name.replace("'", "''")
                for row in conn.execute(f"PRAGMA table_info('{safe}')").fetchall():
                    col_name = row[1]
                    values = [
                        v[0] for v in conn.execute(
                            f"SELECT DISTINCT [{col_name}] FROM [{table_name}] "
                            f"WHERE [{col_name}] IS NOT NULL LIMIT 20"
                        ).fetchall()
                    ]
                    if len(values) < 2:
                        continue
                    result[f"{table_name}.{col_name}"] = {
                        "sample_values": self._sample(values, col_name),
                        "inferred_type": self._infer_type(values),
                    }
        finally:
            conn.close()
        return result

    def build_nosql(self, db_name: str, mongo_uri: str = "mongodb://localhost:27017") -> dict:
        client = MongoClient(mongo_uri)
        mongo_db = client[db_name]
        result = {}

        try:
            for col_name in mongo_db.list_collection_names():
                docs = list(mongo_db[col_name].find({}, {"_id": 0}).limit(50))
                if not docs:
                    continue

                # Deterministic ordered field list (first-seen order across docs)
                seen_fields: set = set()
                fields: list = []
                for doc in docs:
                    for k in doc:
                        if k not in seen_fields:
                            seen_fields.add(k)
                            fields.append(k)

                for field in fields:
                    # Ordered dedup that preserves native Python types.
                    # Previously used str() conversion which caused SQL vs NoSQL
                    # type mismatch and non-deterministic set ordering.
                    seen_keys: set = set()
                    values: list = []
                    for doc in docs:
                        if field in doc and doc[field] is not None:
                            v = doc[field]
                            key = (type(v).__name__, str(v))
                            if key not in seen_keys:
                                seen_keys.add(key)
                                values.append(v)

                    if len(values) < 2:
                        continue
                    result[f"{col_name}.{field}"] = {
                        "sample_values": self._sample(values, field),
                        "inferred_type": self._infer_type(values),
                    }
        finally:
            client.close()
        return result

    def _sample(self, values: list, col_name: str, top_k: int = 3) -> list:
        if len(values) <= top_k:
            return values

        str_values = [str(v) for v in values]
        inferred = self._infer_type(values)

        # Even-spread sampling for numeric/boolean columns — BM25 is meaningless on numbers
        if inferred in ("integer", "float", "numeric", "boolean"):
            step = max(1, len(values) // top_k)
            return values[::step][:top_k]

        try:
            corpus_tokens = bm25s.tokenize(str_values)
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            query = col_name.replace("_", " ").lower()
            query_tokens = bm25s.tokenize([query])
            results, _ = retriever.retrieve(query_tokens, corpus=str_values, k=top_k)
            # Force Python str — bm25s may return numpy.str_ which json.dump rejects
            return [str(v) for v in results[0]]
        except Exception:
            step = max(1, len(values) // top_k)
            return values[::step][:top_k]

    def _infer_type(self, values: list) -> str:
        non_none = [v for v in values if v is not None]
        if not non_none:
            return "unknown"
        # bool must be checked with `any` before the int branch because
        # isinstance(True, int) is True in Python — a mixed [True, 1, 2] list
        # would otherwise fall through and be labelled "integer".
        if any(isinstance(v, bool) for v in non_none):
            return "boolean"
        if all(isinstance(v, int) for v in non_none):
            return "integer"
        if all(isinstance(v, float) for v in non_none):
            return "float"
        if all(isinstance(v, (int, float)) for v in non_none):
            return "numeric"
        str_vals = [str(v) for v in non_none]
        try:
            [int(v) for v in str_vals]
            return "integer"
        except ValueError:
            pass
        try:
            [float(v) for v in str_vals]
            return "float"
        except ValueError:
            pass
        return "string"


def build_all_sql(db_root: str, cache_dir: str):
    builder = PromptSchemaBuilder()
    sql_cache = os.path.join(cache_dir, "sql")
    os.makedirs(sql_cache, exist_ok=True)

    db_names = sorted(
        d for d in os.listdir(db_root)
        if os.path.isdir(os.path.join(db_root, d))
    )
    total = len(db_names)
    for i, db_name in enumerate(db_names, 1):
        out = os.path.join(sql_cache, f"{db_name}.json")
        if os.path.exists(out):
            print(f"[{i}/{total}] {db_name} — skipped")
            continue
        try:
            schema = builder.build_sql(db_root, db_name)
            json.dump(schema, open(out, "w"), indent=2)
            print(f"[{i}/{total}] {db_name} — {len(schema)} columns")
        except Exception as e:
            print(f"[{i}/{total}] {db_name} — ERROR: {e}")
    print(f"\nSQL done. Cached to: {sql_cache}")


def build_all_nosql(cache_dir: str, mongo_uri: str = "mongodb://localhost:27017"):
    builder = PromptSchemaBuilder()
    nosql_cache = os.path.join(cache_dir, "nosql")
    os.makedirs(nosql_cache, exist_ok=True)

    system_dbs = {"admin", "local", "config"}
    client = MongoClient(mongo_uri)
    db_names = sorted(n for n in client.list_database_names() if n not in system_dbs)
    client.close()

    total = len(db_names)
    for i, db_name in enumerate(db_names, 1):
        out = os.path.join(nosql_cache, f"{db_name}.json")
        if os.path.exists(out):
            print(f"[{i}/{total}] {db_name} — skipped")
            continue
        try:
            schema = builder.build_nosql(db_name, mongo_uri)
            json.dump(schema, open(out, "w"), indent=2)
            print(f"[{i}/{total}] {db_name} — {len(schema)} fields")
        except Exception as e:
            print(f"[{i}/{total}] {db_name} — ERROR: {e}")
    print(f"\nNoSQL done. Cached to: {nosql_cache}")


if __name__ == "__main__":
    BASE    = os.path.join(os.path.dirname(__file__), "..", "Data", "Spider")
    DB_ROOT = os.path.join(BASE, "database")
    CACHE   = os.path.join(os.path.dirname(__file__), "..", "Data", "prompt_schema")

    print("=== Building SQL PromptSchema ===")
    build_all_sql(DB_ROOT, CACHE)

    print("\n=== Building NoSQL PromptSchema ===")
    build_all_nosql(CACHE)
