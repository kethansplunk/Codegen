import sqlite3, json, os
from pymongo import MongoClient


class MongoDBConverter:

    def __init__(self, mongo_uri="mongodb://localhost:27017"):
        self.client = MongoClient(mongo_uri)

    def convert_database(self, db_root: str, db_name: str, fk_graph_dir: str) -> dict:
        db_file = os.path.join(db_root, db_name, f"{db_name}.sqlite")
        conn = sqlite3.connect(db_file)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        conn.row_factory = sqlite3.Row

        fk_path = os.path.join(fk_graph_dir, f"{db_name}.json")
        if os.path.exists(fk_path):
            with open(fk_path) as f:
                fk_graph = json.load(f)
        else:
            fk_graph = {"nodes": [], "edges": []}

        mongo_db = self.client[db_name]
        schema = {"db_name": db_name, "collections": {}}

        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()

            for (table_name,) in tables:
                safe = table_name.replace("'", "''")
                col_info = conn.execute(f"PRAGMA table_info('{safe}')").fetchall()
                pk_cols  = [r[1] for r in col_info if r[5] > 0]
                all_cols = [{"name": r[1], "type": r[2]} for r in col_info]
                col_types = {r[1]: r[2] for r in col_info}  # col_name → declared SQLite type

                rows = conn.execute(f"SELECT * FROM [{table_name}]").fetchall()
                docs = [self._row_to_doc(dict(row), col_types) for row in rows]

                fk_edges = [e for e in fk_graph.get("edges", []) if e["from"] == table_name]

                # Always drop and recreate so that empty tables also exist as
                # MongoDB collections — without this, build_nosql's
                # list_collection_names() silently omits 0-row tables, creating
                # an asymmetric schema between the SQL and NoSQL tracks.
                mongo_db[table_name].drop()
                if docs:
                    mongo_db[table_name].insert_many(docs)
                else:
                    mongo_db.create_collection(table_name)

                schema["collections"][table_name] = {
                    "primary_key": pk_cols,
                    "columns": all_cols,
                    "foreign_keys": fk_edges,
                    "row_count": len(docs),
                }
        finally:
            conn.close()

        return schema

    def _row_to_doc(self, row: dict, col_types: dict) -> dict:
        return {k: self._coerce(v, col_types.get(k, "TEXT")) for k, v in row.items()}

    def _coerce(self, value, col_type: str = "TEXT"):
        if value is None:
            return None
        if isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            # Only coerce when the declared SQLite column type is explicitly numeric.
            # TEXT columns (IDs, zip codes, phone numbers, barcodes) must stay as
            # strings to preserve leading zeros and non-numeric formatting.
            base_type = col_type.upper().split("(")[0].strip()
            if base_type in (
                "INTEGER", "INT", "TINYINT", "SMALLINT", "MEDIUMINT",
                "BIGINT", "UNSIGNED BIG INT", "INT2", "INT8",
            ):
                try:
                    return int(value)
                except ValueError:
                    return value
            if base_type in (
                "REAL", "DOUBLE", "DOUBLE PRECISION", "FLOAT",
                "NUMERIC", "DECIMAL",
            ):
                try:
                    return float(value)
                except ValueError:
                    return value
        return value


def convert_all(db_root: str, fk_graph_dir: str, schema_cache_dir: str):
    converter = MongoDBConverter()
    db_names = sorted(
        d for d in os.listdir(db_root)
        if os.path.isdir(os.path.join(db_root, d))
    )
    os.makedirs(schema_cache_dir, exist_ok=True)
    total = len(db_names)

    for i, db_name in enumerate(db_names, 1):
        schema_path = os.path.join(schema_cache_dir, f"{db_name}_schema.json")
        if os.path.exists(schema_path):
            print(f"[{i}/{total}] {db_name} — skipped (cached)")
            continue
        try:
            schema = converter.convert_database(db_root, db_name, fk_graph_dir)
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2)
            n_cols = sum(len(c["columns"]) for c in schema["collections"].values())
            print(f"[{i}/{total}] {db_name} — {len(schema['collections'])} collections, {n_cols} fields")
        except Exception as e:
            print(f"[{i}/{total}] {db_name} — ERROR: {e}")

    print(f"\nDone. Schemas cached to: {schema_cache_dir}")


if __name__ == "__main__":
    BASE      = os.path.join(os.path.dirname(__file__), "..", "Data", "Spider")
    DB_ROOT   = os.path.join(BASE, "database")
    FK_GRAPHS = os.path.join(os.path.dirname(__file__), "..", "Data", "fk_graphs")
    CACHE     = os.path.join(os.path.dirname(__file__), "..", "Data", "mongodb")
    convert_all(DB_ROOT, FK_GRAPHS, CACHE)
