import sqlite3, os
from pymongo import MongoClient

client  = MongoClient()
DB_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Spider", "database"
)
mismatches = []
skipped    = []

db_names      = sorted(d for d in os.listdir(DB_ROOT) if os.path.isdir(os.path.join(DB_ROOT, d)))
mongo_db_names = set(client.list_database_names())

for db_name in db_names:
    # Skip databases that were never converted — they would produce false mismatches
    if db_name not in mongo_db_names:
        skipped.append(db_name)
        continue

    sqlite_path = os.path.join(DB_ROOT, db_name, f"{db_name}.sqlite")
    conn = sqlite3.connect(sqlite_path)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    mongo_db = client[db_name]

    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for (table,) in tables:
            sqlite_count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            mongo_count  = mongo_db[table].count_documents({})

            if sqlite_count != mongo_count:
                mismatches.append(
                    f"{db_name}.{table}: SQLite={sqlite_count}, MongoDB={mongo_count}"
                )
    finally:
        conn.close()

if skipped:
    preview = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
    print(f"Skipped ({len(skipped)} not yet converted): {preview}")

if mismatches:
    print(f"MISMATCHES FOUND ({len(mismatches)}):")
    for m in mismatches:
        print(f"  {m}")
elif not skipped:
    print(f"All {len(db_names)} databases verified — row counts match perfectly.")
else:
    converted = len(db_names) - len(skipped)
    print(f"{converted}/{len(db_names)} converted databases verified — row counts match perfectly.")
