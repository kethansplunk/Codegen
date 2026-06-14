import json, sqlite3, os

BASE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Spider")
db_path = os.path.join(BASE, "database")

train = json.load(open(os.path.join(BASE, "train_spider.json")))
dev   = json.load(open(os.path.join(BASE, "dev.json")))
print(f"Train: {len(train)} (expect 7000)")
print(f"Dev: {len(dev)} (expect 1034)")

dbs = [d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d))]
print(f"Databases: {len(dbs)} (expect 166)")

for db_name in dbs[:5]:
    db_file = os.path.join(db_path, db_name, f"{db_name}.sqlite")
    conn = sqlite3.connect(db_file)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    print(f"  {db_name}: {len(tables)} tables")
    conn.close()
