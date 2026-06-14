import json, sqlite3, os
# Check train + dev counts
train = json.load(open('data/spider/train_spider.json'))
dev = json.load(open('data/spider/dev.json'))
print(f"Train: {len(train)} (expect 7000)")
print(f"Dev: {len(dev)} (expect 1034)")
# Check databases
db_path = 'data/spider/database'
dbs = [d for d in os.listdir(db_path) if os.path.isdir(f"{db_path}/{d}")]
print(f"Databases: {len(dbs)} (expect 166)")
# Spot-check 5 databases
for db_name in dbs[:5]:
    db_file = f"{db_path}/{db_name}/{db_name}.sqlite"
    conn = sqlite3.connect(db_file)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f" {db_name}: {len(tables)} tables")
    conn.close()