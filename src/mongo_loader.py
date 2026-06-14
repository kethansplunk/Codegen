import sqlite3
import os
from pymongo import MongoClient
from src.fk_graph import FKGraphBuilder


def load_all_databases(db_path, mongo_url="mongodb://localhost:27017"):
    """Convert all 166 Spider SQLite databases into MongoDB collections.

    Each SQLite database → one MongoDB database.
    Each table → one MongoDB collection.
    Each row → one MongoDB document.
    """
    client = MongoClient(mongo_url)

    # Loop over all database folders in Spider dataset
    for db_name in os.listdir(db_path):
        sqlite_file = f"{db_path}/{db_name}/{db_name}.sqlite"

        # Skip non-database folders (no .sqlite file inside)
        if not os.path.exists(sqlite_file):
            continue

        # Connect to SQLite and extract table schema
        conn = sqlite3.connect(sqlite_file)
        conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
        tables = FKGraphBuilder()._get_tables(conn)

        # For each table, read all rows and insert into MongoDB
        for table in tables:
            cols = table['columns']
            rows = conn.execute(f"SELECT * FROM {table['name']}").fetchall()

            # Convert each row tuple to a dict using column names
            docs = [dict(zip(cols, row)) for row in rows]

            if docs:
                # client[db_name] = MongoDB database, [table['name']] = collection
                client[db_name][table['name']].insert_many(docs)

        conn.close()
        print(f"✅ Loaded: {db_name} ({len(tables)} collections)")

    print("\n🎉 Phase 5B complete — all Spider databases loaded into MongoDB.")


if __name__ == "__main__":
    load_all_databases("data/spider/database")