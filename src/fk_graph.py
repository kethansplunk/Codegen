import sqlite3, json, os
import networkx as nx


class FKGraphBuilder:

    def build(self, db_root: str, db_name: str) -> dict:
        db_file = os.path.join(db_root, db_name, f"{db_name}.sqlite")
        conn = sqlite3.connect(db_file)
        try:
            tables = self._get_tables(conn)

            # MultiDiGraph allows multiple FK edges between the same pair of tables
            # (e.g. match.home_team_id and match.away_team_id both → team)
            G = nx.MultiDiGraph()
            for t in tables:
                G.add_node(t["name"], columns=t["columns"], pk=t["pk"])

            for t in tables:
                for fk in self._get_foreign_keys(conn, t["name"]):
                    G.add_edge(
                        t["name"], fk["parent_table"],
                        child_col=fk["child_col"],
                        parent_col=fk["parent_col"],
                    )

            centrality = nx.in_degree_centrality(G)
        finally:
            conn.close()

        return {
            "db_name": db_name,
            "nodes": [{"name": n, **G.nodes[n]} for n in G.nodes],
            "edges": [{"from": u, "to": v, **data} for u, v, data in G.edges(data=True)],
            "centrality": centrality,
        }

    def _get_tables(self, conn):
        tables = []
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            safe = name.replace("'", "''")
            cols, pk = [], []
            for row in conn.execute(f"PRAGMA table_info('{safe}')").fetchall():
                _, col_name, dtype, _, _, is_pk = row
                cols.append({"name": col_name, "type": dtype})
                if is_pk:
                    pk.append(col_name)
            tables.append({"name": name, "columns": cols, "pk": pk})
        return tables

    def _get_foreign_keys(self, conn, table_name):
        safe = table_name.replace("'", "''")
        fks = []
        for row in conn.execute(
            f"PRAGMA foreign_key_list('{safe}')"
        ).fetchall():
            fks.append({
                "child_col": row[3],
                "parent_table": row[2],
                "parent_col": row[4],
            })
        return fks


def build_all(db_root: str, cache_dir: str):
    builder = FKGraphBuilder()
    db_names = sorted(
        d for d in os.listdir(db_root)
        if os.path.isdir(os.path.join(db_root, d))
    )
    os.makedirs(cache_dir, exist_ok=True)
    total = len(db_names)

    for i, db_name in enumerate(db_names, 1):
        out_path = os.path.join(cache_dir, f"{db_name}.json")
        if os.path.exists(out_path):
            print(f"[{i}/{total}] {db_name} — skipped (cached)")
            continue
        try:
            graph = builder.build(db_root, db_name)
            with open(out_path, "w") as f:
                json.dump(graph, f, indent=2)
            print(f"[{i}/{total}] {db_name} — {len(graph['nodes'])} tables, {len(graph['edges'])} FK edges")
        except Exception as e:
            print(f"[{i}/{total}] {db_name} — ERROR: {e}")

    print(f"\nDone. Graphs cached to: {cache_dir}")


if __name__ == "__main__":
    BASE     = os.path.join(os.path.dirname(__file__), "..", "Data", "Spider")
    DB_ROOT  = os.path.join(BASE, "database")
    CACHE    = os.path.join(os.path.dirname(__file__), "..", "Data", "fk_graphs")
    build_all(DB_ROOT, CACHE)
