import json
import os
from sqlglot import exp, parse_one


def parse_structural_type(sql: str) -> dict:
    try:
        ast = parse_one(sql, dialect="sqlite")

        # CTE alias names are virtual — exclude them from the real table count.
        # e.g. WITH ranked AS (SELECT id FROM t1) SELECT * FROM ranked
        # should count 1 real table (t1), not 2.
        cte_names = {cte.alias.lower() for cte in ast.find_all(exp.CTE) if cte.alias}

        joins  = list(ast.find_all(exp.Join))
        tables = {
            t.name.lower()
            for t in ast.find_all(exp.Table)
            if t.name and t.name.lower() not in cte_names
        }
        subqs  = list(ast.find_all(exp.Subquery))

        return {
            "num_joins":    len(joins),
            "num_tables":   len(tables),
            "has_group_by": ast.find(exp.Group)     is not None,
            "has_order_by": ast.find(exp.Order)     is not None,
            "has_having":   ast.find(exp.Having)    is not None,
            # Include CTEs (WITH clause) as subquery-like structures
            "has_subquery": len(subqs) > 0 or ast.find(exp.With) is not None,
            # UNION/INTERSECT/EXCEPT have fundamentally different structure from
            # plain SELECT and must not be grouped with them in SAR training pairs
            "has_set_op": (
                ast.find(exp.Union)     is not None
                or ast.find(exp.Intersect) is not None
                or ast.find(exp.Except)    is not None
            ),
        }
    except Exception:
        return None


def build_sql_rag_corpus(train_path: str, output_path: str):
    with open(train_path) as f:
        train_data = json.load(f)

    corpus      = []
    failed      = []
    type_counts = {}

    for entry in train_data:
        sql      = entry["query"]
        question = entry["question"]
        db_name  = entry["db_id"]

        struct = parse_structural_type(sql)
        if struct is None:
            failed.append(sql)
            continue

        type_key = (
            struct["num_joins"],
            struct["num_tables"],
            struct["has_group_by"],
            struct["has_order_by"],
            struct["has_having"],
            struct["has_subquery"],
            struct["has_set_op"],
        )
        type_counts[type_key] = type_counts.get(type_key, 0) + 1

        corpus.append({
            "question":        question,
            "sql":             sql,
            "db_name":         db_name,
            "structural_type": struct,
        })

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"Total:   {len(train_data)}")
    print(f"Parsed:  {len(corpus)}")
    print(f"Failed:  {len(failed)}")
    print(f"Unique structural types: {len(type_counts)}")

    print("\nTop 10 structural types (by frequency):")
    for key, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        joins, tables, grp, ord_, hav, sub, setop = key
        print(
            f"  joins={joins} tables={tables} grp={grp} ord={ord_} "
            f"hav={hav} sub={sub} setop={setop}  →  {count} queries"
        )

    if failed:
        print(f"\nFirst 3 failed SQLs:")
        for s in failed[:3]:
            print(f"  {s}")

    return corpus


if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "..")
    build_sql_rag_corpus(
        train_path  = os.path.join(BASE, "Data", "Spider", "train_spider.json"),
        output_path = os.path.join(BASE, "Data", "rag_corpus", "spider_sql_rag.json"),
    )
