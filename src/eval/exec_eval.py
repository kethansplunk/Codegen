"""
Execution-based SQL evaluation (EX metric).
Adapted from SchemaRAG eval/exec_eval.py:
- result_eq: column-permutation-aware result comparison.
  Two results are equal if there exists a column permutation making them
  identical as multisets (handles SELECT a, b vs SELECT b, a).
- quick_rej: fast early rejection before full permutation search.
- order_matters: True only when query has ORDER BY.
- UTF-8 error handling added (matches our SQLite setup).
- Async execution removed — synchronous is sufficient for dev evaluation.
- replace_cur_year: replaces CURDATE() with literal 2020 for determinism.

Usage:
    from src.eval.exec_eval import evaluate_ex

    score = evaluate_ex(
        pred_sqls=["SELECT name FROM singer WHERE country='France'"],
        gold_sqls=["SELECT Name FROM singer WHERE Country = 'France'"],
        db_dir="Data/Spider/database",
        db_ids=["singer"],
    )
    print(score)  # 1.0
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from itertools import product
from typing import Any, List, Set, Tuple


# ---------------------------------------------------------------------------
# Helpers (ported directly from SchemaRAG eval/exec_eval.py)
# ---------------------------------------------------------------------------

def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    return tuple(element[i] for i in perm)


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def quick_rej(r1: List[Tuple], r2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in r1]
    s2 = [unorder_row(row) for row in r2]
    return s1 == s2 if order_matters else set(s1) == set(s2)


def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d: dict = defaultdict(int)
    for e in l1:
        d[e] += 1
    for e in l2:
        d[e] -= 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets: List[Set], result2: List[Tuple]):
    import random
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)
    for _ in range(20):
        row = random.choice(result2)
        for c1 in range(num_cols):
            for c2 in set(perm_constraints[c1]):
                if row[c2] not in tab1_sets[c1]:
                    perm_constraints[c1].remove(c2)
    return product(*perm_constraints)


def result_eq(r1: List[Tuple], r2: List[Tuple], order_matters: bool) -> bool:
    if not r1 and not r2:
        return True
    if len(r1) != len(r2) or len(r1[0]) != len(r2[0]):
        return False
    if not quick_rej(r1, r2, order_matters):
        return False

    num_cols   = len(r1[0])
    tab1_sets  = [{row[i] for row in r1} for i in range(num_cols)]
    for perm in get_constraint_permutation(tab1_sets, r2):
        if len(perm) != len(set(perm)):
            continue
        r2_perm = r2 if num_cols == 1 else [permute_tuple(row, perm) for row in r2]
        if order_matters:
            if r1 == r2_perm:
                return True
        else:
            if set(r1) == set(r2_perm) and multiset_eq(r1, r2_perm):
                return True
    return False


def replace_cur_year(query: str) -> str:
    return re.sub(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def exec_sql(db_path: str, sql: str) -> Any:
    sql = replace_cur_year(sql)
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception:
        return None


def has_order_by(sql: str) -> bool:
    return bool(re.search(r"\bORDER\s+BY\b", sql, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_ex(
    pred_sqls: List[str],
    gold_sqls: List[str],
    db_dir:    str,
    db_ids:    List[str],
) -> float:
    """
    Compute Execution Accuracy (EX) over a list of (pred, gold, db_id) triples.

    Args:
        pred_sqls: Predicted SQL queries.
        gold_sqls: Ground-truth SQL queries.
        db_dir:    Root directory of SQLite databases (Data/Spider/database).
        db_ids:    Database name for each entry.

    Returns:
        EX score in [0, 1].
    """
    import os
    assert len(pred_sqls) == len(gold_sqls) == len(db_ids)
    correct = 0

    for pred, gold, db_id in zip(pred_sqls, gold_sqls, db_ids):
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        pred_res = exec_sql(db_path, pred)
        gold_res = exec_sql(db_path, gold)

        if pred_res is None or gold_res is None:
            continue

        order = has_order_by(gold)
        if result_eq(pred_res, gold_res, order_matters=order):
            correct += 1

    return correct / len(pred_sqls) if pred_sqls else 0.0
