"""
Deterministic post-generation fixups for predicted SQL.

The Generator occasionally emits SQL that fails to execute because a
SELECT/GROUP BY/ORDER BY column is ambiguous under a JOIN -- e.g.

    SELECT owner_id FROM owners JOIN dogs ON owners.owner_id = dogs.owner_id

raises sqlite3's "ambiguous column name: owner_id" even though both sides are
guaranteed equal by the join condition itself. Found in the Phase 18 SQL
ablation error analysis (evaluation/results/phase18_sql_full.json).

fix_ambiguous_columns() is a narrow, mechanical rewrite: for every `a.col =
b.col` equality inside a JOIN...ON clause, a later *bare* (unqualified)
mention of `col` is safe to qualify with `a.col` -- the ON condition already
guarantees the two sides are equal for every row in the result, for both INNER
and LEFT JOIN. Using the left/first-named side of the condition keeps LEFT
JOIN's null-extension semantics intact (the right side can be NULL on an
unmatched row where the left side can't).

This deliberately does NOT touch the broader "missing DISTINCT after a
one-to-many join" or "ties collapsed by ORDER BY ... LIMIT 1" failure patterns
from the same error analysis -- those require judging the query's semantic
intent, and a blanket rule risks changing results for currently-correct
queries. Left as a documented limitation for Phase 19 instead.
"""

from __future__ import annotations

import re

# Text of one JOIN...ON clause, stopping before the next clause keyword.
_ON_CLAUSE = re.compile(
    r"\bON\b(.+?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_EQUI_PAIR = re.compile(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)")


def fix_ambiguous_columns(sql: str) -> str:
    """Qualify columns that are ambiguous under a JOIN but safe to resolve."""
    if not sql or "JOIN" not in sql.upper():
        return sql

    ambiguous: dict[str, str] = {}
    for on_match in _ON_CLAUSE.finditer(sql):
        for alias1, col1, alias2, col2 in _EQUI_PAIR.findall(on_match.group(1)):
            if col1.lower() == col2.lower():
                ambiguous.setdefault(col1.lower(), f"{alias1}.{col1}")

    if not ambiguous:
        return sql

    fixed = sql
    for col, qualified in ambiguous.items():
        # Negative lookbehind for '.' skips references already qualified,
        # including the ON clause itself (always alias.col there already) --
        # so this is idempotent and never double-qualifies.
        fixed = re.sub(rf"(?<!\.)\b{re.escape(col)}\b", qualified, fixed, flags=re.IGNORECASE)
    return fixed
