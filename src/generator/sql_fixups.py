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

Scoping: a JOIN...ON's aliases are only in scope within its own top-level
SELECT branch. A naive whole-string substitution would leak a qualification
across a UNION/EXCEPT/INTERSECT boundary into a sibling SELECT that never
defined that alias -- confirmed in practice on a Phase 18 rerun, where an
EXCEPT's second branch's `ON T1.transcript_id = T2.transcript_id` caused the
first branch's unrelated, alias-free `transcript_id` to be rewritten to
`T1.transcript_id`, a column that doesn't exist there. fix_ambiguous_columns()
therefore splits on top-level set operators (respecting parenthesis depth, so
operators inside a subquery aren't treated as top-level) and fixes each branch
independently.
"""

from __future__ import annotations

import re

# Text of one JOIN...ON clause, stopping before the next clause keyword.
_ON_CLAUSE = re.compile(
    r"\bON\b(.+?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_EQUI_PAIR = re.compile(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)")
_SET_OP = re.compile(r"\b(UNION\s+ALL|UNION|EXCEPT|INTERSECT)\b", re.IGNORECASE)


def _split_top_level_branches(sql: str) -> list:
    """[(keyword_before, segment), ...] split at paren-depth-0 set operators.

    `keyword_before` is "" for the first segment. Concatenating
    keyword_before + segment for every entry reconstructs `sql` exactly, since
    each segment is the untouched slice between two match boundaries.
    """
    depth = 0
    depth_at = [0] * (len(sql) + 1)
    for i, ch in enumerate(sql):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        depth_at[i + 1] = depth

    boundaries = [m for m in _SET_OP.finditer(sql) if depth_at[m.start()] == 0]
    if not boundaries:
        return [("", sql)]

    branches, prev_end, keyword = [], 0, ""
    for m in boundaries:
        branches.append((keyword, sql[prev_end:m.start()]))
        keyword, prev_end = m.group(1), m.end()
    branches.append((keyword, sql[prev_end:]))
    return branches


def _fix_branch(sql: str) -> str:
    """Ambiguous-column fix for a single SELECT branch (no set operators)."""
    if "JOIN" not in sql.upper():
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


def fix_ambiguous_columns(sql: str) -> str:
    """Qualify columns that are ambiguous under a JOIN but safe to resolve."""
    if not sql or "JOIN" not in sql.upper():
        return sql

    branches = _split_top_level_branches(sql)
    if len(branches) == 1:
        return _fix_branch(sql)
    return "".join(keyword + _fix_branch(segment) for keyword, segment in branches)
