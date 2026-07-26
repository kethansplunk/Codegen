"""
Phase 18 error-analysis fix — ambiguous-column qualifier.

    pytest tests/test_sql_fixups.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator.sql_fixups import fix_ambiguous_columns   # noqa: E402


def test_qualifies_ambiguous_join_key():
    # The real Phase 18 failure: bare `owner_id` is ambiguous under the join,
    # both in SELECT and GROUP BY.
    sql = ("SELECT owner_id, first_name, last_name FROM owners JOIN dogs "
           "ON owners.owner_id = dogs.owner_id GROUP BY owner_id "
           "ORDER BY count(*) DESC LIMIT 1")
    fixed = fix_ambiguous_columns(sql)
    assert "SELECT owners.owner_id, first_name, last_name" in fixed
    assert "GROUP BY owners.owner_id" in fixed
    # Non-ambiguous columns are left untouched.
    assert "first_name" in fixed and "owners.first_name" not in fixed


def test_no_join_is_a_no_op():
    sql = "SELECT name FROM singer WHERE age > 30"
    assert fix_ambiguous_columns(sql) == sql


def test_join_with_no_ambiguous_columns_is_unchanged():
    sql = "SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id"
    assert fix_ambiguous_columns(sql) == sql


def test_idempotent_on_already_qualified_columns():
    sql = ("SELECT owners.owner_id FROM owners JOIN dogs "
           "ON owners.owner_id = dogs.owner_id")
    once  = fix_ambiguous_columns(sql)
    twice = fix_ambiguous_columns(once)
    assert once == twice
    assert "owners.owners.owner_id" not in once


def test_empty_and_none_input():
    assert fix_ambiguous_columns("") == ""
    assert fix_ambiguous_columns(None) is None


def test_does_not_leak_alias_across_except_boundary():
    # Regression: a Phase 18 rerun showed the naive whole-string version
    # rewriting the EXCEPT's *first* branch's bare `transcript_id` (no `T1`
    # in scope there) using the alias from the *second* branch's JOIN...ON.
    sql = ("SELECT transcript_date ,  transcript_id FROM TRANSCRIPTS "
           "EXCEPT SELECT T1.transcript_date ,  T1.transcript_id FROM TRANSCRIPTS AS T1 "
           "JOIN transcript_contents AS T2 ON T1.transcript_id  =  T2.transcript_id "
           "GROUP BY T1.transcript_id ORDER BY count(*) DESC LIMIT 1")
    fixed = fix_ambiguous_columns(sql)
    first_branch = fixed.split("EXCEPT")[0]
    assert "T1." not in first_branch
    assert "GROUP BY T1.transcript_id" in fixed   # second branch still qualified


def test_split_respects_parens_around_subquery():
    # A UNION/EXCEPT keyword inside a parenthesized subquery is not a
    # top-level branch boundary and must not be split on.
    sql = ("SELECT owner_id FROM owners JOIN dogs ON owners.owner_id = dogs.owner_id "
           "WHERE owner_id IN (SELECT x FROM t1 EXCEPT SELECT y FROM t2)")
    fixed = fix_ambiguous_columns(sql)
    assert "(SELECT x FROM t1 EXCEPT SELECT y FROM t2)" in fixed
    assert "SELECT owners.owner_id FROM owners" in fixed
