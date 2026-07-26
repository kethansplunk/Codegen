"""
Phase 18C — CP1 baseline few-shot prompt construction.

Pure-function tests only -- no model/tokenizer loaded, matching the project
convention of running on Mac with no GPU/checkpoints.

    pytest tests/test_run_baseline.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_baseline import (  # noqa: E402
    _format_block,
    _pick_fewshot_examples,
    build_prompt,
    extract_sql,
)

SCHEMA = "# Table: singer\n[\n(singer_id:INT, Primary Key),\n(name:TEXT),\n]"


def test_build_prompt_zero_shot_matches_target_only():
    prompt = build_prompt(SCHEMA, "How many singers?")
    assert prompt == _format_block(SCHEMA, "How many singers?")
    assert prompt.endswith("SELECT ")


def test_build_prompt_prepends_fewshot_before_target():
    fewshot = [{"schema": SCHEMA, "question": "Q1?", "sql": "SELECT count(*) FROM singer"}]
    prompt = build_prompt(SCHEMA, "How many singers?", fewshot)
    assert prompt.index("Q1?") < prompt.index("How many singers?")
    assert prompt.endswith("SELECT ")


def test_format_block_strips_leading_select_and_adds_semicolon():
    block = _format_block(SCHEMA, "Q1?", "SELECT count(*) FROM singer")
    assert "SELECT count(*) FROM singer;" in block
    # the answer's own "SELECT" is stripped, so it isn't doubled with the
    # prompt's own open "SELECT " -- exactly one occurrence, not "SELECT SELECT"
    assert block.count("SELECT") == 1


def test_format_block_without_sql_ends_open_for_completion():
    block = _format_block(SCHEMA, "Q1?")
    assert block.endswith("SELECT ")


def test_pick_fewshot_examples_picks_diverse_shapes(tmp_path):
    train = [
        {"db_name": "a", "sql": "SELECT count(*) FROM t", "schema": "s", "question": "q1"},
        {"db_name": "b", "sql": "SELECT x FROM t JOIN u ON t.id = u.id", "schema": "s", "question": "q2"},
        {"db_name": "c", "sql": "SELECT x FROM t ORDER BY x", "schema": "s", "question": "q3"},
    ]
    path = tmp_path / "train.json"
    path.write_text(json.dumps(train))

    picked = _pick_fewshot_examples(str(path), k=3)
    assert len(picked) == 3
    dbs = {e["db_name"] for e in picked}
    assert dbs == {"a", "b", "c"}


def test_pick_fewshot_examples_zero_k_returns_empty(tmp_path):
    path = tmp_path / "train.json"
    path.write_text("[]")
    assert _pick_fewshot_examples(str(path), k=0) == []


def test_pick_fewshot_examples_never_repeats_a_db(tmp_path):
    # Two candidates for the same shape share a db_name -- only the first should be used,
    # so a later shape can't accidentally reuse it.
    train = [
        {"db_name": "a", "sql": "SELECT count(*) FROM t", "schema": "s", "question": "q1"},
        {"db_name": "a", "sql": "SELECT x FROM t JOIN u ON t.id = u.id", "schema": "s", "question": "q2"},
    ]
    path = tmp_path / "train.json"
    path.write_text(json.dumps(train))
    picked = _pick_fewshot_examples(str(path), k=2)
    assert len(picked) == 1


def test_extract_sql_still_works_unchanged():
    assert extract_sql("count(*) FROM singer;\n\nmore prose") == "SELECT count(*) FROM singer"
