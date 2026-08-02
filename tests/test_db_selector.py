"""
Tests for src/db_selector.py (question -> db_name).

Hermetic: builds a small PromptSchema directory in tmp_path rather than leaning
on Data/prompt_schema/, so the assertions stay stable as the real corpus grows.
One test does read the checked-in corpus, to pin the end-to-end wiring.
"""

from __future__ import annotations

import json
import os

import pytest

from src.db_selector import DBSelector, _db_document, _words


def _write_schema(root: str, db_name: str, columns: list[str]):
    os.makedirs(root, exist_ok=True)
    schema = {c: {"sample_values": [], "inferred_type": "text"} for c in columns}
    with open(os.path.join(root, f"{db_name}.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f)


@pytest.fixture
def schema_dir(tmp_path):
    root = str(tmp_path / "sql")
    _write_schema(root, "concert_singer", [
        "singer.Name", "singer.Country", "stadium.Location", "concert.Year"])
    _write_schema(root, "flight_2", [
        "airlines.Airline", "airports.AirportCode", "flights.SourceAirport"])
    _write_schema(root, "course_teach", [
        "teacher.Name", "course.Staring_Date", "course_arrange.Grade"])
    return root


def test_selects_by_table_name(schema_dir):
    sel = DBSelector(schema_dir=schema_dir)
    assert sel.select("How many stadiums are there?") == "concert_singer"
    assert sel.select("Which airline has the most flights?") == "flight_2"
    assert sel.select("What are the names of all the teachers?") == "course_teach"


def test_plural_question_matches_singular_table(schema_dir):
    # Without stemming "singers" never matches a table called `singer` -- this is
    # worth ~15 points of top-1 accuracy on the real dev set, so it is pinned.
    assert DBSelector(schema_dir=schema_dir).select("How many singers?") == "concert_singer"


def test_camelcase_column_is_split_into_words(schema_dir):
    # "AirportCode" must be reachable by a question saying "airport code".
    assert DBSelector(schema_dir=schema_dir).select("List every airport code") == "flight_2"


def test_rank_returns_k_sorted_descending(schema_dir):
    ranked = DBSelector(schema_dir=schema_dir).rank("How many singers?", k=3)
    assert len(ranked) == 3
    assert [n for n, _ in ranked][0] == "concert_singer"
    assert [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True)


def test_k_is_clamped_to_corpus_size(schema_dir):
    sel = DBSelector(schema_dir=schema_dir)
    assert len(sel.rank("singers", k=99)) == 3
    assert len(sel.rank("singers", k=0)) == 1


def test_unmatched_question_scores_zero(schema_dir):
    # app.py keys its "no database matched" warning off a non-positive top score,
    # so a question sharing no vocabulary with any schema must not score > 0.
    ranked = DBSelector(schema_dir=schema_dir).rank("qwerty zxcvb asdfg", k=1)
    assert ranked[0][1] == 0.0


def test_unreadable_schema_is_skipped(schema_dir):
    with open(os.path.join(schema_dir, "broken.json"), "w", encoding="utf-8") as f:
        f.write("{not valid json")
    sel = DBSelector(schema_dir=schema_dir)
    assert "broken" not in sel.db_names
    assert len(sel.db_names) == 3


def test_empty_schema_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DBSelector(schema_dir=str(tmp_path / "nope"))


def test_document_contains_name_tables_and_columns():
    # Case is preserved here; bm25s lowercases at tokenize time.
    doc = _db_document("flight_2", {"airports.AirportCode": {"sample_values": ["AHF"]}}).lower()
    assert "flight 2" in doc          # db name, underscore-split
    assert "airports" in doc          # table
    assert "airport code" in doc      # column, CamelCase-split
    assert "ahf" not in doc           # sample values deliberately excluded


def test_primary_keys_are_added_to_the_document():
    # PK terms are worth +1.3 points of top-1 (17 fixed / 3 broken over the dev
    # set, McNemar p=0.003), so their presence is pinned rather than incidental.
    schema = {"singer.Singer_ID": {}, "singer.Name": {}}
    plain = _db_document("concert_singer", schema)
    with_pk = _db_document("concert_singer", schema, ["Singer_ID"])
    assert with_pk.lower().count("singer id") == plain.lower().count("singer id") + 1


def test_fields_ranks_question_overlap(schema_dir):
    sel = DBSelector(schema_dir=schema_dir)
    fields = sel.fields("What are the names and countries of singers?", "concert_singer")
    assert "singer.Name" in fields
    assert "singer.Country" in fields
    # A column sharing no word with the question must not be reported.
    assert "concert.Year" not in fields


def test_fields_matches_camelcase_column(schema_dir):
    fields = DBSelector(schema_dir=schema_dir).fields("list every airport code", "flight_2")
    assert "airports.AirportCode" in fields


def test_fields_respects_k_and_unknown_db(schema_dir):
    sel = DBSelector(schema_dir=schema_dir)
    assert len(sel.fields("name country singer stadium concert", "concert_singer", k=2)) == 2
    assert sel.fields("anything", "no_such_db") == []


def test_words_splits_camel_and_snake():
    assert _words("AirportCode") == "Airport Code"
    assert _words("Staring_Date") == "Staring Date"
    assert _words("uid") == "uid"


def test_real_corpus_wiring():
    """Smoke test against the checked-in Data/prompt_schema/sql/ corpus."""
    sel = DBSelector(track="sql")
    assert len(sel.db_names) > 100
    assert "flight_2" in sel.db_names
    # flight_2 is genuinely ambiguous against flight_1/flight_company, so assert
    # on top-3 membership rather than an exact winner.
    ranked = [n for n, _ in sel.rank("How many flights depart from City Aberdeen?", k=3)]
    assert "flight_2" in ranked
