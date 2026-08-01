"""
Phase 20C — SQL-to-NoSQL migration utility tests.

No real/mocked MongoDB needed: migrate() takes an injectable `converter`
(same rationale as Router's injectable linker/sar/generator), so these tests
verify the orchestration logic -- checkpoint-file writing, error handling,
summary formatting -- against a stub that returns a canned schema dict,
exactly what MongoDBConverter.convert_database() would return.

    pytest tests/test_migrate_sql_to_nosql.py -v
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.migrate_sql_to_nosql import format_summary, migrate  # noqa: E402

DB_NAME = "concert_singer"

FAKE_SCHEMA = {
    "db_name": DB_NAME,
    "collections": {
        "singer": {"primary_key": ["singer_id"], "columns": [], "foreign_keys": [], "row_count": 2},
        "concert": {"primary_key": ["concert_id"], "columns": [], "foreign_keys": [], "row_count": 0},
    },
}


class StubConverter:
    def __init__(self, schema=FAKE_SCHEMA):
        self.schema = schema
        self.calls = []

    def convert_database(self, db_root, db_name, fk_graph_dir):
        self.calls.append((db_root, db_name, fk_graph_dir))
        return self.schema


@pytest.fixture
def config(tmp_path):
    db_dir = tmp_path / "database" / DB_NAME
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / f"{DB_NAME}.sqlite")
    conn.execute("CREATE TABLE singer (singer_id INTEGER)")
    conn.commit()
    conn.close()

    return {
        "dataset": {"db_path": str(tmp_path / "database")},
        "fk_graph": {"cache_path": str(tmp_path / "fk_graphs")},
        "mongodb": {"host": "localhost", "port": 27017,
                    "schema_cache": str(tmp_path / "mongodb_cache")},
    }


def test_migrate_writes_schema_cache(config):
    converter = StubConverter()
    schema = migrate(DB_NAME, config, converter=converter)

    assert schema == FAKE_SCHEMA
    assert converter.calls == [
        (config["dataset"]["db_path"], DB_NAME, config["fk_graph"]["cache_path"])
    ]

    cache_path = Path(config["mongodb"]["schema_cache"]) / f"{DB_NAME}_schema.json"
    assert cache_path.exists()
    assert json.loads(cache_path.read_text()) == FAKE_SCHEMA


def test_migrate_missing_database_raises(config):
    with pytest.raises(FileNotFoundError, match="no_such_db"):
        migrate("no_such_db", config, converter=StubConverter())


def test_format_summary_includes_row_counts_and_total():
    out = format_summary(DB_NAME, FAKE_SCHEMA)
    assert "singer" in out
    assert "concert" in out
    assert "2 docs" in out
    assert "total" in out
    assert "2 docs across 2 collections" in out
