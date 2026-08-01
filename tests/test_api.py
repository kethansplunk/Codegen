"""
Phase 20B — FastAPI backend tests.

Same approach as test_router.py: real LangGraph graph, real POSG, a real
temp SQLite DB, but SchemaLinker/SAR/Generator stubbed so this runs on Mac
with no GPU and no checkpoints. The stub Router is inserted directly into
api._router_cache, bypassing _get_router's config-loading/model-build path
entirely — so these tests don't need configs/config.yaml to point at real
checkpoints either.

    pytest tests/test_api.py -v
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api  # noqa: E402
from src.router.langgraph_router import Router  # noqa: E402
from tests.test_router import StubGenerator, StubLinker, StubSar  # noqa: E402

DB_NAME = "concert_singer"
SCHEMA = "Table singer, columns: [singer_id, name, age]"
GOOD = "SELECT count(*) FROM singer"


@pytest.fixture
def db_dir(tmp_path):
    d = tmp_path / "database" / DB_NAME
    d.mkdir(parents=True)
    conn = sqlite3.connect(d / f"{DB_NAME}.sqlite")
    conn.execute("CREATE TABLE singer (singer_id INTEGER, name TEXT, age INTEGER)")
    conn.executemany("INSERT INTO singer VALUES (?, ?, ?)",
                      [(1, "Alice", 30), (2, "Bob", 41)])
    conn.commit()
    conn.close()
    return str(tmp_path / "database")


def make_config(db_dir: str) -> dict:
    return {
        "dataset":   {"db_path": db_dir},
        "generator": {"sql_checkpoint": "unused", "n_candidates": 1, "temperature": 0.8},
        "schema_linker": {}, "sar": {},
        "mongodb":   {"host": "localhost", "port": 27017},
    }


@pytest.fixture
def client(db_dir):
    router = Router(
        track="sql", config=make_config(db_dir),
        linker=StubLinker(), sar=StubSar(),
        generator=StubGenerator([GOOD]),
        verbose=False,
    )
    # Default request fields -> ("sql", "balanced", 3, 1), matching _get_router's
    # cache key exactly, so the real config-loading path is never hit.
    api._router_cache[("sql", "balanced", 3, 1)] = router
    yield TestClient(api.app)
    api._router_cache.clear()


def test_health():
    resp = TestClient(api.app).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query_success(client):
    resp = client.post("/query", json={
        "track": "sql",
        "question": "How many singers are there?",
        "db_name": DB_NAME,
        "schema_text": SCHEMA,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["query"] == GOOD
    assert body["rows"] == [[2]]
    assert body["retries"] == 0


def test_query_reuses_cached_router(client):
    """Two requests with identical settings hit the same cached Router --
    a fresh StubGenerator wasn't built, so a second call still succeeds off
    the same one-shot batch."""
    first = client.post("/query", json={
        "track": "sql", "question": "q1", "db_name": DB_NAME, "schema_text": SCHEMA,
    })
    second = client.post("/query", json={
        "track": "sql", "question": "q2", "db_name": DB_NAME, "schema_text": SCHEMA,
    })
    assert first.status_code == 200
    assert second.status_code == 200
    assert len(api._router_cache) == 1


def test_query_unknown_track_rejected():
    resp = TestClient(api.app).post("/query", json={
        "track": "graphql", "question": "q", "db_name": DB_NAME,
    })
    assert resp.status_code == 422  # pydantic Literal validation


def test_databases_endpoint_shape():
    resp = TestClient(api.app).get("/databases")
    assert resp.status_code == 200
    body = resp.json()
    assert "db_dir" in body
    assert isinstance(body["databases"], list)
