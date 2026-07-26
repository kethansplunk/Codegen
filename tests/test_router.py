"""
Phase 17 — LangGraph router tests.

Runs on Mac with no GPU and no model checkpoints: SchemaLinker / SAR / Generator
are stubbed, but everything below them is the real thing — the real LangGraph
state machine, the real POSG evaluation, and a real SQLite database in a tmpdir
so execution failures are genuine sqlite3 errors rather than mocked ones.

    pytest tests/test_router.py -v
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.router.langgraph_router import Router, _rank   # noqa: E402

DB_NAME = "concert_singer"
SCHEMA  = "Table singer, columns: [singer_id, name, age]"

GOOD  = "SELECT count(*) FROM singer"
GOOD2 = "SELECT name FROM singer"
BAD   = "SELECT count(*) FROM singerz"          # no such table
BAD2  = "SELECT nope FROM singer"               # no such column


# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------

@pytest.fixture
def db_dir(tmp_path):
    """A real Spider-layout SQLite DB, so executability is genuinely evaluated."""
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
        "generator": {"sql_checkpoint": "unused", "n_candidates": 5, "temperature": 0.8},
        "schema_linker": {}, "sar": {},
        "mongodb":   {"host": "localhost", "port": 27017},
    }


class StubLinker:
    def __init__(self, key_fields=("singer.name", "singer.singer_id")):
        self.key_fields = list(key_fields)

    def link(self, question, schema):
        return self.key_fields


class StubSar:
    def __init__(self, examples=None):
        self.examples = examples if examples is not None else [
            {"question": "How many artists?", "sql": "SELECT count(*) FROM singer"},
        ]

    def retrieve(self, question, top_k=3):
        return self.examples


class StubGenerator:
    """
    Returns `batches[0]` on the first call, `batches[1]` on the self-correction
    call, and so on. Records whether each call carried correction context.
    """

    def __init__(self, *batches):
        self.batches = list(batches)
        self.calls   = []

    def generate(self, question, schema, key_fields, sar_examples,
                 previous_attempt=None, error=None):
        self.calls.append({"previous_attempt": previous_attempt, "error": error})
        idx = min(len(self.calls) - 1, len(self.batches) - 1)
        return list(self.batches[idx])


def make_router(db_dir, generator, linker=None, sar=None, **kw):
    return Router(
        track="sql",
        config=make_config(db_dir),
        linker=linker or StubLinker(),
        sar=sar or StubSar(),
        generator=generator,
        verbose=False,
        **kw,
    )


def run(router):
    return router.run("How many singers are there?", DB_NAME, schema=SCHEMA)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_first_candidate_succeeds_without_retrying(db_dir):
    gen = StubGenerator([GOOD, GOOD, GOOD2])
    r = run(make_router(db_dir, gen))

    assert r["status"] == "ok"
    assert r["query"] == GOOD
    assert r["rows"] == [(2,)]
    assert r["retries"] == 0
    assert len(gen.calls) == 1                      # no self-correction call
    assert gen.calls[0]["error"] is None


def test_track_is_fixed_at_construction(db_dir):
    assert make_router(db_dir, StubGenerator([GOOD])).track == "sql"
    with pytest.raises(ValueError, match="track must be"):
        Router(track="graphql", config=make_config(db_dir))


# ---------------------------------------------------------------------------
# Retry ladder: candidates first
# ---------------------------------------------------------------------------

def test_a_working_candidate_in_the_batch_never_needs_a_retry(db_dir):
    """
    POSG's executability dimension *runs* every candidate before ranking, so a
    batch containing any working query gets it ranked first and the retry
    ladder never fires. This is the common case, and it means the ladder is
    reached only when POSG had nothing executable to offer — see the test
    below. Documented as a test so a future ranking change that breaks the
    property is caught.
    """
    gen = StubGenerator([BAD, BAD2, GOOD])
    r = run(make_router(db_dir, gen))

    assert r["status"] == "ok"
    assert r["query"] == GOOD
    assert r["retries"] == 0
    assert len(gen.calls) == 1


def test_walks_candidates_before_re_prompting_when_none_are_executable(db_dir):
    """The retry ladder's real trigger: POSG's Pareto front came back empty."""
    gen = StubGenerator([BAD, BAD2, "SELECT * FROM nope"], [GOOD])
    r = run(make_router(db_dir, gen))

    assert r["status"] == "ok"
    assert r["query"] == GOOD
    assert r["retries"] >= 1, "should have tried more than one candidate"
    candidate_attempts = [h for h in r["history"] if h["source"] == "posg_candidate"]
    assert len(candidate_attempts) >= 2, "should exhaust candidates before re-prompting"
    assert all(h["error"] for h in candidate_attempts)


def test_history_records_each_attempt(db_dir):
    gen = StubGenerator([BAD, BAD2], [GOOD])
    r = run(make_router(db_dir, gen))

    assert len(r["history"]) == r["retries"] + 1
    assert [h["attempt"] for h in r["history"]] == list(range(len(r["history"])))
    assert r["history"][-1]["error"] is None        # last attempt is the one that worked
    assert [h["source"] for h in r["history"]][-1] == "self_correct"


def test_duplicate_candidates_do_not_burn_retries(db_dir):
    # Five identical bad candidates collapse to one, so the router should reach
    # self-correction immediately instead of retrying the same failing query.
    gen = StubGenerator([BAD] * 5, [GOOD])
    r = run(make_router(db_dir, gen))

    assert len(r["ranked"]) == 1 or r["status"] == "ok"
    assert len(gen.calls) == 2, "should have re-prompted rather than retried a duplicate"
    assert r["status"] == "ok"
    assert r["query"] == GOOD


# ---------------------------------------------------------------------------
# Retry ladder: self-correction
# ---------------------------------------------------------------------------

def test_self_correction_fires_only_after_candidates_exhausted(db_dir):
    gen = StubGenerator([BAD, BAD2], [GOOD])
    r = run(make_router(db_dir, gen))

    assert r["status"] == "ok"
    assert r["query"] == GOOD
    assert len(gen.calls) == 2, "exactly one re-prompt"

    correction = gen.calls[1]
    assert correction["previous_attempt"] in (BAD, BAD2)
    assert correction["error"] is not None
    assert "OperationalError" in correction["error"]
    assert any(h["source"] == "self_correct" for h in r["history"])


def test_self_correction_is_attempted_at_most_once(db_dir):
    gen = StubGenerator([BAD, BAD2], [BAD, BAD2], [GOOD])
    r = run(make_router(db_dir, gen))

    assert r["status"] == "failed"
    assert len(gen.calls) == 2, "must not re-prompt repeatedly"


def test_self_correction_returning_nothing_does_not_replay_the_failed_query(db_dir):
    """
    An empty self-correction batch must terminate, not fall through the
    unconditional self_correct -> select edge and re-execute the query that
    just failed (which would also mislabel it as a self-correction attempt).
    """
    gen = StubGenerator([BAD, BAD2], [])
    r = run(make_router(db_dir, gen))

    assert r["status"] == "failed"
    corrections = [h for h in r["history"] if h["source"] == "self_correct"]
    assert corrections == [], f"nothing was generated, so nothing should be attempted: {corrections}"
    queries = [h["query"] for h in r["history"]]
    assert len(queries) == len(set(queries)), f"a query was replayed: {queries}"


def test_gives_up_after_max_retries(db_dir):
    gen = StubGenerator([BAD, BAD2, BAD, BAD2], [BAD, BAD2])
    r = run(make_router(db_dir, gen, max_retries=3))

    assert r["status"] == "failed"
    assert r["retries"] <= 3
    assert r["error"] is not None
    assert len(r["history"]) <= 4                   # first attempt + at most 3 retries


def test_max_retries_zero_disables_the_ladder(db_dir):
    gen = StubGenerator([BAD, GOOD])
    r = run(make_router(db_dir, gen, max_retries=0))

    assert r["retries"] == 0
    assert len(r["history"]) == 1
    assert len(gen.calls) == 1


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_ranked_first_matches_what_posg_would_select(db_dir):
    """The retry ladder must not change which query POSG picks first."""
    import os

    from src.pipeline_sql import _schema_links_set
    from src.posg.posg_sql import ParetoOptimal

    candidates = [BAD, GOOD2, GOOD, BAD2]
    key_fields = ["singer.name", "singer.singer_id"]
    examples   = ["SELECT count(*) FROM singer"]

    pareto = ParetoOptimal(db_path=os.path.join(db_dir, DB_NAME, f"{DB_NAME}.sqlite"))
    links  = _schema_links_set(key_fields)
    ranked = _rank(pareto.evaluate_candidates(candidates, links, examples),
                   pareto.find_pareto_optimal(pareto.evaluate_candidates(candidates, links, examples)),
                   "balanced", key=lambda c: c.sql)

    assert ranked[0] == pareto.select_final_sql(candidates, links, examples, strategy="balanced")


def test_ranking_puts_executable_before_inexecutable(db_dir):
    gen = StubGenerator([BAD, BAD2, GOOD, GOOD2])
    r = run(make_router(db_dir, gen))

    executable = {GOOD, GOOD2}
    positions  = [i for i, q in enumerate(r["ranked"]) if q in executable]
    bad_pos    = [i for i, q in enumerate(r["ranked"]) if q not in executable]
    assert max(positions) < min(bad_pos), f"inexecutable ranked too high: {r['ranked']}"


# ---------------------------------------------------------------------------
# Degraded inputs
# ---------------------------------------------------------------------------

def test_missing_database_is_not_treated_as_a_failure(tmp_path):
    """No local .sqlite => selected but not executed; retrying can't help."""
    gen = StubGenerator([GOOD, GOOD2])
    r = run(make_router(str(tmp_path / "nonexistent"), gen))

    assert r["status"] == "ok"
    assert r["rows"] is None
    assert r["retries"] == 0
    assert len(gen.calls) == 1


def test_empty_key_fields_still_produces_a_query(db_dir):
    gen = StubGenerator([GOOD, GOOD2])
    r = run(make_router(db_dir, gen, linker=StubLinker(key_fields=[])))

    assert r["status"] == "ok"
    assert r["key_fields"] == []
    assert r["query"] in (GOOD, GOOD2)


def test_sar_drops_the_question_itself(db_dir):
    question = "How many singers are there?"
    sar = StubSar([
        {"question": question, "sql": "SELECT count(*) FROM singer"},   # leaks gold
        {"question": "Other?", "sql": "SELECT name FROM singer"},
    ])
    r = make_router(db_dir, StubGenerator([GOOD]), sar=sar).run(question, DB_NAME, schema=SCHEMA)

    assert all(ex["question"] != question for ex in r["sar_examples"])
    assert len(r["sar_examples"]) == 1


def test_empty_generator_output_raises(db_dir):
    with pytest.raises(RuntimeError, match="expected a non-empty list"):
        run(make_router(db_dir, StubGenerator([])))


def test_close_is_idempotent(db_dir):
    router = make_router(db_dir, StubGenerator([GOOD]))
    router.close()
    router.close()


def test_context_manager_closes(db_dir):
    with make_router(db_dir, StubGenerator([GOOD])) as router:
        assert run(router)["status"] == "ok"


# ---------------------------------------------------------------------------
# NoSQL track — same ladder, dict-shaped queries, no live Mongo
# ---------------------------------------------------------------------------

def test_nosql_track_retries_and_self_corrects(monkeypatch):
    """Drives the NoSQL adapter with a fake Mongo so no server is needed."""
    import src.router.langgraph_router as mod

    GOOD_MQL = '{"collection": "singer", "pipeline": [{"$count": "n"}]}'
    BAD_MQL  = '{"collection": "singer", "pipeline": [{"$bogus": 1}]}'

    class FakeParetoMQL:
        """Only what the adapter touches: evaluate/find/db-execute + close."""

        def __init__(self, db_name, mongo_uri):
            self.closed = False

        def evaluate_candidates(self, cands, links, examples):
            from dataclasses import dataclass

            @dataclass
            class C:
                collection: str
                pipeline: list
                index: int

            @dataclass
            class S:
                executability: float
                schema_conformity: float
                example_consistency: float

            out = []
            for i, c in enumerate(cands):
                ok = not any("$bogus" in s for s in map(str, c["pipeline"]))
                out.append((C(c["collection"], c["pipeline"], i),
                            S(1.0 if ok else 0.0, 1.0 if ok else 0.0, 0.5)))
            return out

        def find_pareto_optimal(self, evaluated):
            return [c for c, s in evaluated if s.executability > 0][:1]

        @property
        def db(self):
            outer = self

            class Coll:
                def aggregate(self, pipeline, maxTimeMS=None):
                    if any("$bogus" in s for s in map(str, pipeline)):
                        raise RuntimeError("Unrecognized pipeline stage name: '$bogus'")
                    return [{"n": 2}]

            return {"singer": Coll()}

        def close(self):
            self.closed = True

    monkeypatch.setattr(mod, "_NoSqlTrack", mod._NoSqlTrack)
    monkeypatch.setattr("src.posg.posg_nosql.ParetoOptimalMQL", FakeParetoMQL, raising=False)
    import src.posg.posg_nosql as pn
    monkeypatch.setattr(pn, "ParetoOptimalMQL", FakeParetoMQL)

    gen = StubGenerator([BAD_MQL, BAD_MQL], [GOOD_MQL])
    router = Router(track="nosql", config=make_config("unused"),
                    linker=StubLinker(["singer.name"]), sar=StubSar([]),
                    generator=gen, verbose=False)
    r = router.run("How many singers?", "concert_singer", schema=SCHEMA)
    router.close()

    assert r["track"] == "nosql"
    assert r["status"] == "ok"
    assert r["query"] == {"collection": "singer", "pipeline": [{"$count": "n"}]}
    assert r["rows"] == [{"n": 2}]
    assert len(gen.calls) == 2, "should have needed one self-correction"
    assert "bogus" in gen.calls[1]["error"].lower()
