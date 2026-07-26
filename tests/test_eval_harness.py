"""
Phase 18 — evaluation harness tests.

Runs on Mac with no GPU and no checkpoints. SchemaLinker / SAR / Generator are
stubbed; the POSG ranking, the EX comparison, and the SQLite execution are real.

    pytest tests/test_eval_harness.py -v
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.harness import (  # noqa: E402
    ABLATIONS,
    EvalHarness,
    format_summary,
    plan_generation_passes,
    save_report,
    summarize,
)
from test_router import StubGenerator, StubLinker, StubSar, make_config  # noqa: E402

DB_NAME = "concert_singer"
SCHEMA  = "Table singer, columns: [singer_id, name, age]"

GOLD      = "SELECT count(*) FROM singer"
EQUIV     = "SELECT count(singer_id) FROM singer"  # different text, same result set
DIFFERENT = "SELECT name FROM singer"              # executes, wrong result
BROKEN    = "SELECT count(*) FROM singerz"         # does not execute


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


def entry(question="How many singers are there?", gold=GOLD, key_fields=("singer.name",)):
    return {"question": question, "db_name": DB_NAME, "schema": SCHEMA,
            "key_fields": list(key_fields), "sql": gold}


def make_harness(db_dir, generator, **kw):
    return EvalHarness(track="sql", config=make_config(db_dir),
                       linker=StubLinker(), sar=StubSar(), generator=generator,
                       verbose=False, **kw)


# ---------------------------------------------------------------------------
# Generation-pass planning
# ---------------------------------------------------------------------------

def test_full_and_no_posg_share_one_generation_pass():
    groups = plan_generation_passes(["full", "no_posg"])
    assert len(groups) == 1
    assert sorted(next(iter(groups.values()))) == ["full", "no_posg"]


def test_four_ablations_need_three_passes():
    groups = plan_generation_passes(ABLATIONS)
    assert len(groups) == 3, f"expected 3 generation passes, got {len(groups)}: {groups}"


def test_ablation_groups_have_correct_inputs():
    groups = plan_generation_passes(ABLATIONS)
    assert groups[(True, True)] == ["full", "no_posg"]      # both components on
    assert groups[(False, True)] == ["no_schema_linker"]    # linker off
    assert groups[(True, False)] == ["no_sar"]              # sar off


def test_unknown_ablation_rejected():
    with pytest.raises(ValueError, match="unknown ablation"):
        plan_generation_passes(["no_generator"])


def test_harness_generates_once_per_group(db_dir):
    gen = StubGenerator([GOLD])
    make_harness(db_dir, gen).evaluate([entry()], ablations=ABLATIONS)
    assert len(gen.calls) == 3, "4 ablations must cost 3 generation passes, not 4"


def test_harness_generates_once_for_full_only(db_dir):
    gen = StubGenerator([GOLD])
    make_harness(db_dir, gen).evaluate([entry()], ablations=["full"])
    assert len(gen.calls) == 1


# ---------------------------------------------------------------------------
# Ablations actually change the generator inputs
# ---------------------------------------------------------------------------

def test_no_schema_linker_blanks_key_fields(db_dir):
    seen = []

    class RecordingGen(StubGenerator):
        def generate(self, question, schema, key_fields, sar_examples, **kw):
            seen.append({"key_fields": list(key_fields), "sar": len(sar_examples)})
            return [GOLD]

    make_harness(db_dir, RecordingGen([GOLD])).evaluate(
        [entry()], ablations=["full", "no_schema_linker"])

    by_kf = {bool(s["key_fields"]) for s in seen}
    assert by_kf == {True, False}, f"expected one pass with and one without key fields: {seen}"
    assert all(s["sar"] > 0 for s in seen), "no_schema_linker must not disturb SAR"


def test_no_sar_blanks_examples(db_dir):
    seen = []

    class RecordingGen(StubGenerator):
        def generate(self, question, schema, key_fields, sar_examples, **kw):
            seen.append({"key_fields": list(key_fields), "sar": len(sar_examples)})
            return [GOLD]

    make_harness(db_dir, RecordingGen([GOLD])).evaluate(
        [entry()], ablations=["full", "no_sar"])

    assert {s["sar"] for s in seen} == {0, 1}, f"expected one pass with no examples: {seen}"
    assert all(s["key_fields"] for s in seen), "no_sar must not disturb key fields"


def test_no_posg_takes_greedy_top1_not_the_ranked_best(db_dir):
    """
    Candidate 0 is broken and candidate 1 is correct. POSG must rank the working
    one first; the no_posg arm must keep the broken candidate 0.
    """
    gen = StubGenerator([BROKEN, GOLD])
    report = make_harness(db_dir, gen).evaluate([entry()], ablations=["full", "no_posg"])
    abl = report["results"][0]["ablations"]

    assert abl["full"]["query"] == GOLD
    assert abl["full"]["ex"] == 1.0
    assert abl["no_posg"]["query"] == BROKEN
    assert abl["no_posg"]["ex"] == 0.0
    assert len(gen.calls) == 1, "both arms come from one generation pass"


# ---------------------------------------------------------------------------
# EX semantics
# ---------------------------------------------------------------------------

def test_ex_is_execution_based_not_string_based(db_dir):
    """Different text, same result set => EX 1.0 but exact_match False."""
    report = make_harness(db_dir, StubGenerator([EQUIV])).evaluate([entry()])
    s = report["results"][0]["ablations"]["full"]

    assert s["ex"] == 1.0
    assert s["exact_match"] is False


def test_executable_but_wrong_answer_scores_zero(db_dir):
    report = make_harness(db_dir, StubGenerator([DIFFERENT])).evaluate([entry()])
    s = report["results"][0]["ablations"]["full"]

    assert s["ex"] == 0.0
    assert s["ex_eligible"] is True
    assert s["pred_error"] is None, "it ran fine, it was just wrong"


def test_inexecutable_prediction_scores_zero_with_reason(db_dir):
    report = make_harness(db_dir, StubGenerator([BROKEN])).evaluate([entry()])
    s = report["results"][0]["ablations"]["full"]

    assert s["ex"] == 0.0
    assert s["ex_eligible"] is True
    assert "did not execute" in s["pred_error"]


def test_missing_database_is_excluded_from_ex_not_scored_zero(tmp_path):
    """A missing .sqlite must not silently deflate EX."""
    report = make_harness(str(tmp_path / "nope"), StubGenerator([GOLD])).evaluate([entry()])
    s = report["results"][0]["ablations"]["full"]

    assert s["ex_eligible"] is False
    assert s["ex"] is None
    assert report["summary"]["full"]["n_scored"] == 0
    assert report["summary"]["full"]["n_unscorable"] == 1
    assert report["summary"]["full"]["ex"] is None, "EX over zero eligible questions is undefined"


def test_bad_gold_query_is_excluded_from_ex(db_dir):
    report = make_harness(db_dir, StubGenerator([GOLD])).evaluate([entry(gold=BROKEN)])
    s = report["results"][0]["ablations"]["full"]

    assert s["ex_eligible"] is False
    assert "gold query did not execute" in s["pred_error"]


def test_ex_denominator_excludes_unscorable_questions(db_dir):
    """EX is the mean over eligible questions, not over all of them."""
    entries = [entry(question="q1"), entry(question="q2", gold=BROKEN)]
    report = make_harness(db_dir, StubGenerator([GOLD])).evaluate(entries)
    summary = report["summary"]["full"]

    assert summary["n_total"] == 2
    assert summary["n_scored"] == 1
    assert summary["ex"] == 1.0, "1/1 eligible, not 1/2 total"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_summary_counts_and_rates(db_dir):
    entries = [entry(question="q1"), entry(question="q2"), entry(question="q3")]
    gen = StubGenerator([DIFFERENT])          # always wrong but executable
    report = make_harness(db_dir, gen).evaluate(entries)
    s = report["summary"]["full"]

    assert s["n_total"] == 3 and s["n_scored"] == 3
    assert s["ex"] == 0.0
    assert s["exact_match"] == 0.0


def test_format_summary_flags_target_and_exclusions(db_dir):
    entries = [entry(question="q1"), entry(question="q2", gold=BROKEN)]
    report = make_harness(db_dir, StubGenerator([GOLD])).evaluate(entries)
    text = format_summary(report)

    assert "EX > 82%" in text
    assert "MEETS" in text                       # 1/1 eligible = 100%
    assert "excluded from EX" in text            # the unscorable one is called out
    assert "1/2" in text


def test_format_summary_reports_ablation_deltas(db_dir):
    gen = StubGenerator([BROKEN, GOLD])
    report = make_harness(db_dir, gen).evaluate([entry()], ablations=["full", "no_posg"])
    text = format_summary(report)

    assert "Ablation deltas" in text
    assert "-100.0%" in text, f"no_posg should be 100 points worse here:\n{text}"


def test_summarize_handles_empty_results():
    s = summarize([], ["full"], "sql")
    assert s["full"]["ex"] is None and s["full"]["n_total"] == 0


def test_report_round_trips_to_json(db_dir, tmp_path):
    import json

    report = make_harness(db_dir, StubGenerator([GOLD])).evaluate([entry()])
    path = save_report(report, str(tmp_path / "results" / "r.json"))
    loaded = json.load(open(path))

    assert loaded["track"] == "sql"
    assert loaded["summary"]["full"]["ex"] == 1.0
    assert loaded["results"][0]["ablations"]["full"]["query"] == GOLD


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def test_invalid_track_rejected(db_dir):
    with pytest.raises(ValueError, match="track must be"):
        EvalHarness(track="cypher", config=make_config(db_dir))


def test_cached_key_fields_are_reused_without_calling_the_linker(db_dir):
    class ExplodingLinker:
        def link(self, question, schema):
            raise AssertionError("SchemaLinker must not be called when key_fields are cached")

    h = EvalHarness(track="sql", config=make_config(db_dir), linker=ExplodingLinker(),
                    sar=StubSar(), generator=StubGenerator([GOLD]), verbose=False)
    assert h.evaluate([entry()])["summary"]["full"]["ex"] == 1.0


def test_linker_is_called_when_key_fields_absent(db_dir):
    e = entry()
    del e["key_fields"]
    calls = []

    class CountingLinker(StubLinker):
        def link(self, question, schema):
            calls.append(question)
            return ["singer.name"]

    h = EvalHarness(track="sql", config=make_config(db_dir), linker=CountingLinker(),
                    sar=StubSar(), generator=StubGenerator([GOLD]), verbose=False)
    h.evaluate([e])
    assert len(calls) == 1


def test_empty_generator_output_raises(db_dir):
    with pytest.raises(RuntimeError, match="returned"):
        make_harness(db_dir, StubGenerator([])).evaluate([entry()])


# ---------------------------------------------------------------------------
# CLI helpers (no model loading)
# ---------------------------------------------------------------------------

def test_harness_forwards_seed_to_the_generator(db_dir, monkeypatch):
    """
    Ablation deltas compare separate generation passes, so an unseeded
    generator would confound each component's effect with sampling noise.
    """
    captured = {}

    class FakeGeneratorInfer:
        def __init__(self, **kw):
            captured.update(kw)

        def generate(self, *a, **kw):
            return [GOLD]

    import src.generator.infer as infer_mod
    monkeypatch.setattr(infer_mod, "GeneratorInfer", FakeGeneratorInfer)

    h = EvalHarness(track="sql", config=make_config(db_dir), seed=1234,
                    linker=StubLinker(), sar=StubSar(), verbose=False)
    report = h.evaluate([entry()])

    assert captured["seed"] == 1234
    assert report["seed"] == 1234, "the seed must be recorded in the report"


def test_harness_seeds_by_default(db_dir):
    assert EvalHarness(track="sql", config=make_config(db_dir), verbose=False).seed == 0


def test_batch_n_zero_means_whole_file(tmp_path):
    """--n 0 must not produce an empty sample and a ZeroDivisionError."""
    import json

    from scripts.run_router import _batch

    data = [{"question": "q1", "db_name": DB_NAME}, {"question": "q2", "db_name": DB_NAME}]
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(data))

    seen = []

    class FakeRouter:
        def run(self, question, db_name, schema=None):
            seen.append(question)
            return {"status": "ok", "query": GOLD, "rows": [], "retries": 0,
                    "error": None, "history": []}

    _batch(FakeRouter(), str(path), 0, seed=0)
    assert seen == ["q1", "q2"]


def test_batch_handles_empty_file(tmp_path, capsys):
    import json

    from scripts.run_router import _batch

    path = tmp_path / "empty.json"
    path.write_text(json.dumps([]))

    class ExplodingRouter:
        def run(self, *a, **kw):
            raise AssertionError("must not run anything")

    _batch(ExplodingRouter(), str(path), 0, seed=0)      # must not raise
    assert "nothing to evaluate" in capsys.readouterr().out


def test_hard_filter_sql():
    from scripts.run_eval import _is_hard

    assert _is_hard({"sql": "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id"}, "sql")
    assert _is_hard({"sql": "SELECT count(*) FROM t GROUP BY x"}, "sql")
    assert _is_hard({"sql": "SELECT a FROM t WHERE b IN (SELECT c FROM u)"}, "sql")
    assert not _is_hard({"sql": "SELECT name FROM singer"}, "sql")


def test_hard_filter_nosql():
    from scripts.run_eval import _is_hard

    assert _is_hard({"mql": {"pipeline": [{"$lookup": {}}]}}, "nosql")
    assert _is_hard({"mql": {"pipeline": [{"$group": {}}]}}, "nosql")
    assert _is_hard({"mql_pipeline": [{"$match": {}}, {"$sort": {}}, {"$limit": 1}]}, "nosql")
    assert not _is_hard({"mql": {"pipeline": [{"$count": "n"}]}}, "nosql")


def test_baseline_prompt_is_completion_style():
    from scripts.run_baseline import build_prompt

    p = build_prompt("Table singer, columns: [name]", "How many singers?")
    assert p.rstrip().endswith("SELECT"), "must leave an open SELECT for the model to continue"
    assert "-- Question: How many singers?" in p
    assert "-- Table singer" in p, "schema lines must be commented out"


@pytest.mark.parametrize("completion,expected", [
    ("count(*) FROM singer;\n-- next",   "SELECT count(*) FROM singer"),
    ("name FROM singer\n\n-- another",   "SELECT name FROM singer"),
    ("count(*)\nFROM singer\n-- done",   "SELECT count(*) FROM singer"),
    ("count(*) FROM singer",             "SELECT count(*) FROM singer"),
])
def test_baseline_extraction_cuts_at_statement_boundary(completion, expected):
    from scripts.run_baseline import extract_sql
    assert extract_sql(completion) == expected
