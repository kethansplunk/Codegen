"""
Phase 17 — LangGraph router + self-correction.

Wraps the Phase 16 pipeline (src/pipeline_sql.py / src/pipeline_nosql.py) in a
LangGraph state machine that actually *executes* the selected query and retries
on failure:

    schema_link -> retrieve -> generate -> select -> execute -+-> [ok] -> END
                                             ^                |
                                             +----------------+ [next candidate]
                                             |                |
                                        self_correct <--------+ [candidates exhausted]

Routing is session-based ("Option A", docs/architecture.md): the caller fixes
the track when constructing the Router rather than classifying per question.
A developer works against one database type at a time, so this removes a
classifier that could be wrong.

Retry ladder (max_retries=3 by default):

  retry 1..n-1  Take the next candidate from the POSG ranking. These were
                already generated, so this costs no GPU time and stays fully
                in-distribution for the fine-tuned adapter.
  retry n       Only once the ranked candidates are exhausted, re-prompt the
                generator with the failing query + its execution error
                (GeneratorInfer.generate(previous_attempt=..., error=...)).
                This is the expensive, out-of-distribution path -- see the
                note in src/generator/infer.py's _build_user_prompt.

SchemaLinker / SAR / Generator are built once per Router and reused for every
question in the session, via the injection parameters added to the Phase 16
pipeline modules.

Usage:
    router = Router(track="sql", config=yaml.safe_load(open("configs/config.yaml")))
    result = router.run("How many singers are there?", db_name="concert_singer")
    print(result["query"], result["rows"], result["status"])
    router.close()
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, TypedDict

MAX_RETRIES = 3

# select_final_sql / select_final_mql use these to break ties on the Pareto
# front; the router's ranking mirrors them so ranked[0] is always exactly what
# POSG would have selected on its own.
_STRATEGY_WEIGHTS = {
    "balanced":         (0.5, 0.5),
    "schema_priority":  (0.7, 0.3),
    "example_priority": (0.3, 0.7),
}


class RouterState(TypedDict, total=False):
    """State threaded through the graph. Nodes return partial updates."""
    question:       str
    db_name:        str
    schema:         str
    key_fields:     list
    sar_examples:   list
    ranked:         list          # candidates in POSG preference order
    attempt_idx:    int           # index into `ranked` currently being tried
    query:          Any           # str (sql) or dict (mql) currently selected
    rows:           Optional[list]
    error:          Optional[str]
    retries:        int
    self_corrected: bool          # the one re-prompt has been spent
    history:        list          # [{attempt, source, query, error}] for debugging
    status:         str           # "ok" | "failed"


# ---------------------------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------------------------

def _rank(evaluated, front, strategy: str, key: Callable[[Any], Any]) -> list:
    """
    Order POSG-evaluated candidates best-first so the retry loop can walk them.

    Pareto-front members come first (by the same weighted score select_final_*
    uses to break ties), then any other executable candidate, then inexecutable
    ones as a last resort. The sort is stable and the weights match
    select_final_*, so element 0 is exactly what POSG would have picked alone.
    """
    ws, we    = _STRATEGY_WEIGHTS.get(strategy, (0.5, 0.5))
    score_map = {c.index: s for c, s in evaluated}
    front_idx = {c.index for c in front}

    def weighted(idx: int) -> float:
        s = score_map[idx]
        return ws * s.schema_conformity + we * s.example_consistency

    def tier(c) -> int:
        if c.index in front_idx:
            return 0
        return 1 if score_map[c.index].executability > 0 else 2

    ordered = sorted((c for c, _ in evaluated), key=lambda c: (tier(c), -weighted(c.index)))

    # Duplicate candidates are common at temperature 0.8 -- retrying a query
    # byte-identical to the one that just failed would burn a retry for nothing.
    seen, unique = set(), []
    for c in ordered:
        q = key(c)
        k = q if isinstance(q, str) else json.dumps(q, sort_keys=True, default=str)
        if k in seen:
            continue
        seen.add(k)
        unique.append(q)
    return unique


# ---------------------------------------------------------------------------
# Track adapters — everything that differs between SQL and NoSQL
# ---------------------------------------------------------------------------

class _SqlTrack:
    track = "sql"

    def __init__(self, config: dict):
        self.config = config
        self.db_dir = config["dataset"]["db_path"]

    def format_schema(self, db_name: str) -> str:
        from scripts.build_cot_data import format_schema_text
        return format_schema_text(db_name)

    def checkpoint(self) -> str:
        return self.config["generator"]["sql_checkpoint"]

    def db_path(self, db_name: str) -> str:
        import os
        return os.path.join(self.db_dir, db_name, f"{db_name}.sqlite")

    def parse_candidates(self, raw: list) -> list:
        """Generator output -> executable query objects.

        Runs each candidate through fix_ambiguous_columns() -- a mechanical,
        provably-safe rewrite for the "ambiguous column name" execution
        failures found in the Phase 18 error analysis. See
        src/generator/sql_fixups.py for why this is safe to apply blindly.
        """
        from src.generator.sql_fixups import fix_ambiguous_columns
        return [fix_ambiguous_columns(c) for c in raw]

    def rank_candidates(self, raw: list, key_fields: list, sar_examples: list,
                        db_name: str, strategy: str) -> list:
        import os

        from src.pipeline_sql import _schema_links_set
        from src.posg.posg_sql import ParetoOptimal

        db_path  = self.db_path(db_name)
        pareto   = ParetoOptimal(db_path=db_path if os.path.exists(db_path) else None)
        links    = _schema_links_set(key_fields)
        examples = [ex["sql"] for ex in sar_examples if "sql" in ex]

        evaluated = pareto.evaluate_candidates(self.parse_candidates(raw), links, examples)
        front     = pareto.find_pareto_optimal(evaluated)
        return _rank(evaluated, front, strategy, key=lambda c: c.sql)

    def execute(self, query: str, db_name: str):
        """
        Run the SQL, returning (rows, error).

        (None, None) is a third, distinct outcome meaning "not executed" -- the
        local .sqlite file is missing, so we can neither confirm success nor
        diagnose a failure. It is deliberately not an error: retrying cannot
        conjure the database, so burning all three retries on it would be
        pointless. The caller reports status "ok" with rows=None.
        """
        import os
        import sqlite3

        db_path = self.db_path(db_name)
        if not os.path.exists(db_path):
            return None, None
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            cur = conn.cursor()
            cur.execute(query)
            return cur.fetchall(), None
        except Exception as e:
            # sqlite3's message is the actual self-correction signal
            # ("no such column: T1.singer_name"), so keep the type prefix too.
            return None, f"{type(e).__name__}: {e}"
        finally:
            if conn is not None:
                conn.close()

    def render(self, query) -> str:
        return query if isinstance(query, str) else str(query)

    def close(self):
        pass


class _NoSqlTrack:
    track = "nosql"

    def __init__(self, config: dict):
        self.config    = config
        self.mongo_uri = f"mongodb://{config['mongodb']['host']}:{config['mongodb']['port']}"
        self._pareto_cache: dict = {}

    def format_schema(self, db_name: str) -> str:
        from scripts.build_nosql_cot_data import format_schema_text
        return format_schema_text(db_name)

    def checkpoint(self) -> str:
        return self.config["generator"]["nosql_checkpoint"]

    def _pareto(self, db_name: str):
        """One long-lived Mongo connection per database, for the whole session."""
        from src.posg.posg_nosql import ParetoOptimalMQL
        if db_name not in self._pareto_cache:
            self._pareto_cache[db_name] = ParetoOptimalMQL(db_name=db_name, mongo_uri=self.mongo_uri)
        return self._pareto_cache[db_name]

    def parse_candidates(self, raw: list) -> list:
        """Generator output -> {'collection': ..., 'pipeline': [...]} dicts."""
        from src.pipeline_nosql import _parse_candidate
        return [_parse_candidate(c) for c in raw]

    def rank_candidates(self, raw: list, key_fields: list, sar_examples: list,
                        db_name: str, strategy: str) -> list:
        from src.pipeline_nosql import _collection_links_set

        parsed = self.parse_candidates(raw)
        pareto = self._pareto(db_name)
        links  = _collection_links_set(key_fields)

        evaluated = pareto.evaluate_candidates(parsed, links, sar_examples)
        front     = pareto.find_pareto_optimal(evaluated)
        return _rank(evaluated, front, strategy,
                     key=lambda c: {"collection": c.collection, "pipeline": c.pipeline})

    def execute(self, query: dict, db_name: str):
        pareto = self._pareto(db_name)
        try:
            rows = list(pareto.db[query.get("collection", "")]
                        .aggregate(query.get("pipeline", []), maxTimeMS=3000))
            return rows, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def render(self, query) -> str:
        return json.dumps(query, ensure_ascii=False, default=str)

    def close(self):
        for pareto in self._pareto_cache.values():
            try:
                pareto.close()
            except Exception:
                pass
        self._pareto_cache.clear()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    """
    Session-scoped LangGraph router for one track.

    Args:
        track:       "sql" or "nosql". Fixed for the session (Option A routing).
        config:      Loaded configs/config.yaml dict.
        strategy:    POSG tie-break strategy.
        max_retries: Execution retries allowed after the first attempt.
        seed:        Forwarded to GeneratorInfer.
        linker, sar, generator:
                     Pre-built components. Anything left None is constructed
                     lazily on first use, so tests can drive the graph without
                     loading a 7B model.
        verbose:     Print per-node progress.
    """

    def __init__(
        self,
        track: str,
        config: dict,
        strategy: str = "balanced",
        max_retries: int = MAX_RETRIES,
        seed: int | None = None,
        linker=None,
        sar=None,
        generator=None,
        verbose: bool = True,
    ):
        if track not in ("sql", "nosql"):
            raise ValueError(f"track must be 'sql' or 'nosql', got {track!r}")

        self.track       = track
        self.config      = config
        self.strategy    = strategy
        self.max_retries = max_retries
        self.seed        = seed
        self.verbose     = verbose

        self._adapter   = _SqlTrack(config) if track == "sql" else _NoSqlTrack(config)
        self._linker    = linker
        self._sar       = sar
        self._generator = generator

        self.graph = self._build_graph()

    # --- lazy component construction (built once, reused every question) ---

    @property
    def linker(self):
        if self._linker is None:
            from src.schema_linker.linker import get_schema_linker
            self._log("building SchemaLinker ...")
            self._linker = get_schema_linker(self.config["schema_linker"], track=self.track)
        return self._linker

    @property
    def sar(self):
        if self._sar is None:
            from src.sar.infer import get_sar_retriever
            self._log("building SAR retriever ...")
            self._sar = get_sar_retriever(self.config["sar"], track=self.track)
        return self._sar

    @property
    def generator(self):
        if self._generator is None:
            from src.generator.infer import GeneratorInfer
            self._log("building Generator ...")
            self._generator = GeneratorInfer(
                checkpoint_path=self._adapter.checkpoint(),
                track=self.track,
                n_candidates=self.config["generator"]["n_candidates"],
                temperature=self.config["generator"]["temperature"],
                seed=self.seed,
            )
        return self._generator

    def _log(self, msg: str):
        if self.verbose:
            print(f"[router:{self.track}] {msg}")

    # --- nodes ---

    def _node_schema_link(self, state: RouterState) -> dict:
        schema     = state.get("schema") or self._adapter.format_schema(state["db_name"])
        key_fields = self.linker.link(question=state["question"], schema=schema)
        if not key_fields:
            # Mirrors the Phase 16 pipeline warning. Empty key_fields make
            # schema_conformity a constant for every candidate -- 0.0 on the SQL
            # track, 1.0 on the NoSQL track, which short-circuits on empty links
            # -- so either way the objective carries no signal and ranking
            # silently stops being schema-aware.
            self._log("[warn] SchemaLinker returned no key fields — POSG ranking "
                      "will use executability + example consistency only")
        self._log(f"schema_link -> {len(key_fields)} key fields")
        return {"schema": schema, "key_fields": key_fields}

    def _node_retrieve(self, state: RouterState) -> dict:
        retrieved = self.sar.retrieve(state["question"], top_k=4)
        # Drop an exact question match: on the train split SAR retrieves the
        # question itself, handing the generator the gold query as an example.
        examples = [ex for ex in retrieved
                    if ex["question"].strip() != state["question"].strip()][:3]
        self._log(f"retrieve -> {len(examples)} SAR examples")
        return {"sar_examples": examples}

    def _node_generate(self, state: RouterState) -> dict:
        raw = self.generator.generate(
            state["question"], state["schema"], state["key_fields"], state["sar_examples"],
        )
        if not raw:
            raise RuntimeError(
                f"Generator.generate() returned {raw!r} (expected a non-empty list) "
                f"for question={state['question']!r} db_name={state['db_name']!r}."
            )
        ranked = self._adapter.rank_candidates(
            raw, state["key_fields"], state["sar_examples"], state["db_name"], self.strategy,
        )
        self._log(f"generate -> {len(raw)} candidates, {len(ranked)} unique after ranking")
        return {"ranked": ranked, "attempt_idx": 0}

    def _node_select(self, state: RouterState) -> dict:
        ranked = state.get("ranked", [])
        idx    = state.get("attempt_idx", 0)
        return {"query": ranked[idx] if idx < len(ranked) else None}

    def _node_execute(self, state: RouterState) -> dict:
        query = state.get("query")
        if query is None:
            return {"rows": None, "error": "no candidate left to execute", "status": "failed"}

        rows, error = self._adapter.execute(query, state["db_name"])
        entry = {
            "attempt": state.get("retries", 0),
            "source":  "self_correct" if state.get("self_corrected") else "posg_candidate",
            "query":   self._adapter.render(query),
            "error":   error,
        }
        history = state.get("history", []) + [entry]

        if error is None:
            self._log(f"execute -> ok ({'not executed (no local DB)' if rows is None else f'{len(rows)} rows'})")
            return {"rows": rows, "error": None, "status": "ok", "history": history}

        self._log(f"execute -> FAILED: {error}")
        return {"rows": None, "error": error, "status": "failed", "history": history}

    def _node_self_correct(self, state: RouterState) -> dict:
        self._log("self_correct -> re-prompting generator with the execution error")
        raw = self.generator.generate(
            state["question"], state["schema"], state["key_fields"], state["sar_examples"],
            previous_attempt=self._adapter.render(state["query"]),
            error=state["error"],
        )
        if not raw:
            # Nothing new to try. Clear the candidate list so `select` yields
            # None and `execute` reports "no candidate left" -- returning only
            # self_corrected would leave `ranked`/`attempt_idx` untouched, and
            # the unconditional self_correct -> select edge would re-run the
            # query that just failed and log it a second time, mislabelled as a
            # self-correction attempt.
            self._log("self_correct -> generator returned nothing; giving up")
            return {"ranked": [], "attempt_idx": 0, "self_corrected": True}
        ranked = self._adapter.rank_candidates(
            raw, state["key_fields"], state["sar_examples"], state["db_name"], self.strategy,
        )
        return {"ranked": ranked, "attempt_idx": 0, "self_corrected": True}

    def _node_next_candidate(self, state: RouterState) -> dict:
        return {"attempt_idx": state.get("attempt_idx", 0) + 1,
                "retries":     state.get("retries", 0) + 1}

    def _node_bump_retry(self, state: RouterState) -> dict:
        return {"retries": state.get("retries", 0) + 1}

    # --- edges ---

    def _after_execute(self, state: RouterState) -> str:
        if state.get("status") == "ok":
            return "done"
        if state.get("retries", 0) >= self.max_retries:
            self._log(f"giving up after {self.max_retries} retries")
            return "done"

        has_more = state.get("attempt_idx", 0) + 1 < len(state.get("ranked", []))

        if state.get("self_corrected"):
            # The one re-prompt is spent. Spend whatever budget is left walking
            # the corrected candidate list rather than stopping early.
            return "next_candidate" if has_more else "done"

        # Reserve the last retry for the re-prompt, so a long candidate list
        # can't crowd out self-correction entirely.
        if has_more and state.get("retries", 0) < self.max_retries - 1:
            return "next_candidate"
        return "self_correct"

    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        g = StateGraph(RouterState)
        g.add_node("schema_link",    self._node_schema_link)
        g.add_node("retrieve",       self._node_retrieve)
        g.add_node("generate",       self._node_generate)
        g.add_node("select",         self._node_select)
        g.add_node("execute",        self._node_execute)
        g.add_node("next_candidate", self._node_next_candidate)
        g.add_node("bump_retry",     self._node_bump_retry)
        g.add_node("self_correct",   self._node_self_correct)

        g.set_entry_point("schema_link")
        g.add_edge("schema_link", "retrieve")
        g.add_edge("retrieve",    "generate")
        g.add_edge("generate",    "select")
        g.add_edge("select",      "execute")

        g.add_conditional_edges("execute", self._after_execute, {
            "done":           END,
            "next_candidate": "next_candidate",
            "self_correct":   "bump_retry",
        })
        g.add_edge("next_candidate", "select")
        g.add_edge("bump_retry",     "self_correct")
        g.add_edge("self_correct",   "select")

        return g.compile()

    # --- public API ---

    def run(self, question: str, db_name: str, schema: str | None = None) -> dict:
        """
        Run one question end-to-end, retrying on execution failure.

        Returns a dict with: query (the one that ran, or the last one tried),
        rows, status ("ok"/"failed"), error, retries, history, plus the
        intermediate key_fields / sar_examples / ranked candidates.
        """
        initial: RouterState = {
            "question":       question,
            "db_name":        db_name,
            "schema":         schema,
            "retries":        0,
            "attempt_idx":    0,
            "self_corrected": False,
            "history":        [],
        }
        # recursion_limit guards against a cycle bug in the retry edges; the
        # ladder itself is bounded by max_retries.
        final = self.graph.invoke(initial, {"recursion_limit": 4 * self.max_retries + 20})
        return {
            "question":     question,
            "db_name":      db_name,
            "track":        self.track,
            "query":        final.get("query"),
            "rows":         final.get("rows"),
            "status":       final.get("status", "failed"),
            "error":        final.get("error"),
            "retries":      final.get("retries", 0),
            "history":      final.get("history", []),
            "key_fields":   final.get("key_fields", []),
            "sar_examples": final.get("sar_examples", []),
            "ranked":       final.get("ranked", []),
        }

    def close(self):
        """Release the track's DB connections. Safe to call more than once."""
        self._adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
