"""
Phase 18 — evaluation harness + component ablation.

Scores the full pipeline on a held-out set and replicates SchemaRAG's Table 5
ablation, for either track. Four configurations:

    full              SchemaLinker + SAR + POSG (the shipped pipeline)
    no_schema_linker  key_fields forced to []   -> Generator sees "N/A" key
                      fields, and POSG's schema_conformity goes flat
    no_sar            sar_examples forced to [] -> Generator sees "N/A"
                      examples, and POSG's example_consistency goes flat
    no_posg           POSG selection replaced by the greedy top-1 candidate

Generation is the expensive step, so configurations that share generator inputs
share one generation pass: `full` and `no_posg` differ only in how a candidate
is *picked*, so four numbers cost three generation passes per question, not
four. `plan_generation_passes()` makes that grouping explicit and testable.

Metrics per configuration:
    ex          Execution accuracy — the headline number. Targets from the
                plan: >82% (Spider SQL), >60% (MongoDB NoSQL).
    exact_match String equality with gold, after light normalization. Reported
                because it needs no database, but it badly understates
                correctness (a correct query phrased differently scores 0), so
                never quote it as the result.
    n_scored    How many questions EX could actually be computed on. EX is the
                mean over these, NOT over all questions -- see `_ex_eligible`.

Usage:
    harness = EvalHarness(track="sql", config=config)
    report  = harness.evaluate(entries, ablations=["full", "no_posg"])
    print(report["summary"])
"""

from __future__ import annotations

import json
from typing import Any, Optional

ABLATIONS = ["full", "no_schema_linker", "no_sar", "no_posg"]

# Plan targets (CodeGen_Plan_v6_DualTrack.md, Phase 18).
EX_TARGET = {"sql": 0.82, "nosql": 0.60}


def _norm_sql(s: str) -> str:
    return " ".join(s.strip().rstrip(";").lower().split())


def plan_generation_passes(ablations: list) -> dict:
    """
    Group ablations by the generator inputs they need.

    Returns {(use_schema_linker, use_sar): [ablation, ...]} -- one generation
    pass per key. `full` and `no_posg` land in the same group because they feed
    the generator identically and diverge only at selection time.
    """
    groups: dict = {}
    for ab in ablations:
        if ab not in ABLATIONS:
            raise ValueError(f"unknown ablation {ab!r}; expected one of {ABLATIONS}")
        key = (ab != "no_schema_linker", ab != "no_sar")
        groups.setdefault(key, []).append(ab)
    return groups


# ---------------------------------------------------------------------------
# Track-specific scoring (execution + gold comparison)
# ---------------------------------------------------------------------------

class _SqlScorer:
    """Wraps the router's SQL adapter with gold comparison."""
    track = "sql"
    gold_key = "sql"

    def __init__(self, config: dict):
        from src.router.langgraph_router import _SqlTrack
        self.adapter = _SqlTrack(config)

    def gold_of(self, entry: dict):
        return entry["sql"]

    def compare(self, pred, gold, db_name: str) -> dict:
        """
        Execute pred and gold, returning
        {ex, exact_match, ex_eligible, pred_error}.

        ex_eligible is False when *gold* fails to execute -- almost always a
        missing local .sqlite rather than a bad label. Scoring those as wrong
        would silently deflate EX by however many databases happen to be absent,
        so they are excluded from the denominator and counted separately.
        """
        import os

        from src.eval.exec_eval import exec_sql, has_order_by, result_eq

        out = {"exact_match": _norm_sql(pred) == _norm_sql(gold),
               "ex": None, "ex_eligible": False, "pred_error": None}

        db_path = self.adapter.db_path(db_name)
        if not os.path.exists(db_path):
            out["pred_error"] = "no local database"
            return out

        gold_res = exec_sql(db_path, gold)
        if gold_res is None:
            out["pred_error"] = "gold query did not execute"
            return out

        out["ex_eligible"] = True
        pred_res = exec_sql(db_path, pred)
        if pred_res is None:
            out["ex"] = 0.0
            out["pred_error"] = "prediction did not execute"
            return out

        out["ex"] = 1.0 if result_eq(pred_res, gold_res,
                                     order_matters=has_order_by(gold)) else 0.0
        return out

    def close(self):
        self.adapter.close()


class _NoSqlScorer:
    """Wraps the router's NoSQL adapter with gold comparison."""
    track = "nosql"
    gold_key = "mql"

    def __init__(self, config: dict):
        from src.router.langgraph_router import _NoSqlTrack
        self.adapter = _NoSqlTrack(config)

    def gold_of(self, entry: dict):
        if "mql" in entry:
            return entry["mql"]
        return {"collection": entry.get("mql_collection", ""),
                "pipeline":   entry.get("mql_pipeline", [])}

    def compare(self, pred, gold, db_name: str) -> dict:
        from src.pipeline_nosql import _mql_result_eq

        out = {"exact_match": pred == gold,
               "ex": None, "ex_eligible": False, "pred_error": None}

        gold_res, gold_err = self.adapter.execute(gold, db_name)
        if gold_err is not None:
            out["pred_error"] = f"gold query errored: {gold_err}"
            return out
        # A gold pipeline returning [] almost always means this database was
        # never loaded into Mongo (aggregate() does not raise on a missing
        # collection) rather than a genuinely empty answer. An empty-vs-empty
        # comparison would be a false EX hit, so exclude it -- same rule as
        # src/pipeline_nosql.py's gold_row_count guard.
        if not gold_res:
            out["pred_error"] = "gold query returned 0 rows (database likely not loaded)"
            return out

        out["ex_eligible"] = True
        pred_res, pred_err = self.adapter.execute(pred, db_name)
        if pred_err is not None:
            out["ex"] = 0.0
            out["pred_error"] = pred_err
            return out

        out["ex"] = 1.0 if _mql_result_eq(pred_res, gold_res) else 0.0
        return out

    def close(self):
        self.adapter.close()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class EvalHarness:
    """
    Ablation-aware evaluation over a list of prepared entries.

    Entries are the dicts produced by scripts/build_dev_eval_set.py (SQL) or the
    CoT data files: question, db_name, schema, key_fields, and a gold query.

    Args:
        track:     "sql" or "nosql".
        config:    Loaded configs/config.yaml dict.
        strategy:  POSG tie-break strategy.
        seed:      Pins the Generator's candidate sampling. Ablation deltas are
                   differences between separate generation passes, so without a
                   seed the reported gap between `full` and e.g. `no_sar` mixes
                   the component's real effect with sampling noise at
                   temperature 0.8. Defaults to 0 for that reason -- pass None
                   only if you deliberately want unseeded sampling. Note this
                   reduces run-to-run variance but does not eliminate it; see
                   docs/phase15_posg_findings.md on GPU nondeterminism.
        linker, sar, generator:
                   Pre-built components; anything None is built lazily. `linker`
                   is only ever used for entries with no cached key_fields. An
                   injected generator is used as-is, so `seed` does not apply.
        verbose:   Per-question progress.
    """

    def __init__(self, track: str, config: dict, strategy: str = "balanced",
                 seed: int | None = 0,
                 linker=None, sar=None, generator=None, verbose: bool = True):
        if track not in ("sql", "nosql"):
            raise ValueError(f"track must be 'sql' or 'nosql', got {track!r}")

        self.track    = track
        self.config   = config
        self.strategy = strategy
        self.seed     = seed
        self.verbose  = verbose

        self.scorer     = _SqlScorer(config) if track == "sql" else _NoSqlScorer(config)
        self._linker    = linker
        self._sar       = sar
        self._generator = generator

    # --- lazy components ---

    @property
    def linker(self):
        if self._linker is None:
            from src.schema_linker.linker import get_schema_linker
            self._linker = get_schema_linker(self.config["schema_linker"], track=self.track)
        return self._linker

    @property
    def sar(self):
        if self._sar is None:
            from src.sar.infer import get_sar_retriever
            self._sar = get_sar_retriever(self.config["sar"], track=self.track)
        return self._sar

    @property
    def generator(self):
        if self._generator is None:
            from src.generator.infer import GeneratorInfer
            self._generator = GeneratorInfer(
                checkpoint_path=self.scorer.adapter.checkpoint(),
                track=self.track,
                n_candidates=self.config["generator"]["n_candidates"],
                temperature=self.config["generator"]["temperature"],
                seed=self.seed,
            )
        return self._generator

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # --- per-question ---

    def _inputs_for(self, entry: dict) -> tuple:
        """(schema, key_fields, sar_examples) for one entry, cached values preferred."""
        schema = entry.get("schema")
        if not schema:
            schema = self.scorer.adapter.format_schema(entry["db_name"])

        # Prefer key_fields already in the file. build_dev_eval_set.py wrote
        # them with the *real* SchemaLinker, so reusing them keeps runs
        # comparable and avoids re-paying the API cost on every ablation sweep.
        key_fields = entry.get("key_fields")
        if key_fields is None:
            key_fields = self.linker.link(question=entry["question"], schema=schema)

        retrieved = self.sar.retrieve(entry["question"], top_k=4)
        sar_examples = [ex for ex in retrieved
                        if ex["question"].strip() != entry["question"].strip()][:3]
        return schema, key_fields, sar_examples

    def _run_question(self, entry: dict, ablations: list) -> dict:
        schema, key_fields, sar_examples = self._inputs_for(entry)
        db_name = entry["db_name"]
        gold    = self.scorer.gold_of(entry)

        per_ablation: dict = {}
        for (use_sl, use_sar), group in plan_generation_passes(ablations).items():
            kf  = key_fields   if use_sl  else []
            ex  = sar_examples if use_sar else []

            raw = self.generator.generate(entry["question"], schema, kf, ex)
            if not raw:
                raise RuntimeError(
                    f"Generator.generate() returned {raw!r} for "
                    f"question={entry['question']!r} db_name={db_name!r}"
                )

            ranked = None
            for ab in group:
                if ab == "no_posg":
                    # Greedy top-1: the first sampled candidate, with no Pareto
                    # selection applied. Parsed but deliberately not ranked.
                    pred = self.scorer.adapter.parse_candidates(raw)[0]
                else:
                    if ranked is None:
                        ranked = self.scorer.adapter.rank_candidates(
                            raw, kf, ex, db_name, self.strategy)
                    pred = ranked[0] if ranked else self.scorer.adapter.parse_candidates(raw)[0]

                scored = self.scorer.compare(pred, gold, db_name)
                scored["query"] = pred
                scored["n_candidates"] = len(raw)
                per_ablation[ab] = scored

        return {
            "question":   entry["question"],
            "db_name":    db_name,
            "gold":       gold,
            "key_fields": key_fields,
            "n_sar_examples": len(sar_examples),
            "ablations":  per_ablation,
        }

    # --- public ---

    def evaluate(self, entries: list, ablations: Optional[list] = None) -> dict:
        """
        Run every entry through every ablation.

        Returns {"track", "n_questions", "results": [...], "summary": {ablation: {...}}}.
        """
        ablations = ablations or ["full"]
        for ab in ablations:
            if ab not in ABLATIONS:
                raise ValueError(f"unknown ablation {ab!r}; expected one of {ABLATIONS}")

        results = []
        for i, entry in enumerate(entries, 1):
            self._log(f"[{i}/{len(entries)}] {entry['question']}  (db={entry['db_name']})")
            r = self._run_question(entry, ablations)
            for ab in ablations:
                s = r["ablations"][ab]
                self._log(f"    {ab:17} ex={s['ex']}  exact={s['exact_match']}"
                          + (f"  [{s['pred_error']}]" if s["pred_error"] else ""))
            results.append(r)

        return {
            "track":       self.track,
            "strategy":    self.strategy,
            "seed":        self.seed,
            "n_questions": len(results),
            "ablations":   ablations,
            "results":     results,
            "summary":     summarize(results, ablations, self.track),
        }

    def close(self):
        self.scorer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(results: list, ablations: list, track: str) -> dict:
    """Aggregate per-question results into per-ablation metrics."""
    summary = {}
    for ab in ablations:
        scored   = [r["ablations"][ab] for r in results if ab in r["ablations"]]
        eligible = [s for s in scored if s["ex_eligible"]]
        n_ex     = len(eligible)
        summary[ab] = {
            # EX is the mean over questions where gold executed, not over all
            # questions -- quoting it against the wrong denominator is the
            # easiest way to publish a wrong number here.
            "ex":            (sum(s["ex"] for s in eligible) / n_ex) if n_ex else None,
            "n_scored":      n_ex,
            "n_total":       len(scored),
            "n_unscorable":  len(scored) - n_ex,
            "exact_match":   (sum(s["exact_match"] for s in scored) / len(scored)) if scored else None,
            "n_pred_failed": sum(1 for s in eligible if s["ex"] == 0.0 and s["pred_error"]),
        }
    return summary


def format_summary(report: dict) -> str:
    """Human-readable ablation table, with the plan's EX target called out."""
    track   = report["track"]
    summary = report["summary"]
    target  = EX_TARGET.get(track)

    lines = [
        "",
        f"=== Phase 18 evaluation — track={track}  "
        f"n={report['n_questions']}  strategy={report['strategy']} ===",
        "",
        f"{'configuration':<18} {'EX':>8} {'scored':>8} {'exact':>8}",
        "-" * 46,
    ]
    for ab in report["ablations"]:
        s  = summary[ab]
        ex = f"{s['ex']:.1%}" if s["ex"] is not None else "n/a"
        em = f"{s['exact_match']:.1%}" if s["exact_match"] is not None else "n/a"
        lines.append(f"{ab:<18} {ex:>8} {s['n_scored']:>4}/{s['n_total']:<3} {em:>8}")

    full = summary.get("full")
    if full and full["ex"] is not None and target is not None:
        verdict = "MEETS" if full["ex"] >= target else "below"
        lines += ["", f"Plan target for {track}: EX > {target:.0%} — full pipeline "
                      f"{verdict} it ({full['ex']:.1%})."]

    # Ablation deltas only mean something relative to `full`.
    if full and full["ex"] is not None:
        deltas = [(ab, summary[ab]["ex"]) for ab in report["ablations"]
                  if ab != "full" and summary[ab]["ex"] is not None]
        if deltas:
            lines += ["", "Ablation deltas vs full (negative = component helps):"]
            for ab, ex in deltas:
                lines.append(f"  {ab:<18} {ex - full['ex']:+.1%}")

    unscorable = max((summary[ab]["n_unscorable"] for ab in report["ablations"]), default=0)
    if unscorable:
        lines += ["", f"[!] {unscorable}/{report['n_questions']} questions were excluded from EX "
                      f"(gold query did not execute — usually a missing local database). "
                      f"EX above is the mean over the remainder."]
    return "\n".join(lines)


def save_report(report: dict, path: str) -> str:
    """Write the full report as JSON (queries included) for later error analysis."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return path
