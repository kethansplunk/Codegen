"""
Pareto-Optimal SQL Generator (POSG) — NoSQL / MQL track.
Adapted from SchemaRAG po.py (SQL version) for MongoDB MQL:
- Executability: runs aggregation pipeline on local MongoDB.
- Schema conformity: checks that collections used in pipeline exist in the schema.
- Example consistency: structural similarity based on pipeline stage types
  (no AST available for MQL — use stage-type comparison instead).
- Pareto front + tie-breaking strategy preserved from SQL version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from pymongo import MongoClient


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MQLCandidate:
    collection: str
    pipeline:   list
    index:      int


@dataclass
class EvalScore:
    executability:       float
    schema_conformity:   float
    example_consistency: float


# ---------------------------------------------------------------------------
# Stage-type similarity (replaces AST edit distance for MQL)
# ---------------------------------------------------------------------------

def pipeline_stage_types(pipeline: list) -> List[str]:
    return [list(stage.keys())[0] for stage in pipeline if stage]


def stage_similarity(p1: list, p2: list) -> float:
    t1, t2 = pipeline_stage_types(p1), pipeline_stage_types(p2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    common = len(set(t1) & set(t2))
    return common / max(len(set(t1) | set(t2)), 1)


# ---------------------------------------------------------------------------
# Pareto-Optimal MQL selector
# ---------------------------------------------------------------------------

class ParetoOptimalMQL:
    def __init__(self, db_name: str, mongo_uri: str = "mongodb://localhost:27017"):
        self.db_name   = db_name
        self.mongo_uri = mongo_uri
        self._client: Optional[MongoClient] = None

    @property
    def db(self):
        if self._client is None:
            self._client = MongoClient(self.mongo_uri)
        return self._client[self.db_name]

    # --- executability ---

    def evaluate_executability(self, collection: str, pipeline: list) -> float:
        try:
            list(self.db[collection].aggregate(pipeline, maxTimeMS=3000))
            return 1.0
        except Exception:
            return 0.0

    # --- schema conformity ---

    def evaluate_schema_conformity(
        self, collection: str, pipeline: list, schema_links: Set[str]
    ) -> float:
        used   = {collection.lower()}
        for stage in pipeline:
            if "$lookup" in stage:
                used.add(stage["$lookup"].get("from", "").lower())
        links_lower = {s.lower() for s in schema_links}
        if not links_lower:
            return 1.0
        inter = used & links_lower
        return len(inter) / len(used | links_lower)

    # --- example consistency ---

    def evaluate_example_consistency(
        self, pipeline: list, examples: List[dict]
    ) -> float:
        if not examples:
            return 0.0
        sims = [stage_similarity(pipeline, ex.get("mql_pipeline", [])) for ex in examples]
        return sum(sims) / len(sims)

    # --- evaluate all ---

    def evaluate_candidates(
        self,
        candidates: List[Dict],          # [{"collection": ..., "pipeline": ...}]
        schema_links: Set[str],
        examples: List[dict],
    ) -> List[Tuple[MQLCandidate, EvalScore]]:
        result = []
        for i, cand in enumerate(candidates):
            col      = cand.get("collection", "")
            pipeline = cand.get("pipeline", [])
            score = EvalScore(
                executability       = self.evaluate_executability(col, pipeline),
                schema_conformity   = self.evaluate_schema_conformity(col, pipeline, schema_links),
                example_consistency = self.evaluate_example_consistency(pipeline, examples),
            )
            result.append((MQLCandidate(collection=col, pipeline=pipeline, index=i), score))
        return result

    # --- pareto front ---

    def find_pareto_optimal(
        self, evaluated: List[Tuple[MQLCandidate, EvalScore]]
    ) -> List[MQLCandidate]:
        executable = [(c, s) for c, s in evaluated if s.executability > 0]
        if not executable:
            return []
        pareto = []
        for i, (ci, si) in enumerate(executable):
            dominated = any(
                j != i
                and sj.schema_conformity   >= si.schema_conformity
                and sj.example_consistency >= si.example_consistency
                and (sj.schema_conformity   > si.schema_conformity
                     or sj.example_consistency > si.example_consistency)
                for j, (_, sj) in enumerate(executable)
            )
            if not dominated:
                pareto.append(ci)
        return pareto

    # --- select ---

    def select_final_mql(
        self,
        candidates: List[Dict],
        schema_links: Set[str],
        examples: List[dict],
        strategy: str = "balanced",
    ) -> Optional[Dict]:
        if not candidates:
            return None
        evaluated = self.evaluate_candidates(candidates, schema_links, examples)
        pareto    = self.find_pareto_optimal(evaluated)

        if not pareto:
            for c, s in evaluated:
                if s.executability > 0:
                    return {"collection": c.collection, "pipeline": c.pipeline}
            return {"collection": candidates[0]["collection"], "pipeline": candidates[0]["pipeline"]}

        if len(pareto) == 1:
            c = pareto[0]
            return {"collection": c.collection, "pipeline": c.pipeline}

        score_map = {c.index: s for c, s in evaluated}
        weights   = {"balanced": (0.5, 0.5), "schema_priority": (0.7, 0.3), "example_priority": (0.3, 0.7)}
        ws, we    = weights.get(strategy, (0.5, 0.5))

        best, best_score = None, -1.0
        for c in pareto:
            s     = score_map[c.index]
            score = ws * s.schema_conformity + we * s.example_consistency
            if score > best_score:
                best_score, best = score, c
        if best:
            return {"collection": best.collection, "pipeline": best.pipeline}
        c = pareto[0]
        return {"collection": c.collection, "pipeline": c.pipeline}
