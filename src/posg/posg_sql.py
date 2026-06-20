"""
Pareto-Optimal SQL Generator (POSG) — SQL track.
Adapted from SchemaRAG po.py:
- ASTProcessor: sqlparse-based AST builder + normalized edit distance.
- ParetoOptimal: scores candidates on 3 dimensions:
    1. Executability (0/1) — runs on SQLite
    2. Schema conformity — Jaccard(SQL identifiers, predicted schema links)
    3. Example consistency — 1 - AST_edit_distance(candidate, retrieved_examples)
- Pareto front selection + tie-breaking strategies preserved.
- validate_sql_statement replaced with direct SQLite execution.
- Hardcoded paths removed; db_path passed as argument.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import sqlparse
from sqlparse.tokens import Keyword, Name, Literal


# ---------------------------------------------------------------------------
# AST processor
# ---------------------------------------------------------------------------

class ASTProcessor:
    @staticmethod
    def parse_sql_to_ast(sql: str) -> Dict:
        if not sql or not sql.strip():
            return {"type": "Empty", "value": "", "tokens": []}
        try:
            stmts = sqlparse.parse(sql)
            if not stmts:
                return {"type": "Empty", "value": "", "tokens": []}
            return ASTProcessor._build_ast(stmts[0])
        except Exception as e:
            return {"type": "Error", "value": str(e), "tokens": []}

    @staticmethod
    def _build_ast(token) -> Dict:
        if token is None:
            return {"type": "None", "value": "", "tokens": []}
        token_type  = type(token).__name__
        token_value = str(token).strip()
        if hasattr(token, "tokens") and token.tokens:
            children = [
                ASTProcessor._build_ast(t)
                for t in token.tokens
                if ASTProcessor._is_meaningful(t)
            ]
            return {"type": token_type, "value": token_value,
                    "ttype": str(token.ttype) if hasattr(token, "ttype") and token.ttype else None,
                    "tokens": children}
        return {"type": token_type, "value": token_value,
                "ttype": str(token.ttype) if hasattr(token, "ttype") and token.ttype else None,
                "tokens": []}

    @staticmethod
    def _is_meaningful(token) -> bool:
        if token is None or not str(token).strip():
            return False
        if hasattr(token, "ttype") and token.ttype in (
            sqlparse.tokens.Whitespace,
            sqlparse.tokens.Whitespace.Newline,
            sqlparse.tokens.Comment.Single,
            sqlparse.tokens.Comment.Multiline,
        ):
            return False
        return True

    @staticmethod
    def _node_weight(node: Dict) -> int:
        return 1 + sum(ASTProcessor._node_weight(c) for c in node.get("tokens", []))

    @staticmethod
    def _nodes_equal(n1: Dict, n2: Dict) -> bool:
        if not n1 or not n2:
            return False
        if n1.get("type") != n2.get("type") or n1.get("ttype") != n2.get("ttype"):
            return False
        if n1.get("type") in ("Keyword", "Name", "Literal"):
            return n1.get("value", "").strip().lower() == n2.get("value", "").strip().lower()
        return True

    @classmethod
    def _seq_edit_dist(cls, seq1: List[Dict], seq2: List[Dict], _depth: int = 0) -> int:
        m, n = len(seq1), len(seq2)
        if _depth > 4:
            return abs(m - n)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                else:
                    n1, n2 = seq1[i - 1], seq2[j - 1]
                    sub    = (cls._seq_edit_dist(n1.get("tokens", []), n2.get("tokens", []), _depth + 1)
                               if cls._nodes_equal(n1, n2) else 2)
                    dp[i][j] = min(dp[i-1][j-1] + sub, dp[i-1][j] + 1, dp[i][j-1] + 1)
        return dp[m][n]

    @classmethod
    def edit_distance(cls, ast1: Dict, ast2: Dict) -> float:
        dist = cls._seq_edit_dist(
            [ast1] if ast1 else [], [ast2] if ast2 else []
        ) if ast1 and ast2 else max(cls._node_weight(ast1 or {}), cls._node_weight(ast2 or {}))
        w = max(cls._node_weight(ast1 or {}), cls._node_weight(ast2 or {}))
        return min(1.0, dist / w) if w else 0.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SQLCandidate:
    sql: str
    index: int


@dataclass
class EvalScore:
    executability:     float
    schema_conformity: float
    example_consistency: float


# ---------------------------------------------------------------------------
# SQL keywords to filter from schema extraction
# ---------------------------------------------------------------------------

_SQL_KW = {
    "select","from","where","join","inner","left","right","full","outer","on","and",
    "or","not","in","exists","like","between","is","null","group","by","order","having",
    "limit","offset","distinct","all","union","intersect","except","case","when","then",
    "else","end","insert","update","delete","create","drop","alter","table","view","index",
    "into","values","set","as","asc","desc","count","sum","avg","min","max","with",
    "recursive","over","partition","window","cast","convert","substring","trim","upper",
    "lower","length","coalesce","nullif","round","floor","ceil","abs","mod","power",
    "sqrt","log","exp","concat","replace",
}


# ---------------------------------------------------------------------------
# Pareto-Optimal selector
# ---------------------------------------------------------------------------

class ParetoOptimal:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path      = db_path
        self.ast_proc     = ASTProcessor()

    # --- executability ---

    def evaluate_executability(self, sql: str) -> float:
        if not self.db_path:
            try:
                sqlparse.parse(sql)
                return 1.0
            except Exception:
                return 0.0
        try:
            conn   = sqlite3.connect(self.db_path)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            cursor = conn.cursor()
            sql_exec = sql.rstrip()
            if sql_exec.endswith(";"):
                sql_exec = sql_exec[:-1].rstrip()
            if not re.search(r'\bLIMIT\b', sql_exec, re.IGNORECASE):
                sql_exec += " LIMIT 1"
            cursor.execute(sql_exec)
            conn.close()
            return 1.0
        except Exception:
            return 0.0

    # --- schema conformity ---

    def _extract_schema_from_sql(self, sql: str) -> Set[str]:
        sql_clean = re.sub(r"'[^']*'", "''", sql)
        sql_clean = re.sub(r'"[^"]*"', '""', sql_clean)
        sql_clean = re.sub(r"`[^`]*`", "``", sql_clean)
        words   = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", sql_clean))
        for dotted in re.findall(r"\b([a-zA-Z][a-zA-Z0-9_]*\.[a-zA-Z][a-zA-Z0-9_]*)\b", sql_clean):
            for p in dotted.split("."):
                words.add(p)
        return {w.lower() for w in words if w.lower() not in _SQL_KW}

    def evaluate_schema_conformity(self, sql: str, schema_links: Set[str]) -> float:
        used = self._extract_schema_from_sql(sql)
        if not used and not schema_links:
            return 1.0
        if not used or not schema_links:
            return 0.0
        inter    = used & schema_links
        jaccard  = len(inter) / len(used | schema_links)
        coverage = len(inter) / len(used)
        return (jaccard + coverage) / 2.0

    # --- example consistency ---

    def evaluate_example_consistency(self, sql: str, examples: List[str]) -> float:
        if not examples:
            return 0.0
        ast1 = self.ast_proc.parse_sql_to_ast(sql)
        sims = [
            max(0.0, 1.0 - self.ast_proc.edit_distance(ast1, self.ast_proc.parse_sql_to_ast(ex)))
            for ex in examples
        ]
        return sum(sims) / len(sims)

    # --- evaluate all ---

    def evaluate_candidates(
        self, candidates: List[str], schema_links: Set[str], examples: List[str]
    ) -> List[Tuple[SQLCandidate, EvalScore]]:
        result = []
        for i, sql in enumerate(candidates):
            score = EvalScore(
                executability       = self.evaluate_executability(sql),
                schema_conformity   = self.evaluate_schema_conformity(sql, schema_links),
                example_consistency = self.evaluate_example_consistency(sql, examples),
            )
            result.append((SQLCandidate(sql=sql, index=i), score))
        return result

    # --- pareto front ---

    def find_pareto_optimal(
        self, evaluated: List[Tuple[SQLCandidate, EvalScore]]
    ) -> List[SQLCandidate]:
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

    def select_final_sql(
        self,
        candidates: List[str],
        schema_links: Set[str],
        examples: List[str],
        strategy: str = "balanced",
    ) -> str:
        if not candidates:
            return ""
        evaluated = self.evaluate_candidates(candidates, schema_links, examples)
        pareto    = self.find_pareto_optimal(evaluated)

        if not pareto:
            for c, s in evaluated:
                if s.executability > 0:
                    return c.sql
            return candidates[0]

        if len(pareto) == 1:
            return pareto[0].sql

        score_map = {c.index: s for c, s in evaluated}
        weights   = {"balanced": (0.5, 0.5), "schema_priority": (0.7, 0.3), "example_priority": (0.3, 0.7)}
        ws, we    = weights.get(strategy, (0.5, 0.5))

        best, best_score = None, -1.0
        for c in pareto:
            s     = score_map[c.index]
            score = ws * s.schema_conformity + we * s.example_consistency
            if score > best_score:
                best_score, best = score, c
        return best.sql if best else pareto[0].sql
