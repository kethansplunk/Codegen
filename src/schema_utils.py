"""
Query-time schema utilities.
Adapted from SchemaRAG function.py:
- extract_db_samples_enriched_bm25 uses the QUESTION as the BM25 query (vs our
  build-time PromptSchema which uses the column name).  At inference time this
  produces more question-relevant sample values in the prompt.
- UTF-8 error handling added (matches our mongodb_converter.py convention).
- evidence parameter kept for future BIRD support but defaults to "".
"""

from __future__ import annotations

import logging
import random
import sqlite3
from typing import Any, Dict, List, Union

import bm25s
from nltk.tokenize import word_tokenize


def get_schema_dict(db_path: str) -> Dict[str, Dict[str, str]]:
    """Return {table: {col: type}} from a SQLite file."""
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema: Dict[str, Dict[str, str]] = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info(`{table}`);")
        schema[table] = {col[1]: col[2] for col in cursor.fetchall()}
    conn.close()
    return schema


def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all") -> Any:
    """Execute SQL and return results."""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cursor = conn.cursor()
        cursor.execute(sql)
        if fetch == "all":
            return cursor.fetchall()
        elif fetch == "one":
            return cursor.fetchone()
        elif fetch == "random":
            samples = cursor.fetchmany(10)
            return random.choice(samples) if samples else []
        elif isinstance(fetch, int):
            return cursor.fetchmany(fetch)
        conn.close()
    except Exception as e:
        logging.error(f"execute_sql error: {e}  SQL: {sql}")
        return []


def extract_db_samples_enriched_bm25(
    question: str,
    db_path: str,
    schema_dict: Dict[str, List[str]],
    evidence: str = "",
    sample_limit: int = 2,
) -> str:
    """
    For each column in schema_dict, retrieve the `sample_limit` most
    question-relevant DISTINCT values using BM25.

    Adapted from SchemaRAG function.py:
    - query = question + evidence  (question-aware, not column-name-aware)
    - corpus prefix: "{table} {col} {val}" for better BM25 context
    - long-value guard: avg len > 600 → keep only 1 value
    - NULL tracking preserved

    Args:
        question:     Natural language question.
        db_path:      Path to the SQLite database file.
        schema_dict:  {table: [column, ...]} or {table: {col: type}}.
        evidence:     Optional hint text (BIRD-style). Leave "" for Spider.
        sample_limit: Number of sample values to keep per column.

    Returns:
        Multi-line string of sample values, one section per table.
    """
    query_text = (question + " " + evidence).replace('"', "").replace("'", "").replace("`", "")
    tokenized_query = word_tokenize(query_text)
    query_str       = " ".join(tokenized_query)

    output_lines = ["\n"]

    for table, cols in schema_dict.items():
        output_lines.append(f"## {table} table samples:")
        col_iter = cols if isinstance(cols, list) else list(cols.keys())

        for col in col_iter:
            try:
                rows = execute_sql(
                    db_path,
                    f"SELECT DISTINCT `{col}` FROM `{table}`",
                )
                values = [
                    str(r[0]) if r and r[0] is not None else "NULL"
                    for r in rows
                ]
                is_null_present = "NULL" in values

                if not values:
                    continue

                avg_len = sum(len(v) for v in values) / len(values)
                if avg_len > 600:
                    values = [values[0]]

                if len(values) > sample_limit:
                    corpus  = [f"{table} {col} {v}" for v in values]
                    tokens  = bm25s.tokenize(corpus, stopwords="en")
                    retriever = bm25s.BM25()
                    retriever.index(tokens)
                    q_tokens = bm25s.tokenize(query_str, stopwords="en")
                    results, _ = retriever.retrieve(q_tokens, corpus=values, k=sample_limit)
                    values = list(results[0])
                    if is_null_present:
                        values.append("NULL")

                output_lines.append(
                    f"# Example values for '{table}'.'{col}' column: {values}"
                )
            except Exception as e:
                logging.error(f"extract_db_samples_enriched_bm25: {e}  col={table}.{col}")

    return "\n".join(output_lines)
