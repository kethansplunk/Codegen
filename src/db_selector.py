"""
Database selection from the question alone.

Every entry point so far (Router.run, api.py's /query, app.py's sidebar) requires
the caller to name the database. That is fine for offline eval -- Spider ships a
gold db_name per question -- but it is the wrong shape for a demo, where the
whole point is that the user only has a question.

This picks the database with BM25 over a one-document-per-database corpus built
from Data/prompt_schema/{track}/*.json (Phase 6): the database name, its table
names and its column names. No model, no API call, no GPU -- it is a lexical
match against schema text already on disk, so the index builds in ~0.1s and a
lookup costs ~0.2ms, adding nothing measurable to the demo's latency budget.

Measured on all 1034 gold-labelled Spider dev questions
(Data/cot_data/sql_dev_eval_full.json):

    top-1  80.9%      top-3  91.1%      top-5  95.0%

Stemming is what makes this work at all -- without it top-1 falls to 65.3%,
because "how many singers" never matches a table called `singer`. Sample values
were tried in the corpus and left out: they cost ~1 point of top-1 by drowning
the schema signal in incidental text.

The gap between top-1 and top-5 is why callers get ranked candidates rather
than a bare winner: a wrong database guarantees a wrong answer, so the UI shows
its work and offers a one-click override instead of silently guessing. Reranking
the top-5 with an LLM is the obvious way to close that gap and is not done here
-- it would need a DeepSeek call per question, and is unmeasured.

    from src.db_selector import DBSelector
    sel = DBSelector(track="sql")
    sel.select("How many singers are there?")      # -> "concert_singer"
    sel.rank("How many singers are there?", k=3)   # -> [(db, score), ...]
"""

from __future__ import annotations

import functools
import glob
import json
import logging
import os
import re

import bm25s

try:
    import Stemmer  # PyStemmer
except ImportError:  # pragma: no cover - exercised only on installs missing the dep
    Stemmer = None

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Column names arrive as snake_case or CamelCase and questions never do, so both
# get split into words before indexing ("AirportCode" -> "airport code").
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _words(text: str) -> str:
    return _CAMEL.sub(" ", str(text)).replace("_", " ")


def _db_document(db_name: str, schema: dict) -> str:
    """One BM25 document per database: its name (x2), table names, column names.

    The name is repeated because a question naming the domain outright ("flights
    from Aberdeen") should beat a database that merely happens to share a column
    name. Weights were swept over the full dev set; everything in the 79-81%
    top-1 band was within noise of everything else, so this is the simplest
    configuration that scored at the top of that band rather than a tuned
    optimum worth defending.
    """
    tables, columns = set(), []
    for key in schema:
        table, _, column = key.partition(".")
        tables.add(table)
        columns.append(_words(column))

    return " ".join([
        _words(db_name), _words(db_name),
        " ".join(_words(t) for t in sorted(tables)),
        " ".join(columns),
    ])


class DBSelector:
    """Ranks databases against a natural-language question."""

    def __init__(self, track: str = "sql", schema_dir: str | None = None):
        self.track = track
        self.schema_dir = schema_dir or os.path.join(
            _HERE, "Data", "prompt_schema", track)
        self._stemmer = Stemmer.Stemmer("english") if Stemmer else None
        if self._stemmer is None:
            logging.warning(
                "PyStemmer not installed -- DBSelector top-1 accuracy drops from "
                "~81%% to ~65%%. Install it with: pip install PyStemmer")

        self.db_names, corpus = self._build_corpus()
        if not self.db_names:
            raise FileNotFoundError(
                f"No PromptSchema files under {self.schema_dir} -- "
                f"run Phase 6 (src/prompt_schema.py) first.")
        self._retriever = bm25s.BM25()
        self._retriever.index(self._tokenize(corpus), show_progress=False)

    def _tokenize(self, text):
        return bm25s.tokenize(
            text, stopwords="en", stemmer=self._stemmer, show_progress=False)

    def _build_corpus(self) -> tuple[list[str], list[str]]:
        names, docs = [], []
        for path in sorted(glob.glob(os.path.join(self.schema_dir, "*.json"))):
            db_name = os.path.splitext(os.path.basename(path))[0]
            try:
                with open(path, encoding="utf-8") as f:
                    schema = json.load(f)
            except (OSError, json.JSONDecodeError):
                logging.warning("DBSelector: skipping unreadable schema %s", path)
                continue
            names.append(db_name)
            docs.append(_db_document(db_name, schema))
        return names, docs

    def rank(self, question: str, k: int = 3) -> list[tuple[str, float]]:
        """Top-k (db_name, score) pairs, best first."""
        k = max(1, min(k, len(self.db_names)))
        idx, scores = self._retriever.retrieve(
            self._tokenize(question), k=k, show_progress=False)
        return [(self.db_names[int(i)], float(s))
                for i, s in zip(idx[0], scores[0])]

    def select(self, question: str) -> str:
        """Best-matching db_name."""
        return self.rank(question, k=1)[0][0]


@functools.lru_cache(maxsize=2)
def get_db_selector(track: str = "sql") -> DBSelector:
    """Cached per track -- the index is built once per process."""
    return DBSelector(track=track)
