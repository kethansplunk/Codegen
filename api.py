"""
Phase 20B -- FastAPI backend.

Same pipeline app.py (Streamlit, Phase 20A) and scripts/run_router.py drive --
wraps src/router/langgraph_router.py's Router as HTTP endpoints instead of a
UI or CLI, for programmatic/service-to-service use.

    pip install fastapi uvicorn
    uvicorn api:app --reload

Requires the same things app.py's docstring lists: models/ + indexes/
checkpoints (gitignored, from the Colab run), Data/Spider/database/ for the
SQL track to execute (not just generate) a query, DEEPSEEK_API_KEY in .env,
and a live mongod for the NoSQL track.
"""

from __future__ import annotations

import json
from typing import Literal, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.router.langgraph_router import Router

app = FastAPI(title="CodeGen -- NL to Query API", version="1.0")

_config_cache: Optional[dict] = None
_router_cache: dict[tuple, Router] = {}


def _load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        with open("configs/config.yaml", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def _get_router(track: str, strategy: str, max_retries: int, n_candidates: int) -> Router:
    # Cached per (track, strategy, max_retries, n_candidates), same rationale
    # as app.py's _get_router: the 7B Generator and SAR encoder should load
    # once per settings combination, not once per request.
    key = (track, strategy, max_retries, n_candidates)
    if key not in _router_cache:
        config = _load_config()
        config = {**config, "generator": {**config["generator"], "n_candidates": n_candidates}}
        router = Router(track=track, config=config, strategy=strategy,
                         max_retries=max_retries, seed=0, verbose=False)
        # Force SchemaLinker/SAR/Generator to build now rather than on first
        # .run() call -- otherwise that one-time warm-up cost (SAR corpus
        # encoding, Generator checkpoint load) gets counted as the first
        # request's latency instead of this cache-miss path's. Same fix as
        # app.py's _get_router.
        router.linker
        router.sar
        router.generator
        _router_cache[key] = router
    return _router_cache[key]


class QueryRequest(BaseModel):
    track: Literal["sql", "nosql"]
    question: str
    db_name: str
    schema_text: Optional[str] = None
    strategy: Literal["balanced", "schema_priority", "example_priority"] = "balanced"
    max_retries: int = 3
    n_candidates: int = 1


class QueryResponse(BaseModel):
    status: str
    query: str
    rows: Optional[list] = None
    retries: int
    error: Optional[str] = None
    history: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/databases")
def list_databases():
    """SQL track's available db_names, mirroring app.py's sidebar dropdown."""
    import glob
    import os

    db_dir = _load_config()["dataset"]["db_path"]
    names = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(db_dir, "*"))
        if os.path.isdir(p)
    )
    return {"db_dir": db_dir, "databases": names}


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        500: {"description": "Pipeline failed to build (missing checkpoint/API key) "
                              "or the run itself raised an unhandled error."},
    },
)
def run_query(req: QueryRequest):
    try:
        router = _get_router(req.track, req.strategy, req.max_retries, req.n_candidates)
    except Exception as e:
        raise HTTPException(status_code=500,
                             detail=f"failed to build {req.track} pipeline: {e}")

    try:
        result = router.run(req.question, req.db_name, schema=req.schema_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"run failed: {e}")

    query = result["query"]
    query_str = (query if isinstance(query, str)
                 else json.dumps(query, ensure_ascii=False, default=str))

    return QueryResponse(
        status=result["status"],
        query=query_str,
        rows=result["rows"],
        retries=result["retries"],
        error=result["error"],
        history=result["history"],
    )
