"""
Phase 20A -- Streamlit demo.

Session-based routing (Option A, docs/architecture.md): pick a track once in
the sidebar, then ask questions against it. Wraps src/router/langgraph_router.py
directly -- same Router class scripts/run_router.py uses, so this is the same
pipeline, just with a UI instead of a CLI.

Requires (all gitignored, produced on Colab -- copy them in before running):
  - models/generator_{sql,nosql}/  (LoRA adapter, Phase 14A/14B)
  - models/sar_{sql,nosql}/sar_model.pt  (Phase 12A/12B)
  - Data/Spider/database/  for the SQL track to execute (not just select) a query
  - DEEPSEEK_API_KEY in .env  (SchemaLinker API mode, configs/config.yaml)
  - a live mongod on localhost:27017 for the NoSQL track

    pip install streamlit
    streamlit run app.py
"""

from __future__ import annotations

import glob
import json
import os
import time

import streamlit as st
import yaml

from src.router.langgraph_router import Router

st.set_page_config(page_title="CodeGen — NL to Query", page_icon="🗄️", layout="wide")


@st.cache_data
def _load_config() -> dict:
    with open("configs/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def _spider_db_names(db_dir: str) -> list[str]:
    return sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(db_dir, "*"))
        if os.path.isdir(p)
    )


@st.cache_resource(show_spinner=False)
def _get_router(track: str, strategy: str, max_retries: int, n_candidates: int) -> Router:
    # Cached per (track, strategy, max_retries, n_candidates) so the 7B generator
    # and SAR encoder load exactly once per session, matching the Router
    # docstring's "built once, reused every question" design.
    #
    # n_candidates overrides configs/config.yaml's generator.n_candidates (5,
    # tuned for Phase 18's offline POSG eval fidelity) down to the Phase 20A
    # plan's demo target of k=1 by default -- every extra candidate is another
    # parallel sequence the 7B model has to decode, and a failed first attempt
    # can trigger a second full generation pass via self-correction, so 5
    # candidates routinely blows past the <8s demo latency target. The override
    # is applied to an in-memory copy only, so configs/config.yaml on disk (and
    # scripts/run_eval.py / run_baseline.py, which read it directly) are untouched.
    config = _load_config()
    config = {**config, "generator": {**config["generator"], "n_candidates": n_candidates}}
    router = Router(track=track, config=config, strategy=strategy,
                     max_retries=max_retries, seed=0, verbose=False)

    # linker/sar/generator are lazy @property on Router -- left alone, they'd
    # build on the *first* router.run() call instead of here, so SAR's corpus
    # encoding (~30s) and the Generator's 7B checkpoint load (slow over a
    # Drive-mounted symlink) would get counted as that question's latency
    # instead of one-time warm-up. Touching them now keeps the "Loading
    # pipeline" spinner honest and the reported per-query latency real.
    router.linker
    router.sar
    router.generator
    return router


def _render_rows(rows: list | None):
    if rows is None:
        st.info("Not executed — no local database for this db_name "
                "(missing .sqlite, or no live mongod).")
        return
    if not rows:
        st.write("(0 rows)")
        return
    try:
        st.dataframe(rows, use_container_width=True)
    except Exception:
        st.write(rows)


def main():
    st.title("CodeGen — Natural Language to Query")
    st.caption("SchemaRAG (SQL) / SMART-TEND (NoSQL), routed by LangGraph — Phase 20A demo")

    config = _load_config()

    with st.sidebar:
        st.header("Session")
        track = st.radio("Track", ["sql", "nosql"], horizontal=True)
        strategy = st.selectbox(
            "POSG strategy", ["balanced", "schema_priority", "example_priority"])
        max_retries = st.slider("Max retries", 0, 5, 3)
        n_candidates = st.slider(
            "Candidates (k)", 1, 5, 1,
            help="Generator candidates per attempt. Phase 20A's demo target is "
                 "k=1 (<8s latency); config.yaml's eval default is 5, which is "
                 "accurate but noticeably slower, especially if self-correction "
                 "triggers a second full generation pass.")

        if track == "sql":
            db_dir = config["dataset"]["db_path"]
            names = _spider_db_names(db_dir)
            if names:
                db_name = st.selectbox("Database", names)
            else:
                db_name = st.text_input("Database (db_name)")
                st.warning(f"No databases found under {db_dir} — the query will "
                           f"still be generated but not executed.")
        else:
            db_name = st.text_input(
                "Database (db_name)",
                help="Mongo database name; needs a live mongod on localhost:27017.")

        st.divider()
        st.caption("Needs models/ + indexes/ checkpoints (gitignored, from the Colab "
                   "run) and DEEPSEEK_API_KEY in .env for SchemaLinker API mode.")

    question = st.text_area(
        "Question", placeholder="e.g. How many singers are there?", height=80)
    run = st.button("Run", type="primary", disabled=not question or not db_name)

    if not run:
        return

    try:
        with st.spinner(f"Loading {track} pipeline (first run only) ..."):
            router = _get_router(track, strategy, max_retries, n_candidates)
    except Exception as e:
        st.error(f"Failed to build the {track} pipeline: {e}")
        st.info("Most likely a missing checkpoint under models/ or indexes/, or a "
                "missing DEEPSEEK_API_KEY — see this app's docstring / CLAUDE.md's "
                "Environment Notes.")
        return

    start = time.perf_counter()
    try:
        with st.spinner("Routing ..."):
            result = router.run(question, db_name)
    except Exception as e:
        st.error(f"Run failed: {e}")
        return
    elapsed = time.perf_counter() - start

    status = result["status"]
    cols = st.columns(3)
    cols[0].metric("Status", "ok" if status == "ok" else "failed")
    cols[1].metric("Retries", result["retries"])
    cols[2].metric("Latency", f"{elapsed:.1f}s")

    st.subheader("Query")
    query = result["query"]
    if isinstance(query, str):
        st.code(query, language="sql" if track == "sql" else "text")
    else:
        st.code(json.dumps(query, indent=2, default=str), language="json")

    if status == "ok":
        st.subheader("Rows")
        _render_rows(result["rows"])
    else:
        st.subheader("Error")
        st.error(result["error"])

    if len(result["history"]) > 1:
        with st.expander(f"Retry history ({len(result['history'])} attempts)"):
            for h in result["history"]:
                mark = "ok" if h["error"] is None else f"error: {h['error']}"
                st.markdown(f"**[{h['attempt']}]** ({h['source']}) — {mark}")
                st.code(h["query"] if isinstance(h["query"], str) else str(h["query"]))


if __name__ == "__main__":
    main()
