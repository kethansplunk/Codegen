# CodeGen Project — Claude Code Context

## What This Project Is
An AIML Capstone (Project 6) building a dual-track natural language to query system:
- Track 1: Text-to-SQL (English → PostgreSQL) using SchemaRAG architecture
- Track 2: Text-to-NoSQL (English → MongoDB) using SMART/TEND architecture
- Orchestrated by LangGraph with session-based routing (Option A)

## Current Phase
Phase 19 error analysis is done; Phase 20A (Streamlit demo) is done. **SQL**: 80–81% EX (n=100, held-out `sql_dev_eval_full.json`, Colab A100) — still below the >82% target, up from the original 79–80% after two fixes landed (see below). **NoSQL**: 84.2% EX (n=100, train split — no held-out NoSQL set exists) — meets the >60% target, unaffected by this round of fixes. Full ablation + root-cause breakdown in README.md's "Evaluation + ablation" section. **CP1 baseline has now actually been run**: zero-shot 0.0% EX, few-shot (`--k_shot 3`) 3.0% EX (n=100) — both well below the plan's ~45–55% estimate; diagnosed as codegen-350M being too small/undertrained to produce coherent SQL from a schema-comment-style prompt (it echoes the schema's own `(col:TYPE, examples:...)` annotation syntax back as if it were the answer), not a bug in the harness or prompt. 70 tests green on Mac (`pytest tests/`).

Two fixes landed out of the Phase 19 error analysis:
- `src/generator/sql_fixups.py` deterministically qualifies columns that are ambiguous under a JOIN but were crashing execution outright (safe because the JOIN's `ON` equality already guarantees both sides are equal).
- **`flight_2` database bug** (found via the Phase 20A demo, not the offline error analysis): `Flights.SourceAirport`/`DestAirport` are stored with a leading space that `Airports.AirportCode` and every gold-query literal lack, so any `WHERE`/`JOIN` comparing them silently returned 0/empty instead of erroring — **gold itself** was scoring wrong on 42/1034 held-out questions (4.1%), not just predictions. Scanned all 166 Spider databases; `flight_2` is the only one affected on a column the held-out dev set actually filters/joins on. Fixed via `TRIM()`, applied fresh inside `notebooks/phase18_eval_ablation_res.ipynb` on every run (self-applying, no re-upload needed) — clean copy also linked from `datasets/spider/README.md`.
- Also landed: `src/eval/harness.py` previously scored only the top-ranked POSG candidate (`ranked[0]`); if it failed to execute, the question was marked wrong even when a lower-ranked candidate (already generated in the same batched call, no extra GPU cost) would have executed fine. Added a fallback that walks `ranked[]` on execution failure, mirroring the Router's own retry ladder — checks executability only, never gold, so nothing is answer-leaked into selection.

Deliberately does NOT attempt the other found issues (missing `DISTINCT` after one-to-many joins, ties collapsed by `ORDER BY ... LIMIT 1`, occasional hallucination, a handful of likely Spider gold-label quirks) since those require judging query intent and risk regressing currently-correct queries. Of the 4 missing-`DISTINCT` cases found, only 2 share a narrow, provably-safe shape (bare `SELECT`, no aggregate, joining a "one" table to a "many" table, selecting only "one"-side columns) — scoped but deliberately left unbuilt this round. Full failure-category breakdown with counts is in README.md's "Evaluation + ablation" section.

## Key Reference Documents (read these for full context)
- Project plan (latest): `CodeGen_Plan_v6_DualTrack.md` (20-phase plan, project root)
- Architecture deep dive: `docs/architecture.md`
- Project proposal: `docs/CodeGen_Project_Proposal.docx`
- SchemaRAG paper: `docs/SchemaRAG.pdf`
- TEND paper: `docs/Text_to_Nosql.pdf`
- README.md — authoritative phase-by-phase status table; keep this file in sync with it

## Tech Stack
- Python 3.10, conda env: `text2sql`
- PyTorch with MPS backend (Mac M1 — no CUDA) for local work; Google Colab A100 for generator/SAR training
- Models: DeepSeek API (SchemaLinker, active) / Qwen3-8B (SchemaLinker, switchable to trained mode), Qwen2.5-Coder-7B-Instruct (Generator, LoRA, SQL + NoSQL both trained)
- Vector store: ChromaDB (persistent indexes, Phase 13) | Orchestration: LangGraph | DB: SQLite (train) / MongoDB, PostgreSQL (prod)

## Project Structure
- `src/` — reusable library code (schema_linker/, sar/, generator/, posg/, eval/, router/)
- `scripts/` — data pipeline + Colab training driver scripts
- `notebooks/` — Colab training notebooks (Phase 9A, 12A/B, 13, 14A/B)
- `Data/` — datasets: Spider, fk_graphs, mongodb, prompt_schema, rag_corpus, cot_data, generator_data (gitignored where large)
- `datasets/spider/README.md` — pointer to the `flight_2`-fixed Spider database zip on Drive (the actual data stays gitignored; this is git-tracked so the fix is discoverable)
- `configs/config.yaml` — SchemaLinker mode (api/model) + SAR backend (chroma/memory) switches
- `docs/` — architecture doc + reference papers
- `models/`, `indexes/`, `external/SchemaRAG/` — checkpoints, ChromaDB indexes, reference implementation (gitignored)
- `app.py` — Phase 20A Streamlit demo (wraps `src/router/langgraph_router.py`'s `Router`); `notebooks/phase20a_streamlit_demo.ipynb` launches it on Colab via a `cloudflared` tunnel

## Architecture (SchemaRAG-based)
BM25S PromptSchema → SchemaLinker (DeepSeek API / Qwen3-8B, 3-stage, currently API mode) → SAR (bge-large + Transformer, both tracks trained)
→ Qwen2.5-Coder-7B Generator (both tracks trained) → POSG (Pareto selection, code ready, not yet wired) → Execution

## Completed (see README.md for full table)
- Phases 1–3: env setup, PyTorch MPS, SchemaRAG cloned, deps installed
- Phase 4: Spider dataset validated (7000 Q-SQL pairs, 166 SQLite DBs)
- Phase 5A/5B: FK graph builder + MongoDB converter (all 166 DBs)
- Phase 6: PromptSchema — BM25S column annotations (SQL + NoSQL)
- Phase 7A/7B: RAG corpus builders — SQL (7000 entries) + NoSQL (5697 entries, MongoDB-verified)
- Phase 8A/8B: CoT training data via DeepSeek — SQL (6000+) + NoSQL (5697)
- Phase 9A/9B, 10A/10B, 11A/11B: SchemaLinker SFT/MTL/GRPO — ⏸ deferred, using DeepSeek API instead
- Phase 12A/12B: SAR training — SQL + NoSQL, dual cross-attention, loss 0.15→0.02
- Phase 13: ChromaDB persistent indexes built (SQL + NoSQL)
- Phase 14A: SQL Generator LoRA SFT (6748 examples, A100)
- Phase 14B: NoSQL Generator LoRA SFT (5410 examples, warm-started from 14A)
- Phase 15A/15B: POSG wired for both tracks (`scripts/run_posg_*.py`)
- Phase 16: pipeline extracted to `src/pipeline_sql.py` / `src/pipeline_nosql.py` (`run_pipeline()` accepts injected linker/sar/generator)
- Phase 17: LangGraph router + self-correction (`src/router/langgraph_router.py`, `scripts/run_router.py`, `tests/test_router.py`)
- Phase 18: eval harness + ablation, results produced (`src/eval/harness.py`, `scripts/run_eval.py`, `scripts/run_baseline.py`, `notebooks/phase18_eval_ablation_res.ipynb`) — SQL 80–81% EX (below >82% target, up from 79–80% after the `flight_2` + harness-fallback fixes), NoSQL 84.2% EX (meets >60% target). Ambiguous-column fixup (`src/generator/sql_fixups.py`) landed as a result of the error analysis. CP1 baseline actually run: 3.0% EX few-shot (below the ~45–55% plan estimate, diagnosed as a codegen-350M capacity limitation).
- Phase 19: error analysis done. 19/100 misses in the latest run categorized by root cause (see README's "Evaluation + ablation" section for the full breakdown with counts) — biggest actionable category was missing `DISTINCT` after one-to-many JOINs (4 cases, only 2 safely fixable), followed by hallucinated set-op chains, `ORDER BY...LIMIT 1` ties, wrong-column schema-linking errors, genuine ANY/ALL comprehension confusion, and several likely Spider gold-label quality issues.
- Phase 20A: Streamlit demo (`app.py`) built and verified end-to-end (SQL track, k=1 candidates per the plan's <8s latency target — confirmed 2.7s for a real question). Launches on Colab via `notebooks/phase20a_streamlit_demo.ipynb` + a `cloudflared` quick tunnel.

## Testing
`pytest tests/` — 70 tests, all run on Mac with no GPU and no checkpoints (SchemaLinker/SAR/Generator stubbed; real LangGraph graph, real POSG, real temp SQLite DBs, real regex-based SQL fixup logic). Needs `pytest`, `sqlparse`, `langgraph` installed.

## Known Gaps
- SAR schema fusion gap (Phase 12): SAR never fuses real table/column data into retrieval. **Closed for SQL** — reconfirmed on Phase 18's SQL ablation (0% EX delta from removing SAR, across 3 separate n=100 runs); not worth fixing there. **Open/load-bearing for NoSQL** — Phase 18's NoSQL ablation found the opposite: removing SAR costs 20 points of EX (84.2% → 64.2%), the largest ablation effect in either track. Don't generalize the SQL "not worth fixing" conclusion to NoSQL. See memory note `sar_schema_fusion_gap` for the full evidence trail.
- SchemaLinker model-mode training (9A/9B/10/11) is deferred indefinitely in favor of the DeepSeek API; only switch back if API cost/latency becomes a blocker.
- SQL Generator failure patterns identified but not fixed (require judging query intent, not mechanical): missing `DISTINCT` after one-to-many JOINs, ties collapsed by `ORDER BY ... LIMIT 1`, occasional hallucinated `EXCEPT`/`INTERSECT` chains, a few likely Spider gold-label quirks. See README's "Evaluation + ablation" section for the full Phase 19 breakdown with per-category counts.
- CP1 baseline result is poor (3.0% EX few-shot, 0.0% zero-shot) — not a bug, codegen-350M genuinely can't produce coherent SQL from this prompt style at this scale. No further baseline work planned; the finding itself is the deliverable.
- SQL Generator's remaining `DISTINCT`-after-JOIN gap is only partially mechanically fixable: 2 of 4 found cases share a narrow, provably-safe shape (bare `SELECT`, no aggregate, one-to-many JOIN, only "one"-side columns selected — needs `Data/fk_graphs/` to know which side is which). Scoped in this session's Phase 19 analysis but deliberately not built.

## Environment Notes
- `conda activate text2sql` before any Python work
- PyTorch backend: MPS (not CUDA) on Mac; use Colab A100 for generator/SAR training
- pip installs outside conda need `--break-system-packages`
- `DEEPSEEK_API_KEY` must be set in `.env` at project root (SchemaLinker API mode, CoT data gen)
