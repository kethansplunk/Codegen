# CodeGen — Natural Language to Query Generator

A dual-track system that translates natural language questions into SQL (PostgreSQL) and NoSQL (MongoDB MQL) queries using fine-tuned LLMs.

## What it does

Given a natural language question and a database, the system produces the correct query by routing through a multi-stage pipeline:

1. **PromptSchema** — enriches the schema with sample values per column so the LLM understands what each column contains
2. **SchemaLinker** — identifies the relevant tables and columns from the schema (3-stage: CoT SFT → MTL → GRPO)
3. **SAR (Schema-Aware Retriever)** — retrieves structurally similar past examples using a dual cross-attention model trained on structural type similarity
4. **Generator** — produces the final query using a fine-tuned Qwen2.5-Coder-7B model
5. **POSG** — generates 5 candidates and selects the best one via Pareto-optimal scoring (executability + schema conformity + structural distance); wired end-to-end and validated on both tracks (Phase 15)

## Models

| Component | Base Model | Status |
|---|---|---|
| SchemaLinker | DeepSeek API (primary) / Qwen/Qwen3-8B (switchable) | API active |
| SAR encoder | BAAI/bge-large-en-v1.5 + SchemaAwareModel (~16M params) | ✅ Trained |
| Query Generator | Qwen/Qwen2.5-Coder-7B-Instruct (LoRA, SQL + NoSQL) | ✅ Both trained |
| CoT teacher | DeepSeek-V3 (API) | Used for data gen |

## Current status

| Phase | Description | Status |
|---|---|---|
| 1–3 | Planning, architecture, environment setup | ✅ Done |
| 4 | Spider dataset — 7000 Q-SQL pairs + 166 SQLite databases | ✅ Done |
| 5A | FK graph builder — NetworkX graphs for all 166 databases | ✅ Done |
| 5B | MongoDB converter — all 166 databases converted and verified | ✅ Done |
| 6 | PromptSchema — BM25S column annotations for SQL and NoSQL | ✅ Done |
| 7A | SQL RAG corpus — 7000 Q-SQL pairs with 57 structural types (7-dim) | ✅ Done |
| 7B | NoSQL RAG corpus — 5697 Q-MQL pairs verified against MongoDB | ✅ Done |
| 8A | SQL CoT data — 6000+ CoT examples generated and validated | ✅ Done |
| 8B | NoSQL CoT data — 5697-entry MQL CoT dataset via DeepSeek API | ✅ Done |
| 9A/9B | SchemaLinker SQL/NoSQL SFT | ⏸ Deferred — using DeepSeek API (switchable via config) |
| 10A/10B | Error mining for MTL | ⏸ Deferred — linked to SchemaLinker training |
| 11A/11B | SchemaLinker MTL + GRPO | ⏸ Deferred — linked to SchemaLinker training |
| 12A | SAR SQL training — 7000 entries, 57 types, loss 0.15 → 0.02 | ✅ Done |
| 12B | SAR NoSQL training — 5697 entries, 52 types, loss 0.16 → 0.02 | ✅ Done |
| 13 | ChromaDB index building — SQL + NoSQL persistent vector indexes built | ✅ Done |
| 14A | SQL Generator — LoRA SFT on 6748 examples, validated on A100 | ✅ Done |
| 14B | NoSQL Generator — LoRA SFT on 5410 examples, warm-started from 14A | ✅ Done |
| 15A | POSG wired for SQL — SAR → Generator (5 cand.) → Pareto → EX, validated on Spider dev (63.3% vs 60.0% greedy EX, `--hard` subset) | ✅ Done |
| 15B | POSG wired for NoSQL — mirrors 15A for MQL, validated on train split (76.7% vs 73.3% greedy EX) | ✅ Done |
| 16 | End-to-end pipeline assembly (`src/pipeline_sql.py` / `src/pipeline_nosql.py`) | ✅ Done |
| 17 | LangGraph router + self-correction — session-based routing, execute + retry ladder (`src/router/langgraph_router.py`), 19 tests green on Mac | ✅ Done |
| 18 | Evaluation harness + SchemaRAG Table 5 ablation — **SQL 80–81% EX** (target >82%, up from 79–80% after two Phase 19 fixes; root causes below), **NoSQL 84.2% EX** (meets >60% target); CP1 baseline run — 3.0% EX few-shot, 0.0% zero-shot, both well below the ~45–55% plan estimate | ✅ Done |
| 19 | Error analysis — 19/100 misses in the latest run categorized by root cause; two fixes landed (`flight_2` DB fix, eval harness candidate fallback); several patterns scoped but deliberately left unfixed (see below) | ✅ Done |
| 20A | Streamlit demo (`app.py`) — verified end-to-end on the SQL track, k=1 candidates per the plan's <8s target (confirmed 2.7s); launches on Colab via `notebooks/phase20a_streamlit_demo.ipynb` + `cloudflared` | ✅ Done |
| 20B/20C | FastAPI backend, SQL-to-NoSQL migration utility | ⏳ Pending |

## Setup

```bash
conda activate text2sql
pip install torch transformers datasets peft trl langgraph chromadb pymongo rapidfuzz bm25s sqlglot sqlparse networkx FlagEmbedding bitsandbytes
pip install pytest          # tests/ only
```

`bitsandbytes` is required only when running the Generator on a GPU under 20GiB
(T4 / V100-class), where `src/generator/infer.py` loads the 7B model in int8 so it
fits entirely in VRAM. It is unused on Mac/MPS and on ≥20GiB GPUs (bf16 path).

Configure paths in `configs/config.yaml` before running any scripts.

## Running data pipeline scripts

```bash
# Phase 7B — build NoSQL RAG corpus (runs ~20–30 min, checkpoints every 50)
python scripts/build_nosql_rag_corpus.py

# Phase 8A — build SQL CoT training data (runs ~35–45 min, checkpoints every 50)
python scripts/build_cot_data.py

# Phase 8B — build NoSQL CoT training data (runs ~25–35 min, checkpoints every 50)
python scripts/build_nosql_cot_data.py

# Run 8A then trigger 8B automatically (safe to run while 8A is in progress)
bash scripts/run_phase8_pipeline.sh

# Validate Phase 8B output
python scripts/validate_nosql_cot.py
```

## SchemaLinker — API vs model mode

SchemaLinker is switchable via `configs/config.yaml`:

```yaml
schema_linker:
  mode: api      # "api" → DeepSeek API (active); "model" → trained PEFT adapter
  api_model: deepseek-chat
  api_key_env: DEEPSEEK_API_KEY
```

Set `DEEPSEEK_API_KEY` in a `.env` file at the project root. Switch `mode: model` and set `sql_checkpoint` / `nosql_checkpoint` once training is done.

## SAR — ChromaDB vs in-memory mode

SAR retrieval is switchable via `configs/config.yaml`:

```yaml
sar:
  backend: memory   # "memory" → re-encodes corpus at startup (~1s on GPU, ~30s on CPU) [current default]
                    # "chroma" → pre-built ChromaDB index (instant startup)
```

Default is `memory` as of Phase 15 — ChromaDB's `PersistentClient` can't open the Phase 13 index over a Google Drive FUSE mount on Colab, and re-encoding is cheap at this corpus size. Switch back to `chroma` when running locally off a real filesystem.

## Running POSG (Phase 15)

`scripts/run_posg_sql.py` and `scripts/run_posg_nosql.py` wire SAR retrieval → Generator (5 candidates) → POSG Pareto selection → EX scoring into a runnable pipeline, per track:

```bash
# SQL — sample n questions from a CoT/eval file, compare POSG vs greedy EX
python scripts/run_posg_sql.py --data Data/cot_data/sql_dev_eval_full.json --n 30 --hard

# SQL — single question smoke test
python scripts/run_posg_sql.py --smoke_test

# NoSQL — mirrors the SQL script for MQL
python scripts/run_posg_nosql.py --data Data/cot_data/nosql_cot_train.json --n 30 --hard

# Build a held-out Spider dev-split eval set (real DeepSeek SchemaLinker predictions,
# not oracle labels — train-split smoke tests are memorization-inflated, see findings doc)
python scripts/build_dev_eval_set.py --dev Data/Spider/dev.json --out Data/cot_data/sql_dev_eval_full.json
```

`--strategy` selects Pareto tie-break weighting (`balanced` default, `schema_priority`, `example_priority`). Full methodology, false starts, and a known alias-blindness limitation in `posg_sql.py`'s schema-conformity scoring are written up in `docs/phase15_posg_findings.md`.

## Running the router (Phase 17)

`src/router/langgraph_router.py` wraps the Phase 16 pipeline in a LangGraph state machine that **executes** the selected query and retries on failure. Routing is session-based (Option A): the track is fixed with `--track`, not classified per question. SchemaLinker / SAR / Generator are built once and reused for every question in the session.

```bash
# Single question
python -m scripts.run_router --track sql --db_name concert_singer \
    --question "How many singers are there?"

# Interactive session (the 7B model loads once, then stays warm)
python -m scripts.run_router --track sql --db_name concert_singer

# Batch, with retry statistics
python -m scripts.run_router --track sql --batch Data/cot_data/sql_dev_eval_full.json --n 20

# NoSQL track (needs a live mongod, same as Phase 15B/16)
python -m scripts.run_router --track nosql --db_name concert_singer --question "How many singers?"
```

Retry ladder on execution failure (`--max_retries`, default 3):

1. **Next POSG candidate** — already generated, so no GPU cost and fully in-distribution for the fine-tuned adapter.
2. **Generator re-prompt** — only once the ranked candidates are exhausted, the failing query and its execution error are fed back via `GeneratorInfer.generate(previous_attempt=..., error=...)`. The last retry is reserved for this. Note the two extra prompt sections are *not* in the Phase 14 SFT format, so this path leans on the base instruct model's instruction-following.

Because POSG's executability dimension already runs every candidate, a batch containing any working query gets it ranked first and the ladder never fires — it is reached mainly when POSG's Pareto front came back empty. `tests/test_router.py` pins that behaviour.

Tests run on Mac with no GPU and no checkpoints (SchemaLinker/SAR/Generator stubbed, real graph + real POSG + a real temp SQLite DB):

```bash
pytest tests/test_router.py -v
```

## Evaluation + ablation (Phase 18)

`src/eval/harness.py` scores the pipeline on a held-out set and replicates SchemaRAG's Table 5 ablation. `scripts/run_eval.py` drives it:

```bash
# Full pipeline, 100 held-out Spider dev questions
python -m scripts.run_eval --track sql --data Data/cot_data/sql_dev_eval_full.json --n 100

# Full Table 5 sweep — 4 configurations
python -m scripts.run_eval --track sql --data Data/cot_data/sql_dev_eval_full.json --n 100 --ablation all

# NoSQL (needs a live mongod with the databases loaded)
python -m scripts.run_eval --track nosql --data Data/cot_data/nosql_cot_train.json --n 100

# CP1 baseline — codegen-350M, the "no schema-awareness" floor (~45–55% expected)
python -m scripts.run_baseline --data Data/cot_data/sql_dev_eval_full.json --n 100
```

| Configuration | What changes |
|---|---|
| `full` | SchemaLinker + SAR + POSG |
| `no_schema_linker` | `key_fields` forced to `[]` — Generator sees "N/A", POSG's schema-conformity goes flat |
| `no_sar` | `sar_examples` forced to `[]` — Generator sees "N/A", POSG's example-consistency goes flat |
| `no_posg` | POSG selection replaced by the greedy top-1 candidate |

Configurations sharing generator inputs share a generation pass, so the four-way sweep costs **three** generation passes per question, not four (`full` and `no_posg` differ only in how a candidate is picked). `plan_generation_passes()` makes the grouping explicit and it's pinned by a test.

**Two things to know about the reported EX:**

- EX is the mean over questions whose *gold* query executed, not over all questions. A missing local `.sqlite` (SQL) or a gold pipeline returning 0 rows (NoSQL, meaning the database was never loaded into Mongo) is excluded and counted separately — scoring those as wrong would silently deflate EX. `format_summary()` prints the excluded count.
- `exact_match` is reported because it needs no database, but it badly understates correctness — a correct query phrased differently scores 0. EX is the headline number. Targets: >82% SQL, >60% NoSQL.

Reports are written to `evaluation/results/` with every per-question query kept, as the input for Phase 19 error analysis.

```bash
pytest tests/test_eval_harness.py -v
```

### Results (n=100, Colab A100, 2026-07-26 initial run; fixes below landed 2026-08-01)

| Track | `full` EX | Target | `no_schema_linker` | `no_sar` | `no_posg` |
|---|---|---|---|---|---|
| SQL   | 80–81% (post-fix, 2 runs) | >82% — **below** | −6.0% | −1.0% | −2.0% |
| NoSQL | 84.2% (95/100 scored) | >60% — **meets** | +0.0% | **−20.0%** | −1.1% |

**SAR's contribution is track-dependent** — small but nonzero for SQL (−1.0%, consistent across post-fix runs), but the single largest ablation effect on NoSQL (−20%). Likely explanation: the NoSQL Generator (5,410 fine-tuning examples, warm-started from the SQL adapter, MongoDB's less-regular aggregation-pipeline syntax) leans on SAR's retrieved few-shot examples much more than the SQL Generator does. Don't generalize either track's finding to the other. NoSQL's number is on the **train split** (`nosql_cot_train.json` — no held-out NoSQL set exists), so treat it as an upper bound, not a pure generalization result. Note the ablation *deltas* are far more stable than the absolute `full` number, which still swings ±1 point between `n=100` runs from generation sampling noise (`temperature=0.8`) — the full 1034-question held-out set would remove most of that, but hasn't been run.

**Two fixes landed from the Phase 19 error analysis, moving SQL from 79–80% to 80–81%:**

1. **`flight_2` database bug** (found via the Phase 20A demo, not the offline analysis): `Flights.SourceAirport`/`DestAirport` are stored with a leading space that `Airports.AirportCode` and every gold-query literal lack, so any `WHERE`/`JOIN` comparing them silently returned 0/empty instead of erroring — **gold itself** was scoring wrong on 42/1034 held-out questions (4.1%), not just predictions. Scanned all 166 Spider databases; `flight_2` is the only one where this reaches a column the held-out dev set actually filters/joins on. Fixed via `TRIM()`, applied fresh inside `notebooks/phase18_eval_ablation_res.ipynb` right after the DB unzip step (self-applying on every run). Clean copy linked from `datasets/spider/README.md`.
2. **Eval harness only scored the top-ranked POSG candidate** (`ranked[0]`) — a failed top candidate sank the question even when a lower-ranked one (already generated in the same batched `generate()` call, no extra GPU cost) would have executed fine. `src/eval/harness.py`'s `_pick_executable()` now walks `ranked[]` on execution failure, mirroring the Router's own retry ladder. Checks executability only, never compares to gold mid-selection — no answer-leaking. Applies to `full`/`no_sar`/`no_schema_linker`; `no_posg` stays pure greedy top-1 by design.

**SQL's remaining shortfall (80–81% vs. >82%) is concentrated in the hard-query bucket** (JOIN/subquery/GROUP BY — 71.2% EX vs. 95.1% on easy questions, 19 misses out of 100 scored in the latest run). Bucketed by root cause:

| Category | Count | Mechanically fixable? |
|---|---|---|
| Missing `DISTINCT` after a one-to-many JOIN | 4 | Only 2 of 4 share a safe shape — see below |
| Gold query itself looks questionable | 3+ | No — this is a data-quality thread, not a model bug |
| Wrong column/table picked (schema-linking) | 3–4 | No — requires better retrieval/linking, not post-processing |
| Hallucinated `EXCEPT`/`INTERSECT` chain | 2 | No — traced to SAR retrieving directionally-biased examples in one case |
| Genuine comprehension error (e.g. MIN/MAX confusion on "greater than **any**", literal `=2` instead of counting) | 2 | No — real NL-understanding gap |
| Ties collapsed by `ORDER BY ... LIMIT 1` | 1 | No — needs the `WHERE x = (SELECT MIN/MAX...)` rewrite, changes result shape |
| Minor/debatable (extra column, alternate valid interpretation) | 2 | N/A |

On the missing-`DISTINCT` category: only 2 of the 4 cases share a genuinely safe, mechanical shape — a bare `SELECT` (no aggregate, no `GROUP BY`) joining a "one" table to a "many" table while selecting only "one"-side columns. That's provably lossless to `DISTINCT` (nothing from the "many" side is displayed, so the JOIN can only produce identical fanned-out copies of the same row). The other 2 cases need real restructuring — one has an aggregate (`AVG`) where naively adding `DISTINCT` to the wrong column would itself introduce a bug, the other has no JOIN at all and needs schema knowledge of which column means "degree name." Scoped but **deliberately not built this round** — same judgment call as the original decision to leave `sql_fixups.py` narrow.

One class of failure — **ambiguous column names causing an outright SQLite execution error** (`ambiguous column name: X`) — was a genuine, safely-fixable bug rather than a model-judgment issue, and is fixed: see `src/generator/sql_fixups.py`.

### CP1 baseline — actually run

`scripts/run_baseline.py` (codegen-350M, no SchemaLinker/SAR/POSG/fine-tuning) scored **0.0% EX zero-shot** and **3.0% EX few-shot** (`--k_shot 3`, n=100) — both far below the plan's ~45–55% estimate. Not a harness bug: manual inspection of raw completions shows the model echoing the schema's own `(col:TYPE, examples:...)` comment syntax back as if it were the SQL answer, or degenerating into repeated-token loops, rather than producing malformed-but-plausible SQL. codegen-350M is small and not instruction-tuned; three diverse few-shot exemplars weren't enough to reliably separate "schema description" from "answer" at this scale. The finding — that the SchemaRAG architecture's components (SchemaLinker + SAR + POSG + fine-tuning) take the same base capability from ~3% to 80%+ EX — is the deliverable here, not a fixed baseline number.

## Running the Streamlit demo (Phase 20A)

`app.py` wraps `src/router/langgraph_router.py`'s `Router` directly — same pipeline `scripts/run_router.py` drives, with a UI instead of a CLI:

```bash
pip install streamlit
streamlit run app.py
```

Sidebar controls: track (SQL/NoSQL), POSG strategy, max retries, and **candidates (k)** — defaults to **k=1** per the plan's demo latency target (<8s; confirmed 2.7s on a real question at k=1 on Colab A100). Bump it to 3–5 to demo POSG's candidate ranking instead of greedy decoding, at the cost of latency. The `Router` is cached per settings combination so the 7B Generator and SAR encoder load once per session, not per question.

Needs (all gitignored, produced on Colab): `models/generator_{sql,nosql}/`, `models/sar_{sql,nosql}/sar_model.pt`, `Data/Spider/database/` for the SQL track to execute (not just generate) a query, `DEEPSEEK_API_KEY` in `.env`, and a live `mongod` for the NoSQL track.

On Colab, `notebooks/phase20a_streamlit_demo.ipynb` sets up the same Drive checkpoint/database layout as `phase18_eval_ablation_res.ipynb`, then exposes the app via a `cloudflared` quick tunnel (`npx --yes cloudflared tunnel --url http://127.0.0.1:8501`) — no signup, no browser interstitial (switched from `localtunnel` after its interstitial page was found to intermittently serve HTML in place of Streamlit's JS chunks, breaking the app with a "Failed to load module script" error).

## Training scripts (run on Colab)

```bash
# SchemaLinker Stage 1 — CoT SFT (deferred; using API for now)
python -m src.schema_linker.train_stage1 \
    --data Data/cot_data/sql_cot_train.json \
    --model Qwen/Qwen3-8B --out models/schema_linker_cot

# SAR training — SQL (Phase 12A, complete)
python -m src.sar.train \
    --corpus Data/rag_corpus/spider_sql_rag.json \
    --out models/sar_sql --epochs 10

# SAR training — NoSQL (Phase 12B, complete)
python -m src.sar.train \
    --corpus Data/rag_corpus/spider_nosql_rag.json \
    --out models/sar_nosql --epochs 10

# ChromaDB index building — SQL (Phase 13, run locally or on Colab)
python -m scripts.build_chroma_index \
    --corpus Data/rag_corpus/spider_sql_rag.json \
    --model  models/sar_sql/sar_model.pt \
    --out    indexes/chroma_sql --name sar_sql

# ChromaDB index building — NoSQL
python -m scripts.build_chroma_index \
    --corpus Data/rag_corpus/spider_nosql_rag.json \
    --model  models/sar_nosql/sar_model.pt \
    --out    indexes/chroma_nosql --name sar_nosql

# Phase 14A — build SQL generator training data (local; queries ChromaDB for
# top-3 SAR examples per entry, writes Qwen-format JSONL). ~15-20 min on MPS.
python -m scripts.build_generator_training_data --track sql \
    --cot        Data/cot_data/sql_cot_train.json \
    --chroma_dir indexes/chroma_sql \
    --sar_model  models/sar_sql/sar_model.pt \
    --out        Data/generator_data/sql_generator_train.jsonl

# Phase 14A — fine-tune SQL Generator (Colab A100 recommended: ~45-60 min).
# Prints live [TRAIN] xx% progress + [CKPT] on each epoch checkpoint; auto-resumes.
python -m src.generator.train \
    --data Data/generator_data/sql_generator_train.jsonl \
    --out  models/generator_sql --use_a100

# Phase 14B — build NoSQL generator training data (MQL labels).
python -m scripts.build_generator_training_data --track nosql \
    --cot        Data/cot_data/nosql_cot_train.json \
    --chroma_dir indexes/chroma_nosql \
    --sar_model  models/sar_nosql/sar_model.pt \
    --out        Data/generator_data/nosql_generator_train.jsonl

# Phase 14B — fine-tune NoSQL Generator, warm-started from the 14A checkpoint
# (--init_from merges the SQL adapter into base, then trains a fresh LoRA on MQL).
python -m src.generator.train \
    --data      Data/generator_data/nosql_generator_train.jsonl \
    --out       models/generator_nosql --use_a100 \
    --init_from models/generator_sql
```

## Project structure

```
app.py                        Phase 20A Streamlit demo — wraps router.langgraph_router.Router with a UI

datasets/
  spider/README.md            Pointer to the flight_2-fixed Spider database zip on Drive
                               (Data/Spider/database/ itself stays gitignored, ~870MB)

src/                          reusable library code
  device.py                   MPS / CUDA / CPU detection
  fk_graph.py                 FK graph builder (Phase 5A)
  prompt_schema.py            BM25S column annotation — build time (Phase 6)
  schema_utils.py             BM25S column annotation — query time (inference)
  model_interface.py          Qwen inference wrapper (ModelInterface class)
  mongodb_converter.py        SQLite → MongoDB converter (Phase 5B)
  schema_linker/
    linker.py                 ApiSchemaLinker (DeepSeek) + ModelSchemaLinker — switchable
    train_stage1.py           CoT SFT — LoRA r=64 on Qwen-7B (deferred)
    train_stage2.py           MTL — error detection + correction + generation (deferred)
    train_stage3_grpo.py      GRPO — TP/FP/FN reward (FN penalty = -3) (deferred)
    infer.py                  SchemaLinker inference with retry loop
    fix.py                    BGE embedding fix — snaps hallucinated links to real columns
  sar/
    sar_model.py              SchemaAwareModel — dual cross-attention architecture
    train.py                  SAR contrastive training (triplet loss, margin=0.3)
    infer.py                  SARRetriever + ChromaSARRetriever + get_sar_retriever()
    format_schema.py          Schema text parser for SAR training
  generator/
    train.py                  Qwen2.5-Coder-7B LoRA SFT (14A/14B) — warm-start, auto-resume, live progress
    infer.py                  GeneratorInfer — track-aware (sql/nosql) n-candidate generation for POSG
    sql_fixups.py              Deterministic post-generation SQL fixups (Phase 18 error analysis) — qualifies columns ambiguous under a JOIN
  posg/
    posg_sql.py               Pareto-optimal SQL selector (ASTProcessor + 3-dim Pareto) — wired + validated (15A)
    posg_nosql.py             Pareto-optimal MQL selector (stage-type similarity) — wired + validated (15B)
  pipeline_sql.py              SchemaLinker → SAR → Generator → POSG library for SQL — run_pipeline() (Phase 16)
  pipeline_nosql.py            SchemaLinker → SAR → Generator → POSG library for NoSQL — run_pipeline() (Phase 16)
  eval/
    exec_eval.py              EX metric — column-permutation-aware result comparison
    harness.py                Ablation-aware evaluation + reporting (Phase 18); walks ranked
                               candidates on execution failure instead of scoring only
                               ranked[0] (Phase 19 fix)
  router/
    langgraph_router.py       LangGraph state machine — Router, execute + retry ladder (Phase 17)

scripts/
  validate_spider.py                    Spider download validation (Phase 4)
  Validate_sql2mongo_conversion.py      MongoDB conversion validation (Phase 5B)
  build_rag_corpus.py                   SQL RAG corpus builder (Phase 7A)
  build_nosql_rag_corpus.py             NoSQL RAG corpus builder (Phase 7B)
  build_cot_data.py                     SQL CoT data generator (Phase 8A)
  build_nosql_cot_data.py               NoSQL CoT data generator (Phase 8B)
  run_phase8_pipeline.sh                Runs 8A → verifies → triggers 8B automatically
  validate_nosql_cot.py                 Phase 8B output validation (5 checks)
  build_chroma_index.py                 ChromaDB index builder (Phase 13) — SQL + NoSQL
  build_generator_training_data.py      Generator training data builder — --track sql|nosql (14A/14B)
  build_dev_eval_set.py                 Held-out Spider dev-split eval set via live SchemaLinker (Phase 15A)
  run_posg_sql.py                       SAR → Generator → POSG → EX pipeline for SQL (Phase 15A)
  run_posg_nosql.py                     SAR → Generator → POSG → EX pipeline for NoSQL (Phase 15B)
  run_router.py                         Router session CLI — single/interactive/batch (Phase 17)
  run_eval.py                           Evaluation + Table 5 ablation driver (Phase 18)
  run_baseline.py                       CP1 baseline — codegen-350M EX floor (Phase 18C)

tests/
  test_router.py                        Router graph + retry ladder, stubbed models (Phase 17)
  test_eval_harness.py                  Ablation grouping + EX semantics + reporting (Phase 18)
  test_sql_fixups.py                    Ambiguous-column fixup — correctness + set-operator scoping (Phase 18)

notebooks/
  phase9a_sl_train.ipynb                SchemaLinker SQL SFT on Colab (Phase 9A, deferred)
  phase12a_sar_sql_train.ipynb          SAR SQL training on Colab T4 (Phase 12A) ✅
  phase12b_sar_nosql_train.ipynb        SAR NoSQL training on Colab T4 (Phase 12B) ✅
  phase13_chroma_index.ipynb            ChromaDB index building on Colab (Phase 13) ✅
  phase14a_generator_sql_train.ipynb    SQL Generator fine-tuning on Colab A100 (Phase 14A) ✅
  phase14b_generator_nosql_train.ipynb  NoSQL Generator fine-tuning on Colab A100 (Phase 14B) ✅
  phase18_eval_ablation_res.ipynb       CP1 baseline + full Table 5 ablation sweep, both tracks, on Colab A100 (Phase 18) ✅
                                         includes the flight_2 TRIM() fix (Phase 19), applied fresh every run
  phase20a_streamlit_demo.ipynb         Launches app.py on Colab GPU via a cloudflared tunnel (Phase 20A) ✅

Data/
  Spider/                 7000 Q-SQL pairs + 166 SQLite databases
  fk_graphs/              FK graphs for all 166 databases
  mongodb/                MongoDB schema cache
  prompt_schema/          BM25S column annotations (sql/ + nosql/)
  rag_corpus/             SQL corpus (done) + NoSQL corpus (done)
  cot_data/               SQL CoT data (done) + NoSQL CoT data (done)
  generator_data/         SQL + NoSQL generator training JSONL (14A/14B, gitignored)

external/
  SchemaRAG/              Reference implementation — all scripts audited and adapted
```

## SchemaRAG codebase

The `external/SchemaRAG/` directory contains the SchemaRAG reference implementation (SIGMOD 2026). All scripts were audited and key components were adapted into `src/`. The released data includes `RAG_Spider.json` (3102 Q-SQL pairs) and `RAG_BIRD.json`; CoT training data is not released (hence Phase 8A).

## Reference documents

- `docs/architecture.md` — full architecture with design decisions and component deep dives
- `docs/phase15_posg_findings.md` — POSG wiring methodology, EX results, and known limitations (SQL + NoSQL)
- `CodeGen_Plan_v6_DualTrack.md` — full 20-phase implementation plan (latest)
- `docs/SchemaRAG.pdf` — primary SQL track paper (SIGMOD 2026)
- `docs/Text_to_NoSQL.pdf` — NoSQL track paper (TEND)
