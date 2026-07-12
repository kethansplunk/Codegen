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
| 16 | End-to-end pipeline assembly (`src/pipeline_sql.py` / `src/pipeline_nosql.py`) | ⏳ Next |
| 17–20 | LangGraph router + self-correction, eval, error analysis, demo | ⏳ Pending |

## Setup

```bash
conda activate text2sql
pip install torch transformers datasets peft trl langgraph chromadb pymongo rapidfuzz bm25s sqlglot sqlparse networkx FlagEmbedding
```

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
  posg/
    posg_sql.py               Pareto-optimal SQL selector (ASTProcessor + 3-dim Pareto) — wired + validated (15A)
    posg_nosql.py             Pareto-optimal MQL selector (stage-type similarity) — wired + validated (15B)
  eval/
    exec_eval.py              EX metric — column-permutation-aware result comparison
  router/
    langgraph_router.py       LangGraph state machine (Phase 17, stub)

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

notebooks/
  phase9a_sl_train.ipynb                SchemaLinker SQL SFT on Colab (Phase 9A, deferred)
  phase12a_sar_sql_train.ipynb          SAR SQL training on Colab T4 (Phase 12A) ✅
  phase12b_sar_nosql_train.ipynb        SAR NoSQL training on Colab T4 (Phase 12B) ✅
  phase13_chroma_index.ipynb            ChromaDB index building on Colab (Phase 13) ✅
  phase14a_generator_sql_train.ipynb    SQL Generator fine-tuning on Colab A100 (Phase 14A) ✅
  phase14b_generator_nosql_train.ipynb  NoSQL Generator fine-tuning on Colab A100 (Phase 14B) ✅

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
