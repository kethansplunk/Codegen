# CodeGen — Natural Language to Query Generator

A dual-track system that translates natural language questions into SQL (PostgreSQL) and NoSQL (MongoDB MQL) queries using fine-tuned LLMs.

## What it does

Given a natural language question and a database, the system produces the correct query by routing through a multi-stage pipeline:

1. **PromptSchema** — enriches the schema with sample values per column so the LLM understands what each column contains
2. **SchemaLinker** — identifies the relevant tables and columns from the schema (3-stage: CoT SFT → MTL → GRPO)
3. **SAR (Schema-Aware Retriever)** — retrieves structurally similar past examples using a dual cross-attention model trained on structural type similarity
4. **Generator** — produces the final query using a fine-tuned Qwen2.5-Coder-7B model
5. **POSG** — generates 5 candidates and selects the best one via Pareto-optimal scoring (executability + schema conformity + structural distance)

## Models

| Component | Base Model |
|---|---|
| SchemaLinker | Qwen/Qwen2.5-7B |
| SAR encoder | BAAI/bge-large-en-v1.5 + SchemaAwareModel |
| Query Generator | Qwen/Qwen2.5-Coder-7B-Instruct |
| CoT teacher | DeepSeek-V3 (API) |

## Current status

| Phase | Description | Status |
|---|---|---|
| 1–3 | Planning, architecture, environment setup | ✅ Done |
| 4 | Spider dataset — 7000 Q-SQL pairs + 166 SQLite databases | ✅ Done |
| 5A | FK graph builder — NetworkX graphs for all 166 databases | ✅ Done |
| 5B | MongoDB converter — all 166 databases converted and verified | ✅ Done |
| 6 | PromptSchema — BM25S column annotations for SQL and NoSQL | ✅ Done |
| 7A | SQL RAG corpus — 7000 Q-SQL pairs with 57 structural types (7-dim) | ✅ Done |
| 7B | NoSQL RAG corpus — Q-MQL pair generation via DeepSeek API | 🔄 In progress |
| 8A | SQL CoT data — adapted from SchemaRAG's script_to_COT.py | 🔄 In progress |
| 8B | NoSQL CoT data | ⏳ Blocked on 7B |
| 9–11 | SchemaLinker training (SQL + NoSQL, all 3 stages) | ⏳ Pending — scripts ready |
| 12 | SAR training (SQL + NoSQL) | ⏳ Pending — scripts ready |
| 13–20 | ChromaDB, Generator, POSG, eval, demo | ⏳ Pending |

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
```

## Training scripts (run on Colab)

```bash
# SchemaLinker Stage 1 — CoT SFT
python -m src.schema_linker.train_stage1 \
    --data Data/cot_data/sql_cot_train.json \
    --model Qwen/Qwen2.5-7B --out models/schema_linker_cot

# SAR training
python -m src.sar.train \
    --corpus Data/rag_corpus/spider_sql_rag.json \
    --out models/sar_sql --epochs 10
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
    train_stage1.py           CoT SFT — LoRA r=64 on Qwen-7B
    train_stage2.py           MTL — error detection + correction + generation
    train_stage3_grpo.py      GRPO — TP/FP/FN reward (FN penalty = -3)
    infer.py                  SchemaLinker inference with retry loop
    fix.py                    BGE embedding fix — snaps hallucinated links to real columns
  sar/
    sar_model.py              SchemaAwareModel — dual cross-attention architecture
    train.py                  SAR contrastive training (triplet loss, margin=0.3)
    infer.py                  SARRetriever — pre-computes corpus embeddings at load
    format_schema.py          Schema text parser for SAR training
  generator/
    train.py                  Qwen2.5-Coder-7B fine-tuning (Phase 14, stub)
    infer.py                  Generator inference (Phase 16, stub)
  posg/
    posg_sql.py               Pareto-optimal SQL selector (ASTProcessor + 3-dim Pareto)
    posg_nosql.py             Pareto-optimal MQL selector (stage-type similarity)
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

Data/
  Spider/                 7000 Q-SQL pairs + 166 SQLite databases
  fk_graphs/              FK graphs for all 166 databases
  mongodb/                MongoDB schema cache
  prompt_schema/          BM25S column annotations (sql/ + nosql/)
  rag_corpus/             SQL corpus (done) + NoSQL corpus (in progress)
  cot_data/               SQL CoT data (in progress)

external/
  SchemaRAG/              Reference implementation — all scripts audited and adapted
```

## SchemaRAG codebase

The `external/SchemaRAG/` directory contains the SchemaRAG reference implementation (SIGMOD 2026). All scripts were audited and key components were adapted into `src/`. The released data includes `RAG_Spider.json` (3102 Q-SQL pairs) and `RAG_BIRD.json`; CoT training data is not released (hence Phase 8A).

## Reference documents

- `docs/architecture.md` — full architecture with design decisions and component deep dives
- `CodeGen_Plan_v6_DualTrack.md` — full 20-phase implementation plan (latest)
- `docs/SchemaRAG.pdf` — primary SQL track paper (SIGMOD 2026)
- `docs/Text_to_NoSQL.pdf` — NoSQL track paper (TEND)
