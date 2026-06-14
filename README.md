# CodeGen — Natural Language to Query Generator

A dual-track system that translates natural language questions into SQL (PostgreSQL) and NoSQL (MongoDB MQL) queries using fine-tuned LLMs.

## What it does

Given a natural language question and a database, the system produces the correct query by routing through a multi-stage pipeline:

1. **PromptSchema** — enriches the schema with sample values per column so the LLM understands what each column contains
2. **Schema Linker** — identifies the relevant tables and columns from the schema
3. **SAR (Schema-Aware Retrieval)** — retrieves structurally similar past examples to guide generation
4. **Generator** — produces the final query using a fine-tuned Qwen2.5-Coder-7B model
5. **POSG** — generates 5 candidates and selects the best one

## Models

| Component | Base Model |
|---|---|
| Schema Linker | Qwen/Qwen2.5-7B |
| SAR encoder | BAAI/bge-large-en-v1.5 |
| Query Generator | Qwen/Qwen2.5-Coder-7B-Instruct |

## Current status

| Phase | Description | Status |
|---|---|---|
| 1–3 | Planning, architecture, environment setup | ✅ Done |
| 4 | Spider dataset — 7000 Q-SQL pairs + 166 SQLite databases | ✅ Done |
| 5A | FK graph builder — NetworkX graphs for all 166 databases | ✅ Done |
| 5B | MongoDB converter — all 166 databases converted and verified | ✅ Done |
| 6 | PromptSchema — BM25S column annotations for SQL and NoSQL | ✅ Done |
| 7A | SQL RAG corpus — 7000 Q-SQL pairs with 57 structural type labels | ✅ Done |
| 7B | NoSQL RAG corpus — Q-MQL pair generation via DeepSeek API | ⏳ Pending |
| 8–20 | CoT generation, model training, evaluation, demo | ⏳ Pending |

## Setup

```bash
conda activate text2sql
pip install torch transformers datasets peft trl langgraph chromadb pymongo rapidfuzz bm25s sqlglot networkx
```

Configure paths in `configs/config.yaml` before running any scripts.

## Project structure

```
src/                    reusable library code
  device.py             MPS / CUDA / CPU detection
  fk_graph.py           FK graph builder
  prompt_schema.py      BM25S column annotation
  mongodb_converter.py  SQLite → MongoDB converter
  schema_linker/        3-stage SchemaLinker (pending)
  sar/                  Schema-Aware Retriever (pending)
  generator/            Query generator (pending)
scripts/
  validate_spider.py              Spider download validation
  Validate_sql2mongo_conversion.py MongoDB conversion validation
  build_rag_corpus.py             SQL RAG corpus builder
Data/
  Spider/               7000 Q-SQL pairs + 166 SQLite databases
  fk_graphs/            FK graphs for all 166 databases
  mongodb/              MongoDB schema cache
  prompt_schema/        BM25S column annotations (SQL + NoSQL)
  rag_corpus/           Annotated Q-SQL corpus for SAR training
configs/
  config.yaml           All paths and hyperparameters
docs/
  architecture.md       Full architecture documentation
```

## Reference documents

- `docs/architecture.md` — detailed architecture with design decisions
- `CodeGen_Plan_v5_DualTrack.md` — full 20-phase implementation plan
- `docs/SchemaRAG.pdf` — primary SQL track paper (SIGMOD 2026)
- `docs/Text_to_NoSQL.pdf` — NoSQL track paper (TEND)
