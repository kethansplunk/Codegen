# Codegen — Natural Language to Query Generator

A dual-track system that translates natural language questions into SQL (relational databases) and NoSQL (MongoDB) queries using fine-tuned LLMs.

## What it does

Given a natural language question and a database schema, the system produces the correct SQL or MQL query by routing through a multi-stage pipeline:

1. **Router** — detects whether the target is a SQL or NoSQL database (`src/router/`)
2. **Schema Linker** — identifies the relevant tables/collections and columns from the schema (`src/schema_linker/`)
3. **SAR (Schema-Aware Retrieval)** — retrieves similar examples from a vector index to guide generation (`src/sar/`)
4. **Generator** — produces the final query using a fine-tuned Qwen2.5-Coder-7B-Instruct model (`src/generator/`)

## Models

| Component | Base Model |
|---|---|
| Schema Linker | Qwen/Qwen2.5-7B |
| SAR encoder | BAAI/bge-large-en-v1.5 |
| Query Generator | Qwen/Qwen2.5-Coder-7B-Instruct |

## Project structure

```
src/
  device.py            # MPS / CUDA / CPU detection
  fk_graph.py          # Foreign-key graph builder
  prompt_schema.py     # Schema → prompt formatter
  mongodb_converter.py # Converts relational schema to MongoDB format
  router/              # LangGraph-based SQL vs NoSQL router
  schema_linker/       # 3-stage training (SFT → DPO → GRPO)
  sar/                 # Schema-aware retrieval model
  generator/           # Fine-tuned query generator
configs/
  config.yaml          # All model paths, hyperparameters, dataset paths
Data/
  Spider/              # Spider Text-to-SQL benchmark
  cot_data/            # Chain-of-thought training data
  fk_graphs/           # Cached FK graphs
  mongodb/             # MongoDB schema cache
external/
  SchemaRAG/           # Reference implementation (SchemaRAG paper)
```

## Current status

- Project structure and configuration complete (Phase 3E)
- Source file stubs in place for all pipeline components
- Schema linker training planned as 3-stage: SFT → DPO → GRPO
- Dataset: Spider (SQL) + MongoDB-converted equivalent (NoSQL)
- Hardware target: Apple Silicon (MPS) locally, Google Colab for heavy training

## Setup

```bash
pip install torch transformers datasets peft trl langgraph chromadb pymongo rapidfuzz
```

Configure paths in `configs/config.yaml` before running any training or inference scripts.
