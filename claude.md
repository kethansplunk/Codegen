# CodeGen Project — Claude Code Context

## What This Project Is
An AIML Capstone (Project 6) building a dual-track natural language to query system:
- Track 1: Text-to-SQL (English → PostgreSQL) using SchemaRAG architecture
- Track 2: Text-to-NoSQL (English → MongoDB) using SMART/TEND architecture
- Orchestrated by LangGraph with session-based routing (Option A)

## Current Phase
Phase 14B complete — both Generators (SQL + NoSQL) fine-tuned (Qwen2.5-Coder-7B LoRA) and validated on A100. Next: Phase 15 — wire up POSG (Pareto-optimal candidate selection); code exists but isn't wired into the pipeline yet.

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
- `configs/config.yaml` — SchemaLinker mode (api/model) + SAR backend (chroma/memory) switches
- `docs/` — architecture doc + reference papers
- `models/`, `indexes/`, `external/SchemaRAG/` — checkpoints, ChromaDB indexes, reference implementation (gitignored)

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

## Known Gaps
- SAR schema fusion gap (Phase 12): SAR never fuses real table/column data into retrieval — fix was deferred pending Phase 14 eval results. Now that both generators are trained, revisit this before/alongside Phase 15 POSG wiring.
- SchemaLinker model-mode training (9A/9B/10/11) is deferred indefinitely in favor of the DeepSeek API; only switch back if API cost/latency becomes a blocker.

## Environment Notes
- `conda activate text2sql` before any Python work
- PyTorch backend: MPS (not CUDA) on Mac; use Colab A100 for generator/SAR training
- pip installs outside conda need `--break-system-packages`
- `DEEPSEEK_API_KEY` must be set in `.env` at project root (SchemaLinker API mode, CoT data gen)
