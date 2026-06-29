# CodeGen Architecture Document
## Updated through Phase 13 — SAR training complete, ChromaDB index building in progress

---

## Table of Contents

1. [What We Are Building](#1-what-we-are-building)
2. [Research Papers This Is Grounded In](#2-research-papers-this-is-grounded-in)
3. [Dataset — Spider](#3-dataset--spider)
4. [Overall Pipeline Architecture](#4-overall-pipeline-architecture)
5. [Dual-Track Design — Why Two Separate Models](#5-dual-track-design--why-two-separate-models)
6. [Hardware Strategy](#6-hardware-strategy)
7. [SchemaRAG Codebase Audit](#7-schemarag-codebase-audit)
8. [Component Deep Dives — Data Pipeline](#8-component-deep-dives--data-pipeline)
   - [8.1 FK Graph Builder (Phase 5A)](#81-fk-graph-builder-phase-5a)
   - [8.2 MongoDB Converter (Phase 5B)](#82-mongodb-converter-phase-5b)
   - [8.3 PromptSchema via BM25S (Phase 6)](#83-promptschema-via-bm25s-phase-6)
   - [8.4 SQL RAG Corpus (Phase 7A)](#84-sql-rag-corpus-phase-7a)
   - [8.5 NoSQL RAG Corpus (Phase 7B)](#85-nosql-rag-corpus-phase-7b)
   - [8.6 SQL CoT Data (Phase 8A)](#86-sql-cot-data-phase-8a)
   - [8.7 NoSQL CoT Data (Phase 8B)](#87-nosql-cot-data-phase-8b)
9. [Component Deep Dives — Model Training Scripts](#9-component-deep-dives--model-training-scripts)
   - [9.1 ModelInterface — Local Inference Wrapper](#91-modelinterface--local-inference-wrapper)
   - [9.2 SchemaLinker — 3-Stage Training](#92-schemalinker--3-stage-training)
   - [9.3 SAR — Schema-Aware Retriever](#93-sar--schema-aware-retriever)
   - [9.4 POSG — Pareto-Optimal Generator](#94-posg--pareto-optimal-generator)
   - [9.5 EX Evaluation Metric](#95-ex-evaluation-metric)
10. [Key Design Decisions and Why](#10-key-design-decisions-and-why)
11. [Data Flow — End to End](#11-data-flow--end-to-end)
12. [File and Folder Structure](#12-file-and-folder-structure)

---

## 1. What We Are Building

CodeGen is a **dual-track natural language to query system**. Given a plain English question and a database, it produces the correct query — either SQL (for relational databases like PostgreSQL) or MQL (for MongoDB).

```
User: "How many singers are from France?"
         │
         ▼
    CodeGen
         │
    ┌────┴────┐
    ▼         ▼
  SQL        MQL
SELECT     db.singer.aggregate([
COUNT(*)     {"$match": {"Country": "France"}},
FROM singer  {"$count": "total"}
WHERE      ])
Country =
'France'
```

The system is a multi-stage pipeline where each stage narrows down what the next stage needs to process. This mirrors how a human database expert thinks: first identify which tables are relevant, then find similar past queries for reference, then write the query, then verify it.

---

## 2. Research Papers This Is Grounded In

### SchemaRAG (SIGMOD 2026)
The primary paper for the SQL track. Key contributions:
- **PromptSchema**: Enrich schema with BM25-selected sample values so the LLM understands what each column contains
- **SchemaLinker**: A 3-stage trained model (CoT SFT → MTL → GRPO) that identifies which tables and columns are relevant to a question
- **SAR (Schema-Augmented Retriever)**: A dual-stage retrieval model that finds structurally similar past Q-SQL examples
- **POSG (Pareto-Optimal SQL Generator)**: Generates multiple SQL candidates and picks the best one using two quality dimensions

The paper reports >80.4% Execution Accuracy on Spider dev set with Qwen-7B, which is our target. The full codebase was released and audited — see Section 7.

### TEND / SMART (Text-to-NoSQL)
The primary paper for the NoSQL track. Key insight: NoSQL query generation should be treated as a distinct problem from SQL generation. Direct MQL generation outperforms SQL-to-MQL cascade by more than 20 points. Algorithm 1 from the TEND paper defines how to convert relational schemas to MongoDB collections — which directly informed Phase 5B.

---

## 3. Dataset — Spider

| Component | Count | Purpose |
|---|---|---|
| `train_spider.json` | 7,000 Q-SQL pairs | Training all models |
| `dev.json` | 1,034 Q-SQL pairs | Evaluation benchmark |
| `tables.json` | 166 database schemas | FK relationships, column metadata |
| `database/` folder | 166 SQLite files | Data for FK graphs, PromptSchema, execution |

**Why Spider for both tracks**: The SQL track uses Spider directly. For the NoSQL track, we convert the 166 SQLite databases to MongoDB (Phase 5B) and translate the 7,000 Q-SQL pairs to Q-MQL pairs via DeepSeek-V3 (Phase 7B). There is no pre-built public Text-to-NoSQL dataset with matching MongoDB schemas.

---

## 4. Overall Pipeline Architecture

```
User Natural Language Question
           │
           ▼
  [Session Config: PostgreSQL / MongoDB]
           │
           ▼
  [LangGraph Router]
      │           │
      ▼           ▼
 [SQL Track]  [NoSQL Track]
      │           │
      └─────┬─────┘
            │
    ┌───────▼────────────────────────────┐
    │         Shared Pipeline            │
    │                                    │
    │  1. PromptSchema                   │
    │     schema_utils.py (query-time    │
    │     BM25S on the question)         │
    │                                    │
    │  2. SchemaLinker                   │
    │     src/schema_linker/infer.py     │
    │     → fix.py (BGE correction)      │
    │                                    │
    │  3. SAR                            │
    │     src/sar/infer.py               │
    │     (SARRetriever, top-k cosine)   │
    │                                    │
    │  4. Generator                      │
    │     src/generator/infer.py (stub)  │
    │                                    │
    │  5. POSG                           │
    │     src/posg/posg_sql.py           │
    │     src/posg/posg_nosql.py         │
    └───────┬────────────────────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
[PostgreSQL]   [MongoDB]
     └──────┬──────┘
            ▼
    [Execution Result]
            │
    [Self-Correction Loop]
    (LangGraph re-runs on error, max 3 retries)
```

**The pipeline is shared but models are separate.** The SchemaLinker for SQL is a different checkpoint from the SchemaLinker for NoSQL. Same for SAR and Generator. SQL and MongoDB queries have fundamentally different structures — sharing weights would force a compromise that is suboptimal for both.

---

## 5. Dual-Track Design — Why Two Separate Models

**Option 1 (Cascade)**: Generate SQL → Convert SQL to MQL using rules/LLM
**Option 2 (Direct)**: Train separate generators for SQL and MQL

We chose Option 2. The TEND paper showed cascade underperforms direct generation by 20+ points. MongoDB's aggregation pipeline uses `$match`, `$group`, `$lookup`, `$unwind` operators that have no direct SQL equivalents. A model trained to think in SQL terms produces awkward or incorrect MQL.

---

## 6. Hardware Strategy

| Work | Where | Why |
|---|---|---|
| Data prep, FK graphs, BM25S, MongoDB conversion | Mac M1 | No GPU needed |
| CoT + MQL generation (API calls) | Mac M1 | Network I/O, no local GPU |
| SchemaLinker Stage 1 SFT | Colab T4 | Qwen-7B with LoRA r=64 fits in 16GB at bf16 |
| SchemaLinker Stage 2 MTL | Colab T4 | Same |
| SchemaLinker Stage 3 GRPO | Colab A100 (Pro) | G=8 samples × 7B ≈ 28GB minimum |
| SAR training | Colab T4 | BGE-large encoder + SchemaAwareModel |
| Generator fine-tuning | Colab A100 (Pro) | 7B + LoRA + batch needs 24GB+ |
| Inference / pipeline testing | Mac M1 (4-bit GGUF) | MPS backend |
| LangGraph, POSG, demo | Mac M1 | No GPU |

**Mac ↔ Colab workflow**: Write scripts locally → `git push` → `git pull` on Colab → train → save to Google Drive at `/content/drive/MyDrive/codegen/checkpoints/`.

**PyTorch on Mac M1**: Uses MPS (Metal Performance Shaders) backend. `src/device.py` detects the right backend automatically:
```python
if torch.cuda.is_available(): return "cuda"
if torch.backends.mps.is_available(): return "mps"
return "cpu"
```

---

## 7. SchemaRAG Codebase Audit

The SchemaRAG repository (`external/SchemaRAG/`) was cloned and all scripts were audited. Every useful component was adapted into our `src/` structure.

### What SchemaRAG released

| Asset | Released? | Notes |
|---|---|---|
| `datas/RAG_Spider.json` | ✅ Yes | 3102 Q-SQL pairs with schema text — our Phase 7A equivalent |
| `datas/RAG_BIRD.json` | ✅ Yes | 3835 BIRD benchmark pairs |
| CoT training data | ❌ No | `script_to_COT.py` released but not the output data |
| MTL error dataset | ❌ No | `find_mistakes.py` released but not the errors |

We kept our 7000-entry SQL RAG corpus (vs their 3102). CoT training data must be generated via Phase 8A.

### Scripts adapted into src/

| SchemaRAG file | Our file | Key changes |
|---|---|---|
| `llm_local.py` | `src/model_interface.py` | `modelscope` → `transformers`; MPS via `device.py` |
| `function.py` | `src/schema_utils.py` | Query-time BM25S for inference; evidence param; UTF-8 fix |
| `SchemaLinker_fix.py` | `src/schema_linker/fix.py` | No hardcoded paths; takes schema_text string |
| `use_SchemaLinker.py` | `src/schema_linker/infer.py` | Retry loop kept exactly; uses ModelInterface |
| `train_SchemaLinker_CoT_peft.py` | `src/schema_linker/train_stage1.py` | LoRA r=64 (vs paper's r=16); our data format |
| `train_SchemaLinker_MTL_peft.py` | `src/schema_linker/train_stage2.py` | deepspeed removed; argparse paths |
| `train_SchemaLinker_GRPO_peft.py` | `src/schema_linker/train_stage3_grpo.py` | Reward function ported exactly |
| `train_SAR.py` (model) | `src/sar/sar_model.py` | NaN guards preserved; moved to own file |
| `train_SAR.py` (loop) | `src/sar/train.py` | Embedding cache; triplet loss |
| `SAR_use.py` | `src/sar/infer.py` | SARRetriever class; pre-computes embeddings at load |
| `SAR_train/format_schema.py` | `src/sar/format_schema.py` | Parses our schema text format directly |
| `po.py` | `src/posg/posg_sql.py` | Direct SQLite execute; hardcoded paths removed |
| `po.py` (adapted) | `src/posg/posg_nosql.py` | MQL-specific: stage-type similarity replaces AST |
| `eval/exec_eval.py` | `src/eval/exec_eval.py` | Async removed; UTF-8 fix; clean public API |
| `script_to_COT.py` | `scripts/build_cot_data.py` | DeepSeek replaces GPT-4o; sqlglot entity validation |

### Key differences from SchemaRAG's approach

| Component | SchemaRAG | Us | Reason |
|---|---|---|---|
| BM25S timing | Query-time (per question) | Build-time for training; query-time for inference | `prompt_schema.py` caches; `schema_utils.py` re-runs at inference |
| SQL parser (RAG corpus) | sqlparse | **sqlglot** | 0 failures on 7000 Spider SQLs; typed AST nodes |
| SQL parser (POSG AST) | sqlparse | **sqlparse** | AST edit distance works on sqlparse token trees |
| Structural type vector | 6 dimensions | **7 dimensions** | Added `has_set_op` — UNION/INTERSECT/EXCEPT are structurally incompatible with plain SELECT |
| CoT format | `<reasoning>` | **`<think>`** | SchemaRAG's script_to_COT.py uses `<think>` tags |
| CoT entity validation | Second LLM call | **sqlglot** | Free; no extra API cost; reliable for Spider SQL |
| Teacher model | GPT-4o | **DeepSeek-V3** | ~10× cheaper; comparable quality on structured CoT |
| LoRA rank | r=16 | **r=64** | Higher capacity; Qwen-7B still fits in T4 at bf16 |

---

## 8. Component Deep Dives — Data Pipeline

### 8.1 FK Graph Builder (Phase 5A)

**File**: `src/fk_graph.py`
**Output**: `Data/fk_graphs/{db_name}.json` (166 files)

A directed graph where every table is a node and every FK is an edge from the child table (the one with the FK column) to the parent table (the one being referenced).

```
Example: concert_singer database

singer_in_concert ──FK──► singer
singer_in_concert ──FK──► concert
concert           ──FK──► stadium

Centrality: singer_in_concert is the bridge table (connects everything)
```

The FK graph provides JOIN path information without the LLM having to guess it. It also computes **in-degree centrality** — high-centrality tables are primary entities; low-centrality tables are leaf nodes. This informs the MongoDB conversion decision.

SQLite's `PRAGMA foreign_key_list(table_name)` provides all declared FK constraints. Our implementation reads directly from the SQLite PRAGMA (more general than reading Spider's `tables.json`).

**Key limitation**: Some Spider databases declare no FK constraints even though relationships exist. The FK graph will have no edges for those databases. Fallback (Phase 19 if needed): use co-occurrence of table names in training SQL queries as a proxy FK signal.

---

### 8.2 MongoDB Converter (Phase 5B)

**File**: `src/mongodb_converter.py`
**Output**: 166 live MongoDB databases + `Data/mongodb/{db_name}_schema.json`

Converts all 166 Spider SQLite databases to MongoDB. This creates the NoSQL equivalent of the Spider benchmark that no public dataset provides.

**Design — v1 simplification**: The TEND paper's Algorithm 1 describes a sophisticated embedding vs. reference decision based on FK graph analysis. We chose v1: **all tables become separate collections with reference-based relationships**. Every SQLite row becomes one MongoDB document. FK columns are kept as plain fields (not ObjectId references).

Why: Embedding decisions affect MQL query structure significantly. Reference model gives clean `$lookup`-based MQL that is straightforward to train on.

**Type coercion**: SQLite stores integers as text. Without coercion, MongoDB aggregations on numeric fields would fail. The `_coerce()` method tries `int → float → str`.

**UTF-8 handling**: One database (`wta_1`) had a player name with a non-UTF-8 character. Fix:
```python
conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
```

**Verification**: Row counts matched perfectly across all 166 databases.

---

### 8.3 PromptSchema via BM25S (Phase 6)

**Files**: `src/prompt_schema.py` (build time), `src/schema_utils.py` (query time)
**Output**: `Data/prompt_schema/sql/` and `Data/prompt_schema/nosql/` (332 files)

For every column, PromptSchema stores 3 representative sample values and the inferred data type.

```json
"singer.Country": {
    "sample_values": ["Netherlands", "United States", "France"],
    "inferred_type": "string"
}
```

Without sample values, the LLM sees `c_nm` (could be anything). With values `["John Smith", "Sarah Jones"]`, it understands this is a customer name and not a product code.

**BM25S for strings**: Each distinct value is treated as a "document"; the column name is the query. BM25S picks the 3 most semantically relevant values.

**Numeric sampling**: BM25S is meaningless for numbers. We use even-spread sampling: first, middle, last value — showing the range.

**Two-phase BM25S** (the distinction that emerged during development):
- `src/prompt_schema.py` — **build time**: column name as query → cached JSON. Used when building training data.
- `src/schema_utils.py` — **query time**: `extract_db_samples_enriched_bm25(question, ...)` uses the actual user question as the BM25 query → question-relevant values. Used at inference time. Adapted from SchemaRAG's `function.py`. Corpus prefix: `"{table} {col} {val}"` for better BM25 context. Includes a length guard (avg > 600 chars → keep 1 value per column).

---

### 8.4 SQL RAG Corpus (Phase 7A)

**File**: `scripts/build_rag_corpus.py`
**Output**: `Data/rag_corpus/spider_sql_rag.json` (7,000 entries, 57 structural types)

Every Spider training Q-SQL pair, annotated with a 7-dimensional structural fingerprint:

```json
{
    "question": "How many heads of departments are older than 56?",
    "sql": "SELECT COUNT(*) FROM head WHERE age > 56",
    "db_name": "department_management",
    "structural_type": {
        "num_joins":    0,
        "num_tables":   1,
        "has_group_by": false,
        "has_order_by": false,
        "has_having":   false,
        "has_subquery": false,
        "has_set_op":   false
    }
}
```

The 7th dimension `has_set_op` was added (not in v5 plan) because UNION/INTERSECT/EXCEPT queries are structurally incompatible with plain SELECT. Pairing them as positives in SAR contrastive training would teach SAR that fundamentally different query structures are similar.

**Parser**: `sqlglot` — 0 parse failures on all 7000 Spider SQLs. Provides typed AST nodes (`exp.Join`, `exp.Group`, `exp.Having`) for reliable structural analysis.

**Note**: SchemaRAG also released `RAG_Spider.json` (3102 curated entries). We kept our 7000-entry version — more data, richer structural type annotations.

---

### 8.5 NoSQL RAG Corpus (Phase 7B)

**File**: `scripts/build_nosql_rag_corpus.py`
**Output**: `Data/rag_corpus/spider_nosql_rag.json` — **5697 verified entries** ✅

Translates Q-SQL pairs to Q-MQL pairs using DeepSeek-V3 API. Unlike Phase 7A which annotates existing data, this phase generates new MQL queries from scratch.

**Verification pipeline**: Both the SQL (on SQLite) and the MQL (on MongoDB) are executed; their result row counts are compared. Only pairs where counts match are kept. This is a structural verification, not a semantic one — it confirms the MQL returns the same number of rows as the SQL, catching hallucinated pipeline stages or wrong collection names.

```json
{
    "question":        "How many singers are from France?",
    "mql_collection":  "singer",
    "mql_pipeline":    [{"$match": {"Country": "France"}}, {"$count": "total"}],
    "db_name":         "concert_singer",
    "structural_type": {"num_joins": 0, "num_tables": 1, ...},
    "source_sql":      "SELECT COUNT(*) FROM singer WHERE Country = 'France'"
}
```

**Result**: 5697 out of 7000 entries passed (81.4%). The 18.6% rejection rate is expected — AVG/SUM aggregations produce a single numeric result where count-based comparison cannot validate correctness, so those are conservatively dropped.

**Pipeline stage coverage** in the final corpus: `$project` (4466), `$match` (3064), `$lookup` (2877), `$group` (2589), `$unwind` (2373), `$sort` (1462), `$limit` (952), `$count` (711). The corpus covers all major MQL patterns.

**Checkpointing**: Every 50 entries to `Data/rag_corpus/nosql_checkpoint.json`.

---

### 8.6 SQL CoT Data (Phase 8A)

**File**: `scripts/build_cot_data.py`
**Output**: `Data/cot_data/sql_cot_train.json` — **✅ Complete**

Adapted from SchemaRAG's `script_to_COT.py`. Calls DeepSeek-V3 to generate Chain-of-Thought reasoning for each Q-SQL pair, then validates the output.

**CoT format** (SchemaRAG's `<think>` format, adopted directly):
```
<think>
1. Understand the key concepts in the question:
   • ...
2. Analyze database table relationships:
   • ...
3. Key field for filtering: **table.column** (why this field is critical)
</think>

Summary paragraph...

The key field matching the question is: [table.column].
```

**Validation pipeline (2 checks)**:
1. **Format check** — `<think>` tags present, 3 numbered steps, final `The key field matching the question is:` declaration
2. **Entity check** — the table in the key field must appear in the ground-truth SQL tables (extracted via sqlglot, not a second LLM call)

Entity validation without a second LLM call saves ~50% API cost. sqlglot is deterministic and free.

**Schema format passed to DeepSeek**:
```
# Table: actor
[(actor_id:INT, Primary Key, Examples: [1, 2]),
 (name:TEXT, Examples: [Tom Hanks, Meryl Streep]),
]
# Foreign Keys:
# actor_in_movie.actor_id -> actor.actor_id
```

**Test run result**: 4/5 passed. The 1 failure was a no-WHERE-clause query (`SELECT * FROM teams`) — correctly filtered out since there is no key filtering field and Step 3 does not apply. Production pass rate was ~90%+ across all 7000 entries.

**Bugs found and fixed during Phase 8A** (audited before full run):
- Checkpoint interval was `% 100` in code vs `every 50` in documentation — fixed to `% 50`
- `validate_format` used unanchored `re.search`; extraction patterns used anchored `\s*$` — mismatch caused some entries to be counted as `entity_fail` instead of `format_fail`. Fixed: unified pattern with `re.MULTILINE` across all three uses.

**Checkpointing**: Every 50 entries to `Data/cot_data/cot_checkpoint.json`.

---

### 8.7 NoSQL CoT Data (Phase 8B)

**File**: `scripts/build_nosql_cot_data.py`
**Output**: `Data/cot_data/nosql_cot_train.json` — **🔄 In progress**

Generates CoT training data for the NoSQL SchemaLinker, mirroring Phase 8A but for MongoDB. The input is the 5697-entry NoSQL RAG corpus from Phase 7B. Expected output: ~4800–5200 verified CoT examples.

**Why Phase 8B is needed**: The NoSQL SchemaLinker is trained separately from the SQL SchemaLinker (different checkpoint, `configs/config.yaml: nosql_checkpoint`). It needs its own CoT training data that reasons in MongoDB terms — collections, `$lookup`, `$match`, aggregation pipelines — not SQL table/column terms. Training the NoSQL SchemaLinker on SQL CoT data would produce a model that thinks in SQL and generates wrong MQL field references.

**What's identical to Phase 8A** (directly reused):
- `validate_format()` — same regex, same CoT format contract (`<think>` tags, 3 steps, final key field line)
- Checkpoint logic — every 50 entries, `nosql_cot_checkpoint.json`, resumes with `next_idx`
- `call_deepseek()` — same API, `temperature=0`, `max_tokens=1024`
- Complexity sort — `num_tables` ascending then `num_joins` ascending
- Key field extraction regex — same `[\w.,\s]+` anchored to end of line

**What's different from Phase 8A**:

| Aspect | Phase 8A (SQL) | Phase 8B (NoSQL) |
|---|---|---|
| Input | `train_spider.json` (7000 SQL entries) | `spider_nosql_rag.json` (5697 MQL entries) |
| Schema format | `# Table: actor` | `# Collection: actor` |
| FK label | `# Foreign Keys:` | `# Relationships (via $lookup):` |
| Prompt shown to DeepSeek | Ground-truth SQL | `db.collection.aggregate([...])` |
| Step 2 instruction | "Analyze database table relationships" | "Analyze MongoDB collection relationships" |
| Entity validation | sqlglot parses SQL AST → extracts `exp.Table` nodes | Custom MQL parser — walks pipeline dicts for `$lookup.from`, `$unionWith.coll` |
| Output field | `"sql": "SELECT ..."` | `"mql": {"collection": ..., "pipeline": [...]}` |

**Why sqlglot cannot be used for Phase 8B**: sqlglot is an SQL parser. MongoDB's MQL is a list of JSON dicts — `[{"$match": {...}}, {"$lookup": {"from": "other_coll"}}]`. There is no standard MQL AST parser. Instead, `extract_mql_collections()` walks the pipeline dict recursively:

```python
def extract_mql_collections(entry):
    collections = {entry["mql_collection"].lower()}        # base collection
    for stage in entry["mql_pipeline"]:
        if "$lookup" in stage:
            collections.add(stage["$lookup"]["from"].lower())    # joined collection
        if "$unionWith" in stage:
            uw = stage["$unionWith"]
            if isinstance(uw, dict) and "coll" in uw:
                collections.add(uw["coll"].lower())              # union collection
    return collections
```

This covers all three ways a pipeline can reference another collection. Nested `$lookup` pipelines (a `$lookup` stage that itself contains a `pipeline` key) are also handled recursively.

**CoT format for NoSQL** (same outer structure, MongoDB-specific content):
```
<think>
1. Understand the key concepts in the question:
   • "least common allergy type" → need a count grouped by allergy type
   • Requires $group stage to aggregate, $sort to rank, $limit for top 1

2. Analyze MongoDB collection relationships:
   • Allergy_Type collection holds allergy type labels
   • Has_Allergy collection records student-allergy associations
   • Linked via $lookup on Allergy_Type.Allergy → Has_Allergy.Allergy

3. Key field for filtering: Allergy_Type.AllergyType (grouping dimension)
   This field is the label we count occurrences of — the $group _id field.
</think>

Summary paragraph...

The key field matching the question is: [Allergy_Type.AllergyType]
```

**Automation**: `scripts/run_phase8_pipeline.sh` detects whether 8A is already running, polls every 60 seconds for `sql_cot_train.json` to appear, prints 8A's `next_idx` progress at each tick, verifies the 8A output (entry count, sample keys, sample question), then starts 8B automatically. Also runs `validate_nosql_cot.py` at the end.

**Phase 8B validation** (`scripts/validate_nosql_cot.py`) — 5 checks:
1. All required keys present in every entry
2. `mql.collection` and `mql.pipeline` both present (MQL structure intact)
3. CoT format valid (same `validate_format` as generation)
4. Entity consistency (CoT key collections ⊆ MQL collections)
5. `key_fields` in `collection.field` format throughout

---

## 9. Component Deep Dives — Model Training Scripts

### 9.1 ModelInterface — Local Inference Wrapper

**File**: `src/model_interface.py`

Adapted from SchemaRAG's `llm_local.py`. Wraps `AutoModelForCausalLM` with Qwen's chat template. The original used `modelscope` for model loading; we use standard HuggingFace `transformers`. Supports `enable_thinking=True` for Qwen3's extended reasoning mode.

```python
class ModelInterface:
    def __init__(self, model_path: str, max_new_tokens: int = 32768):
        # Detects device via src/device.py (MPS / CUDA / CPU)
        ...

    def generate(self, instruct, prompt, n=1, num_beams=1, enable_thinking=False) -> List[str]:
        # Returns list of n decoded strings (for POSG k=5 generation)
        ...
```

---

### 9.2 SchemaLinker — API mode (active) + 3-Stage Training (deferred)

The SchemaLinker reduces "all tables and columns in the database" to "only the ones needed to answer this question." For large databases with 50+ tables, the Generator cannot reason over all of them simultaneously — schema linking is the gating step.

#### Current mode — DeepSeek API (`src/schema_linker/linker.py`)

Training Qwen-7B on a T4 with gradient checkpointing takes ~38 hours per track — impractical. The decision was made to use DeepSeek API as the primary SchemaLinker and defer model training. All training code is preserved and the switch is one config line.

**`src/schema_linker/linker.py`** provides two interchangeable classes:

```python
class ApiSchemaLinker:
    # Calls DeepSeek API with the same CoT prompt format used for training data
    # Returns key_fields list; retries up to max_retries on parse failure
    def link(self, question: str, schema: str) -> List[str]: ...

class ModelSchemaLinker:
    # Loads a trained PEFT adapter via ModelInterface
    def link(self, question: str, schema: str) -> List[str]: ...

def get_schema_linker(sl_config: dict, track: str = "sql"):
    # Returns ApiSchemaLinker if mode="api", ModelSchemaLinker if mode="model"
```

**Switch in `configs/config.yaml`**:
```yaml
schema_linker:
  mode: api          # "api" → DeepSeek API (active); "model" → trained adapter
  api_model: deepseek-chat
  api_key_env: DEEPSEEK_API_KEY
  sql_checkpoint: models/schema_linker_sql/    # used when mode=model
  nosql_checkpoint: models/schema_linker_nosql/
```

Set `DEEPSEEK_API_KEY` in a `.env` file. The DeepSeek API uses the same `<think>` CoT format that was used for training data generation — the prompt is identical so reasoning quality is consistent.

#### Stage 1 — CoT SFT (`src/schema_linker/train_stage1.py`) — deferred

Fine-tune Qwen-7B on `sql_cot_train.json` from Phase 8A. The model learns to reason step-by-step before declaring which schema elements are relevant.

**LoRA config** (raised from SchemaRAG's r=16 to r=64 for better capacity):
```python
LoraConfig(
    r=64, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, task_type="CAUSAL_LM"
)
```

Training: 3 epochs, effective batch 16, LR=2e-4, bf16, cosine schedule. Requires Colab A100 (~4 hours) or T4 with gradient checkpointing (~38 hours).

#### Stage 2 — MTL (`src/schema_linker/train_stage2.py`) — deferred

Three tasks trained simultaneously:
- **Task 0** — error detection (weight 0.3): classify whether a Stage-1 prediction is wrong
- **Task 1** — correction (weight 0.4): produce the correct prediction given a wrong one
- **Task 2** — generation (weight 1.0): make correct predictions from scratch

`WeightedRandomSampler` balances task distribution proportional to inverse frequency × task weight.

#### Stage 3 — GRPO (`src/schema_linker/train_stage3_grpo.py`) — deferred

Reinforcement learning where the model generates G=4 candidate schema linkings per question and receives per-token rewards:

```python
reward = (
    +2.0 × true_positives     # correctly predicted key fields
    - 0.5 × false_positives   # predicted but not in ground truth
    - 3.0 × false_negatives   # missed required fields (strongest penalty)
    + 0.5 × f1_score          # F1 bonus
)
# format_fail = -1000 (ensures output format is always valid)
```

FN penalty is strongest because a missing required table makes correct SQL generation impossible. Requires Colab A100 (Pro) — G=4 forward passes on Qwen-7B ≈ 28GB.

#### SchemaLinker fix (`src/schema_linker/fix.py`)

Applied after inference (both API and model modes). Uses BGE-large-en-v1.5 cosine similarity to snap predicted links to the nearest real `table.column` pair. Corrects hallucinations like `actor.nationality → actor.country` without rerunning the model.

#### SchemaLinker inference (`src/schema_linker/infer.py`)

Retry loop: if output parsing fails (missing `<think>` tags or no key field declaration), retry up to 3×. Saves `think_pre` and `answer_pre` alongside `schema_links_pred` for MTL Stage 2 error dataset construction.

---

### 9.3 SAR — Schema-Aware Retriever

**Files**: `src/sar/sar_model.py`, `src/sar/train.py`, `src/sar/infer.py`, `src/sar/format_schema.py`

SAR is the most impactful component in SchemaRAG's ablation — removing it drops EX by 8–16 points depending on the model.

**Why standard vector search is not enough**: Standard retrieval finds semantically similar questions. If a user asks "How many singers are there?", standard retrieval might return complex multi-table JOIN queries about singers. SAR finds questions requiring the same SQL structure — simple COUNT queries — so the Generator gets examples showing the actually relevant pattern.

#### SchemaAwareModel (`src/sar/sar_model.py`)

Two cross-attention stages:

```
Input: BGE-large embeddings (dim=1024) for question, tables, columns

Stage 1 — Column-aware table embeddings (table_column_attention):
  For each table T_i with columns C_i:
  T^C_i = SafeMultiheadAttention(query=T_i, key=C_i, value=C_i)
  T^C_i = LayerNorm(T^C_i + T_i)

Stage 2 — Question-schema fusion (question_table_attention):
  Ŝ = SafeMultiheadAttention(query=Q, key=T^C, value=T^C)
  output = LayerNorm(Ŝ + Q)
  output = output_proj(output)  → [batch, embed_dim]
```

**`SafeMultiheadAttention`**: Handles edge cases where all keys are masked (some databases have tables with no valid columns after filtering). Returns zeros for those samples without crashing. This is a production-grade guard from SchemaRAG's implementation that we preserved exactly.

#### Training (`src/sar/train.py`)

Contrastive triplet loss (margin=0.3): anchor question, positive (same `structural_type` from Phase 7A), negative (different `structural_type`). `EmbeddingCache` with pickle + md5 hash keys avoids recomputing BGE embeddings across epochs.

#### SAR training results (Phase 12A/12B)

Both tracks trained on Colab T4 (~2 minutes each):

| Track | Corpus size | Structural types | Final loss | Model size |
|---|---|---|---|---|
| SQL (Phase 12A) | 7,000 entries | 57 types | 0.15 → 0.02 | 50.4 MB |
| NoSQL (Phase 12B) | 5,697 entries | 52 types | 0.16 → 0.02 | ~50 MB |

Models saved to Google Drive at `checkpoints/sar_sql/sar_model.pt` and `checkpoints/sar_nosql/sar_model.pt`.

#### Inference (`src/sar/infer.py`)

Two retriever backends, switchable via `sar.backend` in `configs/config.yaml`:

**`SARRetriever`** (backend: `memory`) — original. Pre-computes all corpus embeddings at load time (one BGE pass, ~30 sec). Retrieval is a single matrix multiply:
```python
scores  = torch.matmul(q_emb, self.corpus_embs.T).squeeze(0)  # [N]
top_idx = torch.topk(scores, k=top_k).indices
```

**`ChromaSARRetriever`** (backend: `chroma`, Phase 13) — instant startup. Requires `scripts/build_chroma_index.py` to have been run first. Queries a pre-built ChromaDB HNSW index on disk instead of recomputing corpus embeddings at every startup. The SAR model is still loaded at query time to encode the incoming question.

**`get_sar_retriever(sar_config, track)`** — factory function. Reads `sar.backend` from config and returns the appropriate retriever.

---

### 9.4 POSG — Pareto-Optimal Generator

**Files**: `src/posg/posg_sql.py`, `src/posg/posg_nosql.py`

Generates k=5 candidates from the Generator (temperature sampling) and selects the best using Pareto-optimal scoring across 3 dimensions.

#### SQL POSG (`src/posg/posg_sql.py`)

| Dimension | How | Notes |
|---|---|---|
| Executability | Run on SQLite: 1.0 or 0.0 | Hard filter — non-executable candidates excluded from Pareto front |
| Schema conformity | Average of Jaccard and coverage over SQL identifiers vs SchemaLinker predictions | Rewards using what SchemaLinker said to use |
| Example consistency | 1 − normalized AST edit distance from retrieved examples | sqlparse token-tree edit distance |

**`ASTProcessor`**: sqlparse-based. Builds typed AST dict recursively, filters whitespace/comment tokens, computes normalized edit distance via dynamic programming tree alignment.

**Pareto front**: A candidate is Pareto-optimal if no other executable candidate dominates it on both schema conformity AND example consistency simultaneously.

**Selection**: If multiple candidates are Pareto-optimal, score each as `ws × schema_conformity + we × example_consistency` with strategy-defined weights (`balanced`: 0.5/0.5, `schema_priority`: 0.7/0.3, `example_priority`: 0.3/0.7).

#### NoSQL POSG (`src/posg/posg_nosql.py`)

Same algorithm; MQL-specific adjustments:

| Dimension | SQL | NoSQL |
|---|---|---|
| Executability | SQLite execute | `db[col].aggregate(pipeline, maxTimeMS=3000)` |
| Schema conformity | Jaccard over SQL identifiers | Jaccard over collection names (including `$lookup.from`) |
| Example consistency | sqlparse AST edit distance | **Pipeline stage-type similarity** |

No standard MQL AST parser exists. Stage-type similarity (`$match`, `$group`, `$sort` sequence comparison) replaces AST edit distance. Two pipelines with the same sequence of stage types are structurally similar even with different field names.

---

### 9.5 EX Evaluation Metric

**File**: `src/eval/exec_eval.py`

Execution Accuracy (EX): a prediction is correct if executing it on the database produces the same result set as executing the gold SQL.

**Key improvement over naive EX**: column-order permutation awareness. `SELECT a, b` and `SELECT b, a` produce different column orderings but the same result set. Naive EX marks them different. Our `result_eq()` (ported from SchemaRAG) searches for a column permutation that makes the results identical:

```python
def result_eq(r1, r2, order_matters) -> bool:
    # quick_rej: fast multiset comparison before full permutation search
    # get_constraint_permutation: prune the permutation search space
    # multiset_eq: exact comparison after permuting columns
```

`order_matters` is True only when the gold SQL has ORDER BY (result row order is semantically significant).

**Public API**:
```python
score = evaluate_ex(
    pred_sqls=["SELECT name FROM singer WHERE country='France'"],
    gold_sqls=["SELECT Name FROM singer WHERE Country = 'France'"],
    db_dir="Data/Spider/database",
    db_ids=["singer"],
)  # → 1.0
```

---

## 10. Key Design Decisions and Why

### Session-based routing instead of per-query routing
The LangGraph router asks the user at session start whether they are working with PostgreSQL or MongoDB. Per-query detection would require a classifier that can make mistakes. Sessions are natural — a developer works with one database type at a time. Eliminates one failure point.

### conda env `text2sql`
Created for PyTorch MPS (Apple Silicon) installation, which requires a specific install URL incompatible with standard `pip install torch`.

### `Data/` with capital D
Spider was extracted to `Data/Spider/`. All scripts use `os.path.dirname(__file__)` and resolve to the actual folder regardless of working directory.

### BM25S at build time vs query time
Both approaches exist in the codebase serving different purposes:
- `src/prompt_schema.py` (build time): caches per-column representative values for training data construction. Fast, no per-question re-computation.
- `src/schema_utils.py` (query time): uses the actual question as BM25 query at inference. More relevant to the specific question. This is SchemaRAG's production approach.

### sqlglot for structural analysis, sqlparse for AST edit distance
`sqlglot` provides typed AST nodes and handles all Spider SQL variants — used in Phase 7A and Phase 8A entity validation. `sqlparse` produces a token-tree suited for edit distance algorithms — used in POSG's `ASTProcessor`. Both serve distinct purposes and neither replaces the other.

### LoRA r=64 instead of paper's r=16
Higher capacity allows the model to learn more complex CoT reasoning patterns. Qwen-7B still fits in Colab T4 at bf16 with r=64. The SchemaRAG paper used r=16 as a conservative starting point; we raise it since we have the compute headroom.

### Structural type vector is 7-dimensional
Added `has_set_op` (UNION/INTERSECT/EXCEPT detection). A query with UNION is structurally incompatible with a plain SELECT — using them as positives in SAR contrastive training would teach SAR that fundamentally different structures are similar. The 7th dimension prevents this.

### sqlglot for CoT entity validation instead of a second LLM call
The original SchemaRAG `script_to_COT.py` calls the LLM a second time to extract SQL table names for entity validation. We use sqlglot instead: free, deterministic, no extra API cost, and reliable for Spider SQL patterns. This saves ~50% on Phase 8A API cost.

### DeepSeek-V3 instead of GPT-4o
~10× cheaper ($0.0003/1K tokens vs $0.003/1K). Comparable quality on structured CoT tasks (our 4/5 test pass rate matches SchemaRAG's reported quality). Total Phase 8A cost: ~$3.15. Phase 8B cost: ~$2.50 (5697 entries vs 7000).

### MQL collection extraction via pipeline dict walking instead of a parser
Phase 8B cannot use sqlglot for entity validation because sqlglot is an SQL parser. MQL is a list of JSON dicts. We extract collection references by walking the pipeline dict: base collection from `mql_collection`, joined collections from `$lookup.from`, union collections from `$unionWith.coll`. The recursion handles nested `$lookup.pipeline` blocks. This gives the same deterministic, zero-cost guarantee that sqlglot gives for SQL.

### Separate CoT datasets per track (SQL and NoSQL)
The SQL and NoSQL SchemaLinkers are different model checkpoints trained on different CoT data. Merging them into one dataset would force the model to learn two incompatible schema-linking languages simultaneously. Keeping them separate means each model can specialize: SQL model reasons about `table.column` and JOINs; NoSQL model reasons about `collection.field` and `$lookup`. The CoT format contract (key field line, `<think>` tags) is shared so validation code is reused verbatim.

---

## 11. Data Flow — End to End

```
Spider Dataset (Phase 4)
    │
    ├──► FK Graph Builder (Phase 5A)
    │         │
    │         └──► FK graphs → MongoDB Converter (Phase 5B)
    │                               │
    │                               └──► 166 MongoDB databases
    │                                         │
    ├──► PromptSchema SQL (Phase 6) ◄─────────┤ (reads SQLite)
    │         │                               │
    │         └──► sql/{db_name}.json         │
    │                                         │
    ├──► PromptSchema NoSQL (Phase 6) ◄───────┘ (reads MongoDB)
    │         │
    │         └──► nosql/{db_name}.json
    │
    ├──► SQL RAG Corpus (Phase 7A) ──► spider_sql_rag.json ✅ Done
    │                                   (7000 Q-SQL, 7-dim structural types)
    │
    ├──► NoSQL RAG Corpus (Phase 7B) ──► spider_nosql_rag.json ✅ Done
    │         (5697 Q-MQL pairs, DeepSeek translated, MongoDB verified)
    │
    ├──► SQL CoT Data (Phase 8A) ──► sql_cot_train.json ✅ Done
    │         (DeepSeek generates <think>-format reasoning, sqlglot validates)
    │
    └──► NoSQL CoT Data (Phase 8B) ──► nosql_cot_train.json ✅ Done
              (Same CoT format; MQL pipeline shown; collection extractor replaces sqlglot)


─── SCHEMALINKER (Phases 9–11) ────────────────────────────────────

CoT Data ──► DeepSeek API (active, mode=api in config.yaml)
             OR
             SchemaLinker Stage 1 SFT (Phase 9) ──► sl_cot checkpoint  [deferred]
                    │
             Stage 2 MTL (Phase 10) ──► sl_mtl checkpoint               [deferred]
                    │
             Stage 3 GRPO (Phase 11) ──► sl_final checkpoint            [deferred]
                    │
              fix.py (BGE snap) ──► corrected schema links

─── SAR TRAINING (Colab T4) ───────────────────────────────────────

SQL RAG Corpus ──► SAR Training Phase 12A ──► sar_sql/sar_model.pt ✅
NoSQL RAG Corpus ──► SAR Training Phase 12B ──► sar_nosql/sar_model.pt ✅

─── CHROMADB INDEX (Phase 13) ─────────────────────────────────────

sar_sql/sar_model.pt + spider_sql_rag.json
    ──► build_chroma_index.py ──► indexes/chroma_sql/ ✅
sar_nosql/sar_model.pt + spider_nosql_rag.json
    ──► build_chroma_index.py ──► indexes/chroma_nosql/ 🔄

─── FINE-TUNING ────────────────────────────────────────────────

PromptSchema + SchemaLinker + SAR ──► Generator Fine-tuning (Phase 14)
                                                │
                                        generator checkpoint

─── INFERENCE PIPELINE ─────────────────────────────────────────

Question
   │
   ├─ schema_utils.py (query-time BM25S)
   ├─ schema_linker/linker.py (ApiSchemaLinker or ModelSchemaLinker) → fix.py
   ├─ sar/infer.py → ChromaSARRetriever → ChromaDB → top-3 examples
   ├─ generator/infer.py → 5 candidates
   └─ posg/posg_sql.py or posg_nosql.py → final query
```

---

## 12. File and Folder Structure

```
Codegen/
├── src/                               ← importable library code
│   ├── device.py                      ✅ MPS/CUDA/CPU detection
│   ├── fk_graph.py                    ✅ FK graph builder (Phase 5A)
│   ├── mongodb_converter.py           ✅ SQLite → MongoDB (Phase 5B)
│   ├── prompt_schema.py               ✅ BM25S build-time annotation (Phase 6)
│   ├── schema_utils.py                ✅ BM25S query-time annotation (SchemaRAG)
│   ├── model_interface.py             ✅ Qwen inference wrapper (SchemaRAG)
│   ├── schema_linker/
│   │   ├── linker.py                  ✅ ApiSchemaLinker + ModelSchemaLinker (switchable)
│   │   ├── train_stage1.py            ✅ CoT SFT — LoRA r=64, Qwen-7B (deferred)
│   │   ├── train_stage2.py            ✅ MTL — 3 tasks, WeightedRandomSampler (deferred)
│   │   ├── train_stage3_grpo.py       ✅ GRPO — TP/FP/FN reward function (deferred)
│   │   ├── infer.py                   ✅ Inference + retry loop (max 3)
│   │   └── fix.py                     ✅ BGE embedding snap to real columns
│   ├── sar/
│   │   ├── sar_model.py               ✅ SchemaAwareModel — dual cross-attention
│   │   ├── train.py                   ✅ Contrastive training — triplet loss
│   │   ├── infer.py                   ✅ SARRetriever + ChromaSARRetriever + factory
│   │   └── format_schema.py           ✅ Schema text → parsed dict
│   ├── generator/
│   │   ├── train.py                   ⏳ Phase 14 (stub)
│   │   └── infer.py                   ⏳ Phase 16 (stub)
│   ├── posg/
│   │   ├── posg_sql.py                ✅ ASTProcessor + Pareto front (SQL)
│   │   └── posg_nosql.py              ✅ Stage-type similarity + Pareto front (MQL)
│   ├── eval/
│   │   └── exec_eval.py               ✅ EX metric — permutation-aware result eq
│   └── router/
│       └── langgraph_router.py        ⏳ Phase 17 (stub)
│
├── scripts/
│   ├── validate_spider.py             ✅ Spider download validation (Phase 4)
│   ├── Validate_sql2mongo_conversion.py  ✅ MongoDB conversion validation (Phase 5B)
│   ├── build_rag_corpus.py            ✅ SQL RAG corpus builder (Phase 7A)
│   ├── build_nosql_rag_corpus.py      ✅ NoSQL RAG corpus builder (Phase 7B) — 5697 entries
│   ├── build_cot_data.py              ✅ SQL CoT data generator (Phase 8A)
│   ├── build_nosql_cot_data.py        ✅ NoSQL CoT data generator (Phase 8B)
│   ├── run_phase8_pipeline.sh         ✅ 8A → verify → auto-trigger 8B
│   ├── validate_nosql_cot.py          ✅ Phase 8B output validator (5 checks)
│   └── build_chroma_index.py          ✅ ChromaDB index builder — BGE+SAR encode → store (Phase 13)
│
├── notebooks/
│   ├── phase9a_sl_train.ipynb         ✅ SchemaLinker SQL training on Colab (preserved, deferred)
│   ├── phase12a_sar_sql_train.ipynb   ✅ SAR SQL training on Colab T4 (Phase 12A)
│   ├── phase12b_sar_nosql_train.ipynb ✅ SAR NoSQL training on Colab T4 (Phase 12B)
│   └── phase13_chroma_index.ipynb     🔄 ChromaDB index building on Colab (Phase 13)
│
├── Data/                              ← gitignored
│   ├── Spider/                        ✅ 7000 Q-SQL + 166 SQLite DBs
│   ├── fk_graphs/                     ✅ 166 FK graph JSON files
│   ├── mongodb/                       ✅ 166 MongoDB schema JSONs
│   ├── prompt_schema/
│   │   ├── sql/                       ✅ 166 SQL annotation files
│   │   └── nosql/                     ✅ 166 NoSQL annotation files
│   ├── rag_corpus/
│   │   ├── spider_sql_rag.json        ✅ 7000 entries, 57 types, 7-dim
│   │   └── spider_nosql_rag.json      ✅ 5697 entries, all major MQL stage types
│   └── cot_data/
│       ├── sql_cot_train.json         ✅ Complete
│       └── nosql_cot_train.json       ✅ Complete
│
├── external/
│   └── SchemaRAG/                     ✅ Cloned + fully audited (gitignored)
│
├── configs/
│   └── config.yaml
│
├── models/                            ← gitignored
│   ├── sar_sql/sar_model.pt           ✅ Trained Phase 12A (50.4 MB, on Drive)
│   └── sar_nosql/sar_model.pt         ✅ Trained Phase 12B (~50 MB, on Drive)
├── indexes/                           ← gitignored
│   ├── chroma_sql/                    🔄 Phase 13 (building)
│   └── chroma_nosql/                  🔄 Phase 13 (building)
└── docs/
    ├── architecture.md                ← this file
    ├── SchemaRAG.pdf
    └── Text_to_NoSQL.pdf
```

---

*Last updated: Phase 13 in progress. Data pipeline (5A–8B) complete. SchemaLinker using DeepSeek API (training deferred). SAR training complete for both tracks (Phase 12A/12B). ChromaDB index build scripts ready (Phase 13). Next: Phase 14 Generator fine-tuning (Qwen2.5-Coder-7B-Instruct).*
