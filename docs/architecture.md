# CodeGen Architecture Document
## Work in Progress — Updated through Phase 7A

---

## Table of Contents

1. [What We Are Building](#1-what-we-are-building)
2. [Research Papers This Is Grounded In](#2-research-papers-this-is-grounded-in)
3. [Dataset — Spider](#3-dataset--spider)
4. [Overall Pipeline Architecture](#4-overall-pipeline-architecture)
5. [Dual-Track Design — Why Two Separate Models](#5-dual-track-design--why-two-separate-models)
6. [Hardware Strategy](#6-hardware-strategy)
7. [Component Deep Dives](#7-component-deep-dives)
   - [7.1 FK Graph Builder](#71-fk-graph-builder-phase-5a)
   - [7.2 MongoDB Converter](#72-mongodb-converter-phase-5b)
   - [7.3 PromptSchema via BM25S](#73-promptschema-via-bm25s-phase-6)
   - [7.4 SQL RAG Corpus](#74-sql-rag-corpus-phase-7a)
8. [Upcoming Components — Preview](#8-upcoming-components--preview)
   - [8.1 SchemaLinker](#81-schemalinker-phases-911)
   - [8.2 SAR — Schema-Augmented Retriever](#82-sar--schema-augmented-retriever-phase-12)
   - [8.3 Generator](#83-generator-phase-14)
   - [8.4 POSG](#84-posg-phase-15)
9. [Key Design Decisions and Why](#9-key-design-decisions-and-why)
10. [Data Flow — End to End](#10-data-flow--end-to-end)
11. [File and Folder Structure](#11-file-and-folder-structure)

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

The system does not just use a single LLM prompt. It is a multi-stage pipeline where each stage narrows down what the next stage needs to process. This mirrors how a human database expert thinks: first identify which tables are relevant, then find similar past queries for reference, then write the query, then verify it.

---

## 2. Research Papers This Is Grounded In

### SchemaRAG (SIGMOD 2026)
The primary paper for the SQL track. The name stands for Schema-Retrieval-Augmented Generation. Key contributions:
- **PromptSchema**: Enrich schema with BM25-selected sample values so the LLM understands what each column contains
- **SchemaLinker**: A 3-stage trained model (SFT → MTL → GRPO) that identifies which tables and columns are relevant to a question
- **SAR (Schema-Augmented Retriever)**: A dual-stage retrieval model that finds structurally similar past Q-SQL examples
- **POSG (Pareto-Optimal SQL Generator)**: Generates multiple SQL candidates and picks the best one using two quality dimensions

The paper reports >82% Execution Accuracy on Spider dev set, which is our target.

### TEND / SMART (Text-to-NoSQL)
The primary paper for the NoSQL track. Key insight: NoSQL query generation should be treated as a distinct problem from SQL generation, not as a post-processing step on SQL output. The paper shows direct MQL generation outperforms SQL-to-MQL cascade by more than 20 points. This is why we train separate models for each track rather than sharing a single generator.

Algorithm 1 from the TEND paper defines how to convert relational schemas to MongoDB collections — which directly informed our Phase 5B implementation.

---

## 3. Dataset — Spider

Spider is the benchmark dataset used for both tracks.

| Component | Count | Purpose |
|---|---|---|
| `train_spider.json` | 7,000 Q-SQL pairs | Training all models |
| `dev.json` | 1,034 Q-SQL pairs | Evaluation benchmark |
| `tables.json` | 166 database schemas | FK relationships, column metadata |
| `database/` folder | 166 SQLite files | Actual data for FK graphs, PromptSchema, execution |

**Important**: HuggingFace's Spider dataset only downloads the Q-SQL pairs (Parquet format), NOT the 166 SQLite database files. We downloaded the full dataset via Google Drive to get the SQLite files, which are needed for FK graph building, PromptSchema sample extraction, MongoDB conversion, and SQL execution during POSG.

**Why Spider for both tracks**: The SQL track uses Spider directly. For the NoSQL track, we convert the 166 SQLite databases to MongoDB (Phase 5B) and translate the 7,000 Q-SQL pairs to Q-MQL pairs using an LLM (Phase 7B). There is no pre-built public Text-to-NoSQL training dataset with matching MongoDB schemas, so we build it ourselves.

---

## 4. Overall Pipeline Architecture

```
User Natural Language Question
           │
           ▼
  [Session Config: PostgreSQL / MongoDB]
  (User selects at session start — no per-query routing overhead)
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
    │     Enrich schema with sample      │
    │     values per column              │
    │                                    │
    │  2. SchemaLinker                   │
    │     Identify relevant tables and   │
    │     columns from enriched schema   │
    │                                    │
    │  3. SAR (Schema-Aware Retriever)   │
    │     Retrieve top-3 structurally    │
    │     similar past examples          │
    │                                    │
    │  4. Generator                      │
    │     Produce query from:            │
    │     schema + linked entities +     │
    │     3 similar examples             │
    │                                    │
    │  5. POSG                           │
    │     Generate 5 candidates,         │
    │     select best via Pareto         │
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
    (LangGraph re-runs on error)
```

**The pipeline is shared but models are separate.** The SchemaLinker for SQL is a different checkpoint from the SchemaLinker for NoSQL. Same for SAR and Generator. This is intentional — SQL and MongoDB queries have fundamentally different structures, and sharing weights would force the model to find a compromise that is suboptimal for both.

---

## 5. Dual-Track Design — Why Two Separate Models

When we first designed this system, there were two options:

**Option 1 (Cascade)**: Generate SQL → Convert SQL to MQL using rules/LLM
**Option 2 (Direct)**: Train separate generators for SQL and MQL

We chose Option 2 because the TEND paper showed cascade (Option 1) underperforms direct generation by 20+ points. The reason is structural: MongoDB's aggregation pipeline uses operators like `$match`, `$group`, `$lookup`, `$unwind` that have no direct SQL equivalents. A model trained to think in SQL terms will produce awkward or incorrect MQL.

The cost of Option 2 is more training work (two SchemaLinkers, two SARs, two Generators). The benefit is each model can fully optimize for its output format without compromises.

---

## 6. Hardware Strategy

| Work | Where | Why |
|---|---|---|
| Data prep (Phases 3–7) | Mac M1 | No GPU needed — file I/O, Python scripts |
| CoT data generation (Phase 8) | Mac M1 | API calls to DeepSeek, no local GPU |
| SchemaLinker Stage 1 SFT | Colab T4 | 16GB fits Qwen-7B with LoRA r=16 |
| SchemaLinker Stage 2 MTL | Colab T4 | Same |
| SchemaLinker Stage 3 GRPO | Colab A100 | G=8 samples × 7B params ≈ 28GB minimum |
| SAR training | Colab T4 | bge-large encoder + small Transformer |
| Generator fine-tuning | Colab A100 | 7B + LoRA + batch size needs 24GB+ |
| Inference / pipeline testing | Mac M1 (4-bit) | MPS backend, quantized models |
| LangGraph, POSG, demo | Mac M1 | No GPU needed |

**Mac ↔ Colab workflow**: Write scripts locally → `git push` → `git pull` on Colab → train → save checkpoints to Google Drive at `/content/drive/MyDrive/codegen/checkpoints/`.

**PyTorch on Mac M1**: Uses MPS (Metal Performance Shaders) backend instead of CUDA. The `src/device.py` helper detects the right backend automatically:
```python
if torch.cuda.is_available(): return "cuda"     # Colab GPU
if torch.backends.mps.is_available(): return "mps"   # Mac M1
return "cpu"                                          # fallback
```

---

## 7. Component Deep Dives

### 7.1 FK Graph Builder (Phase 5A)

**File**: `src/fk_graph.py`
**Output**: `Data/fk_graphs/{db_name}.json` (166 files)

#### What it is

A graph where every table is a node and every foreign key relationship is a directed edge from the child table (the one with the FK column) to the parent table (the one being referenced).

```
Example: concert_singer database

singer_in_concert ──FK──► singer
singer_in_concert ──FK──► concert
concert           ──FK──► stadium

Centrality: singer_in_concert is the bridge table (connects everything)
```

#### Why we need it

When a user asks "List all singers who performed in concerts at stadiums with capacity over 5000", the system needs to know that answering this requires joining `singer`, `singer_in_concert`, `concert`, and `stadium`. The FK graph provides this JOIN path automatically without the LLM having to guess.

The graph also computes **centrality** — which tables are most referenced by others. High-centrality tables are primary entities (customers, products, singers). Low-centrality tables are leaf nodes (individual transaction records). This information feeds into the MongoDB conversion to decide document structure.

#### How it works technically

SQLite has a built-in `PRAGMA foreign_key_list(table_name)` command that returns all declared FK constraints for a table:
```
PRAGMA foreign_key_list('singer_in_concert')
→ (0, 0, 'concert', 'concert_ID', 'concert_ID', ...)
   (1, 0, 'singer',  'Singer_ID',  'Singer_ID',  ...)
```

The builder reads these for every table in every database and builds a `networkx.DiGraph`. NetworkX then computes in-degree centrality in one line.

#### What SchemaRAG does differently

SchemaRAG reads FK relationships from `tables.json` (Spider's pre-computed metadata file). Our implementation reads directly from the SQLite `PRAGMA`, which is more general — it works for any SQLite database, not just Spider. This means the FK graph builder can be reused for production databases beyond the training set.

#### Key fallback for missing FKs

Some databases in Spider have no declared FK constraints even though the relationships exist (developers sometimes skip writing them). For those cases, the FK graph has no edges. The project plan's risk table documents a fallback: use co-occurrence of table names across training SQL queries as a proxy FK signal (Phase 5A risk L5). This is deferred to Phase 18 if needed.

---

### 7.2 MongoDB Converter (Phase 5B)

**File**: `src/mongodb_converter.py`
**Output**: 166 live MongoDB databases + `Data/mongodb/{db_name}_schema.json` (166 files)

#### What it is

Converts all 166 Spider SQLite databases to MongoDB format, creating a NoSQL equivalent of the Spider benchmark.

#### Why we need it

The TEND paper trained on MongoDB databases derived from Spider's relational schemas, but no pre-built version of this dataset is publicly available. We must recreate it. Without this, we have no training data for the NoSQL track.

#### Design choice — v1 simplification

The TEND paper's Algorithm 1 describes a sophisticated conversion that decides whether to embed related documents or use references based on FK graph analysis. For example, a junction table like `singer_in_concert` could either become its own collection (with references to `singer` and `concert`) or be embedded as an array inside the `concert` document.

We chose v1 simplification: **all tables become separate collections with reference-based FK relationships**. Every SQLite row becomes one MongoDB document. FK columns are kept as plain fields (not converted to ObjectId references).

Why: Embedding decisions are complex and affect MQL query structure significantly. Starting with the simpler reference model gives us clean `$lookup`-based MQL that is easier to train on. If the NoSQL generator quality is insufficient, we revisit embedding in Phase 19.

#### Type coercion

SQLite is weakly typed — it stores integers as text and vice versa. Without type coercion, MongoDB documents would have inconsistent field types for the same column across rows, breaking aggregations. The `_coerce()` method tries `int` first, then `float`, then leaves as string:

```python
# SQLite might store "1992" as a string
# _coerce converts it to integer 1992
# so MongoDB can do: {$match: {Song_release_year: {$gt: 1990}}}
```

#### UTF-8 handling

One database (`wta_1`) contained a player name with a special character (`Albarracín` with `Ñ`) stored in non-UTF-8 encoding in SQLite. Without handling this, the entire database conversion would fail. The fix is one line added to the SQLite connection:

```python
conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
```

This tells SQLite to replace any undecodable bytes with the Unicode replacement character instead of raising an exception.

#### Verification

All 166 databases were validated by comparing SQLite row counts to MongoDB document counts table by table. The result was a perfect match across all 166 databases and all their tables.

---

### 7.3 PromptSchema via BM25S (Phase 6)

**File**: `src/prompt_schema.py`
**Output**: `Data/prompt_schema/sql/{db_name}.json` and `Data/prompt_schema/nosql/{db_name}.json` (332 files total)

#### What it is

For every column in every database, PromptSchema stores:
- 3 representative sample values
- The inferred data type (integer, float, string, boolean)

```json
"singer.Country": {
    "sample_values": ["Netherlands", "United States", "France"],
    "inferred_type": "string"
},
"concert.Year": {
    "sample_values": [2012, 2015, 2018],
    "inferred_type": "integer"
}
```

#### Why we need it

Many real-world databases have cryptic column names: `reg_dt`, `c_nm`, `amt`, `flg_1`. The SchemaLinker model receives the schema as text and needs to understand what each column means to decide if it is relevant to the question. Without sample values, `c_nm` could be anything. With sample values `["John Smith", "Sarah Jones", "Michael Brown"]`, it is clearly a customer name.

SchemaRAG demonstrates this explicitly — removing PromptSchema drops schema linking accuracy measurably on databases with non-descriptive column names.

#### Where BM25S comes in

BM25S is a text ranking algorithm (a fast implementation of the BM25 ranking function). Given a query and a list of documents, it scores each document by relevance to the query.

For string columns, we have up to 20 distinct values and need to pick the 3 most representative. We treat each value as a "document" and use the column name as the query:

```
Column: "Country"
Candidate values: ["Netherlands", "United States", "France", "Germany", "Japan", ...]
BM25S query: "country"
→ Picks the 3 values most relevant to the concept "country"
```

For numeric columns (integers, floats), BM25S is meaningless — there are no text tokens to match. We use even-spread sampling instead (first, middle, last) to show the range of values.

#### Key difference from SchemaRAG's BM25S usage

SchemaRAG runs BM25S at **query time**: for each incoming user question, it picks column values that are most relevant to that specific question. This gives better per-question context but requires running BM25S 7,000 times (once per training example) or on every inference call.

We run BM25S at **build time**: once per column, pick values representative of the column's general content, cache the result. The tradeoff is minor — for schema disambiguation, knowing that `singer.Country` contains country names is almost as useful as knowing which specific countries are relevant to the current question.

This is an engineering decision, not a research one. The constraint is that we are building a training pipeline, not a per-query inference system at this stage.

---

### 7.4 SQL RAG Corpus (Phase 7A)

**File**: `scripts/build_rag_corpus.py`
**Output**: `Data/rag_corpus/spider_sql_rag.json` (7,000 annotated Q-SQL pairs)

#### What it is

Every Spider training Q-SQL pair, annotated with a 6-dimensional structural fingerprint:

```json
{
    "question": "How many heads of departments are older than 56?",
    "sql": "SELECT COUNT(*) FROM head WHERE age > 56",
    "db_name": "department_management",
    "structural_type": {
        "num_joins": 0,
        "num_tables": 1,
        "has_group_by": false,
        "has_order_by": false,
        "has_having": false,
        "has_subquery": false
    }
}
```

The 7,000 pairs produced 57 unique structural types. The most common type (simple 1-table SELECT with no clauses) has 2,189 examples. Complex types with HAVING or subqueries have 100–200 examples each.

#### Why we need it

The SAR (Schema-Augmented Retriever) is trained with contrastive learning. Contrastive learning requires knowing which pairs are "similar" (should have close embeddings) and which are "different" (should have distant embeddings). The structural type defines this:

- Two queries with the same structural type = **positive pair** (similar)
- Two queries with different structural types = **negative pair** (different)

Without this annotation, "similar" has no definition and contrastive training cannot proceed.

At inference time, this corpus becomes the search index. When a user asks a question, SAR encodes it, searches this corpus for structurally similar past examples, and retrieves the top-3 to use as few-shot examples for the Generator.

#### Why sqlglot instead of sqlparse

SchemaRAG's `script_to_RAG.py` uses `sqlparse` for SQL parsing. We use `sqlglot` for three reasons:

1. **Correctness on Spider SQL**: Spider contains complex SQL with nested subqueries, set operations (INTERSECT, UNION), and non-standard SQLite syntax. `sqlglot` handles all of these correctly. `sqlparse` is older and has known parsing failures on complex SQL.

2. **AST with typed nodes**: `sqlglot` produces a typed Abstract Syntax Tree where each node has a specific type (`exp.Join`, `exp.Group`, `exp.Having`, etc.). This makes counting structural elements reliable. `sqlparse` produces a flatter token stream that requires more fragile regex/keyword matching.

3. **Zero parse failures on Spider**: Testing on all 7,000 Spider training queries showed 0 parse failures with `sqlglot`. This validated the choice.

#### What SchemaRAG's RAG script does differently

SchemaRAG's `script_to_RAG.py` does something fundamentally different — it **generates new Q-SQL pairs** using an LLM and validates them with Claude API. This is dataset augmentation (creating more training data), not structural annotation (labeling existing data).

Our Phase 7A does structural annotation of the existing Spider pairs. We defer augmentation to Phase 18 if SAR retrieval quality is insufficient on evaluation.

---

## 8. Upcoming Components — Preview

These components are not yet implemented. This section documents what they are and how they connect to what we have built.

### 8.1 SchemaLinker (Phases 9–11)

**Model**: Qwen/Qwen2.5-7B fine-tuned in 3 stages
**Input**: User question + PromptSchema-enriched database schema
**Output**: List of relevant tables and columns

The SchemaLinker reduces the schema from "all tables and columns in the database" to "only the ones needed to answer this question." This is critical for large databases with 50+ tables — the Generator cannot reason over all of them simultaneously.

**3-stage training**:

Stage 1 — SFT (Supervised Fine-tuning):
Train on Chain-of-Thought reasoning data distilled from GPT-4o/DeepSeek. The model learns to reason step-by-step about which schema elements are relevant. Training data contains explicit reasoning: "The user asks about singers, so I need the singer table. They ask about concerts, so I need the concert table. The question filters by nationality, so I need singer.Nationality."

Stage 2 — MTL (Multi-Task Learning with error correction):
Build an error dataset from Stage 1 failures. Train the model with three tasks simultaneously: detect errors in wrong predictions (weight 3), correct wrong predictions (weight 3), and make correct predictions from scratch (weight 10). This stage makes the model robust to its own mistakes.

Stage 3 — GRPO (Group Relative Policy Optimization):
Reinforcement learning where the model generates G=8 candidate schema linkings per question and receives rewards based on accuracy. Missing a required table (false negative) is penalized 6× more harshly than including an unnecessary table (false positive). Rationale: a missing table makes correct SQL generation impossible; an extra table is ignorable by the Generator.

GRPO requires an A100 GPU because it runs G=8 forward passes simultaneously for the 7B parameter model.

### 8.2 SAR — Schema-Augmented Retriever (Phase 12)

**Architecture**: BAAI/bge-large-en-v1.5 encoder + 2-stage Transformer
**Input**: User question + database schema
**Output**: Embedding used to retrieve similar past examples from ChromaDB

SAR is the most impactful component in SchemaRAG's ablation study — removing it drops Execution Accuracy by 8–16 points depending on the model.

**Why standard vector search is not enough**: Standard retrieval finds semantically similar questions. If a user asks "How many singers are there?", standard retrieval might return questions about singers even if they are complex multi-table JOIN queries. SAR finds questions that require the same SQL structure — simple COUNT queries — so the Generator gets examples that actually show the relevant pattern.

**Stage 1 — Schema-Aware Representation**:
For each question, produce an embedding that captures both the question's meaning AND the database structure. Implementation:
1. Encode each table and its columns separately using bge-large (1024-dim vectors)
2. Run cross-attention: each table embedding attends to its own column embeddings (table learns from its columns)
3. Run cross-attention: question embedding attends to all column-aware table embeddings (question learns from schema)
4. Training signal: the resulting embedding should be similar to the embedding of the correct SQL

**Stage 2 — Contrastive Enhancement**:
Stack the question embedding and schema-aware embedding, feed through a 3-layer Transformer with a causal mask (question can attend to schema, but schema cannot attend back — prevents circular reasoning). Training uses contrastive loss: pairs with the same structural type from Phase 7A should have similar embeddings, pairs with different structural types should have different embeddings.

**The SchemaRAG codebase provides `train_SAR.py`** with the complete implementation including `SchemaAwareModel` and `ContrastiveLearningModel` classes. We will adapt this for both SQL and NoSQL tracks (NoSQL uses 2 layers instead of 3, with 2 attention heads instead of 8, based on the paper's findings for smaller datasets).

### 8.3 Generator (Phase 14)

**Model**: Qwen/Qwen2.5-Coder-7B-Instruct fine-tuned with LoRA
**Input**: PromptSchema output + SchemaLinker output + SAR top-3 examples + question
**Output**: SQL or MQL query

Qwen2.5-Coder was chosen over a general-purpose model because it was pre-trained on code and SQL. The fine-tuning teaches it the specific prompt format and improves accuracy on the Spider schema style.

The full prompt given to the Generator:
```
You are a SQL expert. Generate a single SQL query.

Database: concert_singer
Schema (with sample values):
# Table: singer
[(Singer_ID:INTEGER, Examples: [1, 2, 3]), (Name:TEXT, Examples: ['Joe Sharp', ...]), ...]

Relevant schema elements (from SchemaLinker):
Tables: singer, concert
Columns: singer.Singer_ID, singer.Country, concert.concert_ID

Similar examples (from SAR):
Q: "How many singers are from the UK?" | SQL: SELECT COUNT(*) FROM singer WHERE Country = 'UK'
Q: "List all American singers." | SQL: SELECT Name FROM singer WHERE Country = 'USA'
Q: "Count French performers." | SQL: SELECT COUNT(*) FROM singer WHERE Country = 'France'

Question: How many singers are from France?
SQL:
```

### 8.4 POSG (Phase 15)

**What it is**: Generate 5 SQL candidates from the Generator (with temperature sampling), then select the best one.

Selection uses two dimensions:
1. **Schema Linking Conformity (SSL)**: Jaccard similarity between tables/columns used in the generated SQL and the SchemaLinker's predictions. High score = SQL uses what the schema linker said to use.
2. **Example Consistency (SEC)**: Average AST edit distance between the generated SQL and the 3 retrieved examples. High score = SQL structure resembles the examples.

Find the Pareto-optimal candidates (no other candidate dominates on both dimensions simultaneously) and return the one with the highest geometric mean of SSL and SEC.

For SQL execution, POSG can also do a hard filter: discard any candidate that raises a SQLite error. This prevents syntactically invalid SQL from reaching the user.

---

## 9. Key Design Decisions and Why

### Why session-based routing instead of per-query routing

The LangGraph router asks the user at session start whether they are working with PostgreSQL or MongoDB. An alternative would be detecting the intent from the query itself.

We chose session-based because:
- Per-query detection requires a classifier that can make mistakes (SQL question sent to MongoDB pipeline produces nonsense)
- Sessions are natural — a developer is usually working with one database type at a time
- Eliminates one potential failure point from the pipeline

A fallback exists: if the input looks like an existing SQL query (matches `SELECT|INSERT|UPDATE` regex), the system routes to a SQL-to-NoSQL migration utility regardless of session config.

### Why conda env `text2sql` and not a venv

The `text2sql` conda environment was created specifically to handle PyTorch MPS (Apple Silicon) installation, which requires a specific install URL and does not work with standard `pip install torch`. Conda manages the environment separately from the system Python, avoiding conflicts with macOS system packages. All development work uses this environment.

### Why `Data/` with capital D instead of `data/`

The Spider dataset was originally extracted to `Data/Spider/` (capital D). The `config.yaml` uses lowercase `data/spider/` (the intended convention). All `src/` scripts use `os.path.dirname(__file__)` to compute paths relative to the script file's own location, so they work correctly regardless of the working directory, and they resolve to the actual `Data/` folder. This is a known inconsistency that will be cleaned up in a later phase.

### Python module access and sys.path

Python does not automatically make parent directories importable. Running `python src/fk_graph.py` from the project root adds the project root to `sys.path`, making `from src.fk_graph import FKGraphBuilder` work from any other script that is also run from the project root. This is why all scripts are run from the project root, not from inside their directories.

### Why not use SchemaRAG's existing scripts directly

SchemaRAG provides reference implementations that were helpful for understanding the architecture. However, direct reuse was not possible for several reasons:

1. **Path assumptions**: SchemaRAG's scripts hardcode paths like `./data/database/` that assume you are running from the SchemaRAG directory
2. **Different purpose**: `BM25s_constrcut_db.py` runs BM25 at query time; we need a build-time caching version
3. **`script_to_RAG.py` generates new pairs**: We need structural annotation of existing pairs, not LLM-based augmentation
4. **NoSQL extension**: SchemaRAG is SQL-only; we needed to extend every component to MongoDB
5. **Parser choice**: We use `sqlglot` instead of `sqlparse` for better coverage of Spider's SQL

---

## 10. Data Flow — End to End

This shows how data produced in each phase feeds into later phases.

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
    ├──► SQL RAG Corpus (Phase 7A)
    │         │
    │         └──► spider_sql_rag.json
    │                   (7000 Q-SQL pairs with structural types)
    │
    └──► NoSQL RAG Corpus (Phase 7B) [PENDING]
              │
              └──► spider_nosql_rag.json
                    (Q-MQL pairs from DeepSeek translation)


─── TRAINING PHASES (Colab) ───────────────────────────────────

CoT Data (Phase 8) ──► SchemaLinker SFT (Phase 9)
                                │
                          SchemaLinker MTL (Phase 10)
                                │
                          SchemaLinker GRPO (Phase 11)
                                │
                          SchemaLinker checkpoints
                                │
                                ▼
SQL RAG Corpus ──► SAR Training (Phase 12) ──► SAR checkpoints
                                │
                          ChromaDB Index (Phase 13)

─── FINE-TUNING ────────────────────────────────────────────────

PromptSchema + SchemaLinker + SAR ──► Generator Fine-tuning (Phase 14)
                                                │
                                        Generator checkpoints

─── INFERENCE PIPELINE ─────────────────────────────────────────

Question
   │
   ├─ PromptSchema (pre-cached JSON)
   ├─ SchemaLinker (loaded checkpoint)
   ├─ SAR → ChromaDB query → top-3 examples
   ├─ Generator → 5 candidates
   └─ POSG → select best → final query
```

---

## 11. File and Folder Structure

```
Codegen/
├── src/                          ← reusable library code (imported by other files)
│   ├── device.py                 ← MPS/CUDA/CPU detection (Phase 3D)
│   ├── fk_graph.py               ← FK graph builder for all 166 dbs (Phase 5A)
│   ├── mongodb_converter.py      ← SQLite → MongoDB converter (Phase 5B)
│   ├── prompt_schema.py          ← BM25S column annotation (Phase 6)
│   ├── schema_linker/            ← 3-stage SchemaLinker (Phases 9–11, pending)
│   ├── sar/                      ← Schema-Augmented Retriever (Phase 12, pending)
│   ├── generator/                ← Query generator (Phase 14, pending)
│   ├── posg/                     ← Pareto selection (Phase 15, pending)
│   └── router/                   ← LangGraph router (Phase 17, pending)
│
├── scripts/                      ← one-off build/validation scripts
│   ├── validate_spider.py        ← verify Spider download (Phase 4)
│   ├── Validate_sql2mongo_conversion.py ← verify MongoDB conversion (Phase 5B)
│   └── build_rag_corpus.py       ← build SQL RAG corpus (Phase 7A)
│
├── Data/                         ← all data (gitignored)
│   ├── Spider/                   ← Spider dataset
│   │   ├── train_spider.json     ← 7000 Q-SQL training pairs
│   │   ├── dev.json              ← 1034 Q-SQL evaluation pairs
│   │   ├── tables.json           ← schema metadata
│   │   └── database/             ← 166 SQLite files
│   ├── fk_graphs/                ← FK graphs (166 JSON files) — Phase 5A output
│   ├── mongodb/                  ← MongoDB schema cache (166 JSON files) — Phase 5B output
│   ├── prompt_schema/
│   │   ├── sql/                  ← SQL column annotations (166 JSON files) — Phase 6 output
│   │   └── nosql/                ← NoSQL field annotations (166 JSON files) — Phase 6 output
│   ├── rag_corpus/
│   │   └── spider_sql_rag.json   ← 7000 annotated Q-SQL pairs — Phase 7A output
│   └── cot_data/                 ← CoT training data (Phase 8, pending)
│
├── configs/
│   └── config.yaml               ← all paths and hyperparameters
│
├── external/
│   └── SchemaRAG/                ← reference implementation (gitignored)
│
├── models/                       ← trained checkpoints (gitignored)
├── indexes/                      ← ChromaDB vector stores (gitignored)
├── evaluation/                   ← eval scripts and results
└── docs/
    └── architecture.md           ← this document
```

---

*Last updated: Phase 7A complete. Next update will cover Phase 7B (NoSQL RAG Corpus) and Phase 8 (CoT Data Generation).*
