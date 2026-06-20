"""
Phase 8A — SQL SchemaLinker CoT Data Generation

Adapted from SchemaRAG's script_to_COT.py:
- Same CoT prompt format and <think> tag structure
- Same 3-step reasoning template
- Same format validation logic
- Entity consistency validated with sqlglot (free) instead of a second LLM call
- DeepSeek-V3 replaces GPT-4o (same capability, ~10x cheaper)
- Schema formatted from our FK graphs + PromptSchema outputs
"""

import json
import os
import re
import time
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from sqlglot import exp, parse_one

load_dotenv()

DEEPSEEK_CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

BASE           = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH     = os.path.join(BASE, "Data", "Spider", "train_spider.json")
SQL_CORPUS     = os.path.join(BASE, "Data", "rag_corpus", "spider_sql_rag.json")
FK_GRAPH_DIR   = os.path.join(BASE, "Data", "fk_graphs")
PS_DIR         = os.path.join(BASE, "Data", "prompt_schema", "sql")
OUTPUT_PATH    = os.path.join(BASE, "Data", "cot_data", "sql_cot_train.json")
CHECKPOINT     = os.path.join(BASE, "Data", "cot_data", "cot_checkpoint.json")


# ---------------------------------------------------------------------------
# Schema formatting
# ---------------------------------------------------------------------------

def format_schema_text(db_name: str) -> str:
    """
    Build a schema text string from FK graph + PromptSchema.

    Output format (matches SchemaRAG's RAG_Spider.json):
        # Table: actor
        [
        (actor_id:INT, Primary Key, Examples: [1, 2]),
        (name:TEXT, Examples: [Tom Hanks, Meryl Streep]),
        ]
        # Foreign Keys:
        # actor_in_movie.actor_id -> actor.actor_id
    """
    fk_path = os.path.join(FK_GRAPH_DIR, f"{db_name}.json")
    ps_path  = os.path.join(PS_DIR, f"{db_name}.json")

    if not os.path.exists(fk_path) or not os.path.exists(ps_path):
        return ""

    fk_data = json.load(open(fk_path))
    ps_data  = json.load(open(ps_path))

    pk_map: Dict[str, Set[str]] = {}
    for node in fk_data["nodes"]:
        pk_map[node["name"]] = set(node.get("pk", []))

    lines = []
    for node in fk_data["nodes"]:
        table = node["name"]
        lines.append(f"# Table: {table}")
        lines.append("[")
        for col in node["columns"]:
            col_name = col["name"]
            col_type = col["type"]
            ps_key   = f"{table}.{col_name}"
            parts    = [f"{col_name}:{col_type}"]

            if col_name in pk_map[table]:
                parts.append("Primary Key")

            if ps_key in ps_data:
                samples = ps_data[ps_key].get("sample_values", [])
                if samples:
                    formatted = ", ".join(str(v) for v in samples[:3])
                    parts.append(f"Examples: [{formatted}]")

            lines.append(f"({', '.join(parts)}),")
        lines.append("]")
        lines.append("")

    if fk_data["edges"]:
        lines.append("# Foreign Keys:")
        for edge in fk_data["edges"]:
            lines.append(
                f"# {edge['from']}.{edge['child_col']} -> "
                f"{edge['to']}.{edge['parent_col']}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CoT generation
# ---------------------------------------------------------------------------

COT_PROMPT_TEMPLATE = """You are an expert in database schema linking for Text-to-SQL tasks.

Given a natural language question and a database schema, identify which tables and columns are needed to answer the question.

**Database Schema:**
{schema}

**Question:** {question}

**Ground Truth SQL (for reference):**
{sql}

Please provide your reasoning in the following format:

<think>
1. Understand the key concepts in the question:
   • [Identify key phrases in the question]
   • [Map them to what they mean in database terms]
   • [Note what operations are required]

2. Analyze database table relationships:
   • [Identify which tables contain relevant information]
   • [Explain the relationships between tables using foreign keys]
   • [Note how tables need to be joined]

3. Key field for filtering: **actual_table.actual_column** (explain why this field is critical)
   Additional explanation of why this field is most relevant.
</think>

Provide a summary paragraph explaining the reasoning, emphasizing the most critical field(s).

IMPORTANT: You MUST end your response with EXACTLY this line (keep the square brackets):
The key field matching the question is: [actual_table.actual_column]."""


def call_deepseek(question: str, sql: str, schema_text: str) -> Dict:
    prompt = COT_PROMPT_TEMPLATE.format(
        schema=schema_text, question=question, sql=sql
    )
    try:
        response = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a database schema linking expert. Follow the output format exactly as specified.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1024,
        )
        return {"success": True, "cot": response.choices[0].message.content}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Format validation  (from SchemaRAG script_to_COT.py)
# ---------------------------------------------------------------------------

def validate_format(cot: str) -> Tuple[bool, str]:
    cot = cot.strip()

    if "<think>" not in cot:
        return False, "Missing <think> tag"
    if "</think>" not in cot:
        return False, "Missing </think> tag"

    think_start = cot.find("<think>")
    think_end   = cot.find("</think>")
    if think_start >= think_end:
        return False, "<think> and </think> tags in wrong order"

    if not re.search(r"1\.\s*Understand the key concepts", cot, re.IGNORECASE):
        return False, "Missing Step 1"
    if not re.search(r"2\.\s*Analyze database table relationships", cot, re.IGNORECASE):
        return False, "Missing Step 2"
    if not re.search(r"3\.\s*Key field for filtering", cot, re.IGNORECASE):
        return False, "Missing Step 3"

    # Accept  [table.col]  or  table.col.  — anchored to end of line so validation
    # and extraction patterns agree (avoids entity_fail miscounting format errors)
    pattern = r"The\s+key\s+field\s+matching\s+the\s+question\s+is:\s*\[?([\w.,\s]+?)\]?\.?\s*$"
    if not re.search(pattern, cot, re.IGNORECASE | re.MULTILINE):
        return False, "Missing final key field declaration"

    return True, "ok"


# ---------------------------------------------------------------------------
# Entity consistency validation using sqlglot (no extra LLM call)
# ---------------------------------------------------------------------------

def extract_sql_tables(sql: str) -> Set[str]:
    try:
        ast = parse_one(sql, dialect="sqlite")
        cte_names = {cte.alias.lower() for cte in ast.find_all(exp.CTE) if cte.alias}
        return {
            t.name.lower()
            for t in ast.find_all(exp.Table)
            if t.name and t.name.lower() not in cte_names
        }
    except Exception:
        return set()


def extract_cot_key_tables(cot: str) -> Set[str]:
    pattern = r"The\s+key\s+field\s+matching\s+the\s+question\s+is:\s*\[?([\w.,\s]+?)\]?\.?\s*$"
    matches = re.findall(pattern, cot, re.IGNORECASE | re.MULTILINE)
    tables = set()
    for m in matches:
        for field in m.split(","):
            field = field.strip()
            if "." in field:
                tables.add(field.split(".")[0].lower())
    return tables


def validate_entities(cot: str, sql: str) -> bool:
    sql_tables  = extract_sql_tables(sql)
    cot_tables  = extract_cot_key_tables(cot)
    if not cot_tables:
        return False
    # Every table the CoT claims as key must actually appear in the SQL
    return cot_tables.issubset(sql_tables)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_cot_data():
    # Load Spider train set (has question, query, db_id)
    train_data = json.load(open(TRAIN_PATH))

    # Load structural types from Phase 7A to sort by complexity
    sql_corpus = json.load(open(SQL_CORPUS))
    complexity_map = {
        (e["question"], e["db_name"]): e["structural_type"]
        for e in sql_corpus
    }

    # Sort simplest first
    train_data.sort(key=lambda e: (
        complexity_map.get((e["question"], e["db_id"]), {}).get("num_tables", 99),
        complexity_map.get((e["question"], e["db_id"]), {}).get("num_joins",  99),
    ))

    # Resume from checkpoint
    start_idx = 0
    corpus    = []
    if os.path.exists(CHECKPOINT):
        ckpt      = json.load(open(CHECKPOINT))
        corpus    = ckpt["corpus"]
        start_idx = ckpt["next_idx"]
        print(f"Resuming from index {start_idx}, {len(corpus)} entries so far")

    stats = {
        "total": len(train_data), "attempted": 0,
        "generated": 0, "format_fail": 0,
        "entity_fail": 0, "verified": 0,
    }

    for i, entry in enumerate(train_data[start_idx:], start=start_idx):
        question = entry["question"]
        sql      = entry["query"]
        db_name  = entry["db_id"]

        schema_text = format_schema_text(db_name)
        if not schema_text:
            stats["format_fail"] += 1
            continue

        stats["attempted"] += 1

        result = call_deepseek(question, sql, schema_text)
        if not result["success"]:
            stats["format_fail"] += 1
            time.sleep(1)
            continue

        stats["generated"] += 1
        cot = result["cot"]

        fmt_ok, fmt_msg = validate_format(cot)
        if not fmt_ok:
            stats["format_fail"] += 1
            time.sleep(0.1)
            continue

        if not validate_entities(cot, sql):
            stats["entity_fail"] += 1
            time.sleep(0.1)
            continue

        stats["verified"] += 1
        key_fields = re.findall(
            r"The\s+key\s+field\s+matching\s+the\s+question\s+is:\s*\[?([\w.,\s]+?)\]?\.?\s*$",
            cot, re.IGNORECASE | re.MULTILINE,
        )
        corpus.append({
            "question":   question,
            "sql":        sql,
            "db_name":    db_name,
            "schema":     schema_text,
            "cot":        cot,
            "key_fields": [kf.strip() for kf in (key_fields[-1].split(",") if key_fields else [])],
        })

        if (i + 1) % 50 == 0:
            pct = stats["verified"] / max(stats["attempted"], 1) * 100
            print(
                f"[{i+1}/{stats['total']}] verified={stats['verified']} "
                f"fmt_fail={stats['format_fail']} entity_fail={stats['entity_fail']} "
                f"({pct:.1f}%)"
            )
            json.dump({"corpus": corpus, "next_idx": i + 1}, open(CHECKPOINT, "w"), indent=2)

        time.sleep(0.1)

    json.dump(corpus, open(OUTPUT_PATH, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"Total entries  : {stats['total']}")
    print(f"Attempted      : {stats['attempted']}")
    print(f"Generated      : {stats['generated']}")
    print(f"Format failed  : {stats['format_fail']}")
    print(f"Entity failed  : {stats['entity_fail']}")
    print(f"Verified saved : {stats['verified']}")
    if stats["attempted"]:
        print(f"Success rate   : {stats['verified']/stats['attempted']*100:.1f}%")
    print(f"Output         : {OUTPUT_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    build_cot_data()
