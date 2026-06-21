"""
Phase 8B — NoSQL SchemaLinker CoT Data Generation

Adapted from build_cot_data.py (Phase 8A):
- Same CoT format: <think> tags, 3-step reasoning, key field line
- Same validate_format, checkpoint interval, and main loop structure
- Schema text uses MongoDB terminology (collections → fields, $lookup for joins)
- Entity validation parses MQL pipeline dicts instead of SQL AST (no sqlglot)
- Input: spider_nosql_rag.json (5697 Q-MQL pairs from Phase 7B)
"""

import json
import os
import re
import time
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

BASE         = os.path.join(os.path.dirname(__file__), "..")
NOSQL_CORPUS = os.path.join(BASE, "Data", "rag_corpus", "spider_nosql_rag.json")
FK_GRAPH_DIR = os.path.join(BASE, "Data", "fk_graphs")
PS_DIR       = os.path.join(BASE, "Data", "prompt_schema", "nosql")
OUTPUT_PATH  = os.path.join(BASE, "Data", "cot_data", "nosql_cot_train.json")
CHECKPOINT   = os.path.join(BASE, "Data", "cot_data", "nosql_cot_checkpoint.json")


# ---------------------------------------------------------------------------
# Schema formatting (adapted from 8A — "Table" → "Collection", same FK graph)
# ---------------------------------------------------------------------------

def format_schema_text(db_name: str) -> str:
    """
    Build a MongoDB schema text string from FK graph + NoSQL PromptSchema.

    Output format:
        # Collection: actor
        [
        (actor_id:INT, Primary Key, Examples: [1, 2]),
        (name:TEXT, Examples: [Tom Hanks, Meryl Streep]),
        ]
        # Relationships (via $lookup):
        # actor_in_movie.actor_id → actor.actor_id
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
        collection = node["name"]
        lines.append(f"# Collection: {collection}")
        lines.append("[")
        for col in node["columns"]:
            field_name = col["name"]
            field_type = col["type"]
            ps_key     = f"{collection}.{field_name}"
            parts      = [f"{field_name}:{field_type}"]

            if field_name in pk_map[collection]:
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
        lines.append("# Relationships (via $lookup):")
        for edge in fk_data["edges"]:
            lines.append(
                f"# {edge['from']}.{edge['child_col']} → "
                f"{edge['to']}.{edge['parent_col']}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MQL formatting
# ---------------------------------------------------------------------------

def format_mql(collection: str, pipeline: List[Dict]) -> str:
    return f"db.{collection}.aggregate({json.dumps(pipeline, indent=2)})"


# ---------------------------------------------------------------------------
# CoT generation
# ---------------------------------------------------------------------------

COT_PROMPT_TEMPLATE = """You are an expert in MongoDB schema linking for Text-to-NoSQL tasks.

Given a natural language question and a MongoDB database schema, identify which collections and fields are needed to answer the question.

**Database Schema:**
{schema}

**Question:** {question}

**Ground Truth MQL (for reference):**
{mql}

Please provide your reasoning in the following format:

<think>
1. Understand the key concepts in the question:
   • [Identify key phrases in the question]
   • [Map them to what they mean in MongoDB terms]
   • [Note what pipeline operations are required ($match, $group, $lookup, etc.)]

2. Analyze MongoDB collection relationships:
   • [Identify which collections contain relevant information]
   • [Explain relationships between collections via $lookup]
   • [Note how collections are joined and what fields link them]

3. Key field for filtering: **actual_collection.actual_field** (explain why this field is critical)
   Additional explanation of why this field is most relevant.
</think>

Provide a summary paragraph explaining the reasoning, emphasizing the most critical field(s).

IMPORTANT: You MUST end your response with EXACTLY this line (keep the square brackets):
The key field matching the question is: [actual_collection.actual_field]."""


def call_deepseek(question: str, mql: str, schema_text: str) -> Dict:
    prompt = COT_PROMPT_TEMPLATE.format(
        schema=schema_text, question=question, mql=mql
    )
    try:
        response = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a MongoDB schema linking expert. Follow the output format exactly as specified.",
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
# Format validation — copied verbatim from Phase 8A, same CoT format contract
# Step 2 text updated to match the NoSQL prompt wording
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
    if not re.search(r"2\.\s*Analyze MongoDB collection relationships", cot, re.IGNORECASE):
        return False, "Missing Step 2"
    if not re.search(r"3\.\s*Key field for filtering", cot, re.IGNORECASE):
        return False, "Missing Step 3"

    # Same anchored pattern as Phase 8A — brackets optional, anchored to EOL
    pattern = r"The\s+key\s+field\s+matching\s+the\s+question\s+is:\s*\[?([\w.,\s]+?)\]?\.?\s*$"
    if not re.search(pattern, cot, re.IGNORECASE | re.MULTILINE):
        return False, "Missing final key field declaration"

    return True, "ok"


# ---------------------------------------------------------------------------
# Entity validation — MQL pipeline parser (replaces sqlglot from Phase 8A)
# Covers: base collection, $lookup.from, nested $lookup, $unionWith
# ---------------------------------------------------------------------------

def extract_mql_collections(entry: Dict) -> Set[str]:
    """
    Extract all collection names referenced in the MQL pipeline.
    Handles $lookup (including nested pipelines) and $unionWith.
    """
    collections = {entry["mql_collection"].lower()}

    def _walk_pipeline(pipeline: List[Dict]) -> None:
        for stage in pipeline:
            if "$lookup" in stage:
                lk = stage["$lookup"]
                if "from" in lk:
                    collections.add(lk["from"].lower())
                # nested pipeline inside $lookup
                if "pipeline" in lk:
                    _walk_pipeline(lk["pipeline"])
            if "$unionWith" in stage:
                uw = stage["$unionWith"]
                if isinstance(uw, str):
                    collections.add(uw.lower())
                elif isinstance(uw, dict) and "coll" in uw:
                    collections.add(uw["coll"].lower())
                    if "pipeline" in uw:
                        _walk_pipeline(uw["pipeline"])

    _walk_pipeline(entry.get("mql_pipeline", []))
    return collections


def extract_cot_key_collections(cot: str) -> Set[str]:
    """Extract collection names from the CoT's final key field declaration."""
    pattern = r"The\s+key\s+field\s+matching\s+the\s+question\s+is:\s*\[?([\w.,\s]+?)\]?\.?\s*$"
    matches = re.findall(pattern, cot, re.IGNORECASE | re.MULTILINE)
    collections = set()
    for m in matches:
        for field in m.split(","):
            field = field.strip()
            if "." in field:
                collections.add(field.split(".")[0].lower())
    return collections


def validate_entities(cot: str, entry: Dict) -> bool:
    mql_collections = extract_mql_collections(entry)
    cot_collections = extract_cot_key_collections(cot)
    if not cot_collections:
        return False
    return cot_collections.issubset(mql_collections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_nosql_cot_data():
    corpus_data = json.load(open(NOSQL_CORPUS))

    # Sort simplest first — same strategy as Phase 8A
    corpus_data.sort(key=lambda e: (
        e["structural_type"].get("num_tables", 99),
        e["structural_type"].get("num_joins",  99),
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
        "total": len(corpus_data), "attempted": 0,
        "generated": 0, "format_fail": 0,
        "entity_fail": 0, "verified": 0,
    }

    for i, entry in enumerate(corpus_data[start_idx:], start=start_idx):
        question   = entry["question"]
        collection = entry["mql_collection"]
        pipeline   = entry["mql_pipeline"]
        db_name    = entry["db_name"]

        schema_text = format_schema_text(db_name)
        if not schema_text:
            stats["format_fail"] += 1
            continue

        stats["attempted"] += 1
        mql_str = format_mql(collection, pipeline)

        result = call_deepseek(question, mql_str, schema_text)
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

        if not validate_entities(cot, entry):
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
            "mql":        {"collection": collection, "pipeline": pipeline},
            "db_name":    db_name,
            "schema":     schema_text,
            "cot":        cot,
            "key_fields": [kf.strip() for kf in (key_fields[-1].split(",") if key_fields else [])],
            "source_sql": entry.get("source_sql", ""),
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
    build_nosql_cot_data()
