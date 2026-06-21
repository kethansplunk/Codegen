"""
Phase 8B validation — nosql_cot_train.json

Checks:
  1. File exists and is valid JSON
  2. Entry count and key schema
  3. CoT format (re-runs validate_format on every entry)
  4. Entity consistency (re-runs validate_entities on every entry)
  5. key_fields not empty and in collection.field format
  6. MQL structure (collection + pipeline present)
  7. DB coverage (how many unique databases)
  8. Pipeline stage type distribution
  9. Sample entry print
"""

import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple

BASE        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INPUT_PATH  = os.path.join(BASE, "Data", "cot_data", "nosql_cot_train.json")
CKPT_PATH   = os.path.join(BASE, "Data", "cot_data", "nosql_cot_checkpoint.json")

# ── reuse validators from the generator script ──────────────────────────────

def validate_format(cot: str) -> Tuple[bool, str]:
    cot = cot.strip()
    if "<think>" not in cot:
        return False, "Missing <think> tag"
    if "</think>" not in cot:
        return False, "Missing </think> tag"
    if cot.find("<think>") >= cot.find("</think>"):
        return False, "<think></think> wrong order"
    if not re.search(r"1\.\s*Understand the key concepts", cot, re.IGNORECASE):
        return False, "Missing Step 1"
    if not re.search(r"2\.\s*Analyze MongoDB collection relationships", cot, re.IGNORECASE):
        return False, "Missing Step 2"
    if not re.search(r"3\.\s*Key field for filtering", cot, re.IGNORECASE):
        return False, "Missing Step 3"
    pattern = r"The\s+key\s+field\s+matching\s+the\s+question\s+is:\s*\[?([\w.,\s]+?)\]?\.?\s*$"
    if not re.search(pattern, cot, re.IGNORECASE | re.MULTILINE):
        return False, "Missing final key field line"
    return True, "ok"


def extract_mql_collections(entry: Dict) -> Set[str]:
    mql = entry.get("mql", {})
    collections = {mql.get("collection", "").lower()}

    def _walk(pipeline: List[Dict]) -> None:
        for stage in pipeline:
            if "$lookup" in stage:
                lk = stage["$lookup"]
                if "from" in lk:
                    collections.add(lk["from"].lower())
                if "pipeline" in lk:
                    _walk(lk["pipeline"])
            if "$unionWith" in stage:
                uw = stage["$unionWith"]
                if isinstance(uw, str):
                    collections.add(uw.lower())
                elif isinstance(uw, dict):
                    if "coll" in uw:
                        collections.add(uw["coll"].lower())
                    if "pipeline" in uw:
                        _walk(uw["pipeline"])

    _walk(mql.get("pipeline", []))
    return collections


def extract_cot_key_collections(cot: str) -> Set[str]:
    pattern = r"The\s+key\s+field\s+matching\s+the\s+question\s+is:\s*\[?([\w.,\s]+?)\]?\.?\s*$"
    matches = re.findall(pattern, cot, re.IGNORECASE | re.MULTILINE)
    result = set()
    for m in matches:
        for field in m.split(","):
            field = field.strip()
            if "." in field:
                result.add(field.split(".")[0].lower())
    return result


# ── helpers ──────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"question", "mql", "db_name", "schema", "cot", "key_fields", "source_sql"}
KF_PATTERN    = re.compile(r"^[\w]+\.[\w]+$")


def check_key_fields(kf_list: List[str]) -> Tuple[bool, str]:
    if not kf_list:
        return False, "empty"
    for kf in kf_list:
        if not KF_PATTERN.match(kf.strip()):
            return False, f"bad format: '{kf}'"
    return True, "ok"


def pipeline_stage_types(pipeline: List[Dict]) -> List[str]:
    types = []
    for stage in pipeline:
        types.extend(stage.keys())
    return types


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # check for in-progress checkpoint if final file not ready
    source = INPUT_PATH
    if not os.path.exists(INPUT_PATH):
        if os.path.exists(CKPT_PATH):
            print(f"Final file not found — validating checkpoint instead: {CKPT_PATH}")
            raw = json.load(open(CKPT_PATH))
            data = raw["corpus"]
            print(f"Checkpoint progress: next_idx={raw['next_idx']} / total in corpus={len(data)}\n")
            source = CKPT_PATH
        else:
            print("ERROR: nosql_cot_train.json not found and no checkpoint exists.")
            print("Run scripts/build_nosql_cot_data.py first.")
            sys.exit(1)
    else:
        data = json.load(open(INPUT_PATH))

    total = len(data)
    print(f"{'='*60}")
    print(f"  Phase 8B Validation — {os.path.basename(source)}")
    print(f"  Entries to validate: {total}")
    print(f"{'='*60}\n")

    # ── 1. Required keys ────────────────────────────────────────────────────
    missing_keys = [i for i, e in enumerate(data) if not REQUIRED_KEYS.issubset(e.keys())]
    status = "✅" if not missing_keys else f"❌ {len(missing_keys)} entries"
    print(f"[1] Required keys present          {status}")
    if missing_keys[:3]:
        for i in missing_keys[:3]:
            print(f"    entry {i}: has {set(data[i].keys())}, missing {REQUIRED_KEYS - set(data[i].keys())}")

    # ── 2. MQL structure ────────────────────────────────────────────────────
    bad_mql = [i for i, e in enumerate(data)
               if not isinstance(e.get("mql"), dict)
               or "collection" not in e.get("mql", {})
               or "pipeline" not in e.get("mql", {})]
    status = "✅" if not bad_mql else f"❌ {len(bad_mql)} entries"
    print(f"[2] MQL has collection + pipeline  {status}")

    # ── 3. CoT format ───────────────────────────────────────────────────────
    fmt_fails = []
    for i, e in enumerate(data):
        ok, msg = validate_format(e.get("cot", ""))
        if not ok:
            fmt_fails.append((i, msg))
    status = "✅" if not fmt_fails else f"❌ {len(fmt_fails)} entries"
    print(f"[3] CoT format valid               {status}")
    if fmt_fails[:3]:
        for i, msg in fmt_fails[:3]:
            print(f"    entry {i}: {msg}")

    # ── 4. Entity consistency ───────────────────────────────────────────────
    entity_fails = []
    for i, e in enumerate(data):
        mql_cols = extract_mql_collections(e)
        cot_cols = extract_cot_key_collections(e.get("cot", ""))
        if not cot_cols or not cot_cols.issubset(mql_cols):
            entity_fails.append((i, cot_cols, mql_cols))
    status = "✅" if not entity_fails else f"❌ {len(entity_fails)} entries"
    print(f"[4] Entity consistency             {status}")
    if entity_fails[:3]:
        for i, cot_cols, mql_cols in entity_fails[:3]:
            print(f"    entry {i}: CoT={cot_cols}  MQL={mql_cols}")

    # ── 5. key_fields format ────────────────────────────────────────────────
    kf_fails = []
    for i, e in enumerate(data):
        ok, msg = check_key_fields(e.get("key_fields", []))
        if not ok:
            kf_fails.append((i, msg))
    status = "✅" if not kf_fails else f"❌ {len(kf_fails)} entries"
    print(f"[5] key_fields format (col.field)  {status}")
    if kf_fails[:3]:
        for i, msg in kf_fails[:3]:
            print(f"    entry {i}: {msg}")

    # ── 6. DB coverage ──────────────────────────────────────────────────────
    dbs = {e["db_name"] for e in data}
    print(f"[6] DB coverage                    {len(dbs)} / 166 unique databases")

    # ── 7. Pipeline stage distribution ──────────────────────────────────────
    stage_counter: Counter = Counter()
    for e in data:
        pipeline = e.get("mql", {}).get("pipeline", [])
        stage_counter.update(pipeline_stage_types(pipeline))
    print(f"[7] Pipeline stage distribution:")
    for stage, count in stage_counter.most_common(10):
        bar = "█" * (count // (total // 20 or 1))
        print(f"    {stage:<15} {count:>5}  {bar}")

    # ── 8. Summary ──────────────────────────────────────────────────────────
    all_checks = [not missing_keys, not bad_mql, not fmt_fails, not entity_fails, not kf_fails]
    passed = sum(all_checks)
    print(f"\n{'─'*60}")
    print(f"Checks passed: {passed}/5")
    print(f"Total entries: {total}")
    pct_fail = (len(fmt_fails) + len(entity_fails)) / max(total, 1) * 100
    print(f"Post-hoc failure rate: {pct_fail:.1f}% (should be ~0% — all failures filtered at generation)")

    # ── 9. Sample entries ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("SAMPLE ENTRIES\n")
    for i in [0, total // 2, total - 1]:
        if i >= total:
            continue
        e = data[i]
        mql = e.get("mql", {})
        print(f"[Entry {i}]")
        print(f"  question   : {e['question']}")
        print(f"  db_name    : {e['db_name']}")
        print(f"  collection : {mql.get('collection')}")
        print(f"  stages     : {[list(s.keys())[0] for s in mql.get('pipeline', [])]}")
        print(f"  key_fields : {e['key_fields']}")
        cot_preview = e.get("cot", "")[:120].replace("\n", " ")
        print(f"  cot start  : {cot_preview}...")
        print()


if __name__ == "__main__":
    main()
