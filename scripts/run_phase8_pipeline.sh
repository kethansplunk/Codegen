#!/bin/bash
# Runs Phase 8A (if not done), then Phase 8B automatically.
# Safe to run while 8A is already in progress in another terminal —
# this script just waits for sql_cot_train.json to appear, then kicks off 8B.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SQL_OUTPUT="$ROOT/Data/cot_data/sql_cot_train.json"
NOSQL_OUTPUT="$ROOT/Data/cot_data/nosql_cot_train.json"

cd "$ROOT"

# ── Phase 8A ────────────────────────────────────────────────────────────────

if [ -f "$SQL_OUTPUT" ]; then
    echo "[Phase 8A] Already complete: $SQL_OUTPUT"
else
    # Check if 8A is already running in another terminal
    if pgrep -f "build_cot_data.py" > /dev/null 2>&1; then
        echo "[Phase 8A] Detected running in another terminal — waiting for it to finish..."
        echo "           Polling every 60s. Press Ctrl+C to cancel."
        while [ ! -f "$SQL_OUTPUT" ]; do
            sleep 60
            if [ -f "$ROOT/Data/cot_data/cot_checkpoint.json" ]; then
                IDX=$(python -c "import json; d=json.load(open('$ROOT/Data/cot_data/cot_checkpoint.json')); print(d['next_idx'])" 2>/dev/null || echo "?")
                echo "  [$(date +%H:%M)] 8A progress: next_idx=$IDX / 7000"
            fi
        done
    else
        echo "[Phase 8A] Starting..."
        python scripts/build_cot_data.py
    fi
    echo "[Phase 8A] Done → $SQL_OUTPUT"
fi

# ── Verify 8A output ────────────────────────────────────────────────────────

echo ""
echo "[Verify 8A] Inspecting sql_cot_train.json..."
python -c "
import json, sys
data = json.load(open('$SQL_OUTPUT'))
print(f'  Total entries : {len(data)}')
print(f'  Sample keys   : {list(data[0].keys())}')
print(f'  Sample question: {data[0][\"question\"]}')
print(f'  Sample key_fields: {data[0][\"key_fields\"]}')
if len(data) < 5000:
    print('  WARNING: fewer than 5000 entries — check format/entity fail rates', file=sys.stderr)
"
echo ""

# ── Phase 8B ────────────────────────────────────────────────────────────────

if [ -f "$NOSQL_OUTPUT" ]; then
    echo "[Phase 8B] Already complete: $NOSQL_OUTPUT"
else
    echo "[Phase 8B] Starting..."
    python scripts/build_nosql_cot_data.py
    echo "[Phase 8B] Done → $NOSQL_OUTPUT"
fi

# ── Validate 8B ─────────────────────────────────────────────────────────────

echo ""
echo "[Validate] Running Phase 8B validation..."
python scripts/validate_nosql_cot.py
