"""
Phase 17 — LangGraph router session CLI.

Session-based routing (Option A): pick the track once with --track, then ask
questions against it. SchemaLinker / SAR / Generator are built once and reused
for every question in the session, so the 7B model is loaded exactly once no
matter how many questions you ask.

    # single question
    python -m scripts.run_router --track sql --db_name concert_singer \\
        --question "How many singers are there?"

    # interactive session (Ctrl-D or 'quit' to exit)
    python -m scripts.run_router --track sql --db_name concert_singer

    # batch from a CoT/dev-eval JSON file, reporting retry statistics
    python -m scripts.run_router --track sql --batch Data/cot_data/sql_dev_eval_full.json --n 20

The SQL track executes against Data/Spider/database/<db>/<db>.sqlite; if that
file is missing, the query is selected but not executed (reported as
"not executed"). The NoSQL track needs a live mongod, same as Phase 15B/16.
"""

from __future__ import annotations

import argparse
import json
import random

import yaml

from src.router.langgraph_router import MAX_RETRIES, Router


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_result(r: dict, show_rows: int = 5):
    print(f"\n  status:  {r['status']}"
          + (f"   (after {r['retries']} retr{'y' if r['retries'] == 1 else 'ies'})"
             if r["retries"] else ""))
    query = r["query"]
    print(f"  query:   {query if isinstance(query, str) else json.dumps(query, ensure_ascii=False, default=str)}")

    if r["status"] == "ok":
        rows = r["rows"]
        if rows is None:
            print("  rows:    not executed (no local database for this db_name)")
        else:
            print(f"  rows:    {len(rows)}")
            for row in rows[:show_rows]:
                print(f"             {row}")
            if len(rows) > show_rows:
                print(f"             ... {len(rows) - show_rows} more")
    else:
        print(f"  error:   {r['error']}")

    if len(r["history"]) > 1:
        print("  retry history:")
        for h in r["history"]:
            status = "ok" if h["error"] is None else h["error"]
            print(f"    [{h['attempt']}] ({h['source']}) {h['query'][:80]}\n         -> {status}")


def _batch(router: Router, path: str, n: int, seed: int):
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)

    # --n 0 means "whole file", matching run_posg_*.py and run_eval.py. Without
    # this, min(0, len) would sample nothing and every rate below would divide
    # by zero.
    sample = entries if not n or n >= len(entries) else None
    if sample is None:
        random.seed(seed)
        sample = random.sample(entries, n)
    if not sample:
        print(f"[!] {path} contains no entries — nothing to evaluate.")
        return

    results = []
    for i, e in enumerate(sample, 1):
        print(f"\n[{i}/{len(sample)}] {e['question']}  (db={e['db_name']})")
        r = router.run(e["question"], e["db_name"], schema=e.get("schema"))
        _print_result(r)
        results.append(r)

    ok        = [r for r in results if r["status"] == "ok"]
    executed  = [r for r in ok if r["rows"] is not None]
    retried   = [r for r in results if r["retries"] > 0]
    recovered = [r for r in retried if r["status"] == "ok"]
    corrected = [r for r in results
                 if any(h["source"] == "self_correct" for h in r["history"])]

    print(f"\n=== Router batch: {len(results)} questions ===")
    print(f"executed without error : {len(ok)}/{len(results)} "
          f"({len(ok) / len(results):.1%})   [{len(executed)} actually ran against a DB]")
    print(f"needed >=1 retry       : {len(retried)}/{len(results)}"
          + (f"   of which recovered: {len(recovered)}/{len(retried)}" if retried else ""))
    print(f"reached self-correction: {len(corrected)}/{len(results)}")
    # Retries are the whole point of Phase 17 -- if nothing ever fails on the
    # first candidate there is no evidence the ladder works, so say so plainly
    # rather than letting a clean run read as a validated one.
    if not retried:
        print("[!] No question ever failed its first candidate, so the retry ladder "
              "was never exercised. Try --batch on a harder set before concluding "
              "self-correction works.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--track", required=True, choices=["sql", "nosql"],
                        help="Fixed for the session (Option A routing).")
    parser.add_argument("--strategy", default="balanced",
                        choices=["balanced", "schema_priority", "example_priority"])
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES,
                        help="Execution retries after the first attempt. The last "
                             "one is reserved for the generator re-prompt.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds Generator candidate sampling, and --batch sampling.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-node logging.")

    parser.add_argument("--question")
    parser.add_argument("--db_name")
    parser.add_argument("--batch", help="JSON file of entries with question/db_name[/schema].")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of --batch entries to sample. 0 = whole file.")

    args = parser.parse_args()

    if not args.batch and not args.db_name:
        parser.error("--db_name is required unless --batch is set")

    config = _load_config(args.config)
    router = Router(
        track=args.track,
        config=config,
        strategy=args.strategy,
        max_retries=args.max_retries,
        seed=args.seed,
        verbose=not args.quiet,
    )

    try:
        if args.batch:
            _batch(router, args.batch, args.n, args.seed)
        elif args.question:
            r = router.run(args.question, args.db_name)
            _print_result(r)
        else:
            print(f"Interactive session — track={args.track} db={args.db_name}. "
                  f"Blank line, 'quit', or Ctrl-D to exit.")
            while True:
                try:
                    q = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not q or q.lower() in ("quit", "exit"):
                    break
                _print_result(router.run(q, args.db_name))
    finally:
        router.close()


if __name__ == "__main__":
    main()
