"""
Phase 18C — CP1 baseline: codegen-350M EX on Spider dev.

Establishes the "without schema-awareness" floor the full pipeline is measured
against. This is a plain pretrained code LM prompted with the schema and the
question — no SchemaLinker, no SAR retrieval, no POSG, no fine-tuning. The plan
expects roughly 45–55% EX; the gap to the full pipeline's target (>82%) is what
the SchemaRAG architecture is claimed to buy.

    python -m scripts.run_baseline --data Data/cot_data/sql_dev_eval_full.json --n 100

EX is computed by the same scorer the Phase 18 harness uses
(src/eval/harness.py), including the same rule for excluding questions whose
gold query cannot execute locally — so the two numbers are directly comparable.

codegen-350M is not instruction-tuned, so it is prompted in the completion style
these models were trained on (SQL comments, then a bare `SELECT`) rather than
with a chat template. Greedy decoding; the completion is cut at the first
statement boundary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime

import yaml

from src.eval.harness import _SqlScorer, save_report, summarize

DEFAULT_MODEL = "Salesforce/codegen-350M-multi"


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(schema: str, question: str) -> str:
    """Completion-style prompt: comment block, then an open SELECT to continue."""
    schema_lines = "\n".join(f"-- {line}" for line in schema.strip().splitlines())
    return (
        "-- Database schema:\n"
        f"{schema_lines}\n"
        "--\n"
        f"-- Question: {question.strip()}\n"
        "-- SQL query:\n"
        "SELECT "
    )


def extract_sql(completion: str) -> str:
    """
    Trim a raw completion to one statement.

    The model keeps going after the query — more comments, a second SELECT,
    prose — so cut at the first semicolon, blank line, or comment marker.
    """
    sql = "SELECT " + completion.strip()
    sql = sql.split(";")[0]
    sql = re.split(r"\n\s*\n|\n\s*--", sql)[0]
    return " ".join(sql.split())


class BaselineModel:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.device import get_device

        self.device = get_device()
        self.max_new_tokens = max_new_tokens
        print(f"Loading {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, schema: str, question: str) -> str:
        import torch

        prompt = build_prompt(schema, question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=1536).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                      # greedy: this is a floor, not a search
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_sql(completion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", required=True,
                        help="JSON entries with question/db_name/schema/sql. Use a HELD-OUT set.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=100, help="0 = whole file.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        entries = json.load(f)
    if args.n and args.n < len(entries):
        random.seed(args.seed)
        entries = random.sample(entries, args.n)

    config = _load_config(args.config)
    scorer = _SqlScorer(config)
    model  = BaselineModel(args.model, max_new_tokens=args.max_new_tokens)

    results = []
    for i, e in enumerate(entries, 1):
        schema = e.get("schema") or scorer.adapter.format_schema(e["db_name"])
        pred   = model.generate(schema, e["question"])
        scored = scorer.compare(pred, e["sql"], e["db_name"])
        scored["query"] = pred
        if not args.quiet:
            print(f"[{i}/{len(entries)}] {e['question']}")
            print(f"    pred: {pred[:110]}")
            print(f"    ex={scored['ex']}"
                  + (f"  [{scored['pred_error']}]" if scored["pred_error"] else ""))
        results.append({"question": e["question"], "db_name": e["db_name"],
                        "gold": e["sql"], "ablations": {"baseline": scored}})

    summary = summarize(results, ["baseline"], "sql")["baseline"]
    scorer.close()

    print(f"\n=== CP1 baseline — {args.model} — {len(results)} questions ===")
    ex = f"{summary['ex']:.1%}" if summary["ex"] is not None else "n/a"
    print(f"EX          : {ex}  (scored on {summary['n_scored']}/{summary['n_total']})")
    print(f"exact-match : {summary['exact_match']:.1%}")
    if summary["n_unscorable"]:
        print(f"[!] {summary['n_unscorable']} questions excluded from EX "
              f"(gold query did not execute — usually a missing local database).")
    if summary["ex"] is not None:
        print(f"\nPlan expects roughly 45–55% here. The full pipeline targets >82% "
              f"(run scripts/run_eval.py on the same --data for a paired comparison).")

    report = {"track": "sql", "strategy": f"baseline:{args.model}",
              "n_questions": len(results), "ablations": ["baseline"],
              "results": results, "summary": {"baseline": summary}}
    out = args.out or f"evaluation/results/cp1_baseline_{datetime.now():%Y%m%d_%H%M%S}.json"
    print(f"\nFull per-question report -> {save_report(report, out)}")


if __name__ == "__main__":
    main()
