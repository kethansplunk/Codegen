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

**Zero-shot (`--k_shot 0`, the default) measured 0.0% EX** in practice, well
below the plan's ~45-55% estimate. Manual inspection of raw completions showed
genuinely incoherent output (hallucinated tables, malformed nested aggregates,
leaked Python-style triple-quote markers) rather than a prompt/extraction bug --
this small, non-instruction-tuned, non-SQL-specialized model appears to need
in-context examples to produce syntactically valid SQL at all.

    # Few-shot: prepend k worked examples (from the Spider *train* split, so
    # no overlap with a held-out --data set) before the target question.
    python -m scripts.run_baseline --data Data/cot_data/sql_dev_eval_full.json --n 100 --k_shot 3

`_pick_fewshot_examples()` deliberately picks structurally *diverse* exemplars
(a plain COUNT, a JOIN, an ORDER BY/GROUP BY) rather than k similar ones --
early manual testing with same-shape few-shot examples showed the model
pattern-matching the exemplars' surface structure too literally (e.g. copying
a `WHERE x = y` clause into every completion regardless of the question).
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
MODEL_MAX_CONTEXT = 2048   # codegen-350M's n_ctx; input + max_new_tokens must fit within this


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_block(schema: str, question: str, sql: str = None) -> str:
    """One schema/question/[answer] comment block, shared by exemplars and the target."""
    schema_lines = "\n".join(f"-- {line}" for line in schema.strip().splitlines())
    block = (
        "-- Database schema:\n"
        f"{schema_lines}\n"
        "--\n"
        f"-- Question: {question.strip()}\n"
        "-- SQL query:\n"
        "SELECT "
    )
    if sql is not None:
        answer = sql.strip()
        if answer.upper().startswith("SELECT"):
            answer = answer[len("SELECT"):].strip()
        block += f"{answer.rstrip(';')};\n\n"
    return block


def build_prompt(schema: str, question: str, fewshot: list = None) -> str:
    """Completion-style prompt: optional worked examples, then an open SELECT to continue."""
    prefix = "".join(_format_block(ex["schema"], ex["question"], ex["sql"]) for ex in (fewshot or []))
    return prefix + _format_block(schema, question)


def _pick_fewshot_examples(train_path: str, k: int) -> list:
    """
    k structurally *diverse* exemplars from the Spider train split -- deliberately
    not k similar ones, since a same-shape few-shot set was observed to make the
    model copy the exemplars' surface structure rather than reason about the
    actual question. Picks the first short (<100 char SQL) match for each shape
    in turn, so the result is fixed and reproducible, not randomly sampled.
    """
    if k <= 0:
        return []
    with open(train_path, encoding="utf-8") as f:
        train = json.load(f)

    shapes = [
        lambda s: "JOIN" not in s.upper() and "GROUP BY" not in s.upper() and "ORDER BY" not in s.upper(),
        lambda s: "JOIN" in s.upper(),
        lambda s: "ORDER BY" in s.upper() or "GROUP BY" in s.upper(),
    ]
    picked, used_db = [], set()
    for shape in shapes[:k]:
        for e in train:
            sql = e.get("sql", "")
            if (shape(sql) and len(sql) < 100 and e["db_name"] not in used_db
                    and e.get("schema")):
                picked.append(e)
                used_db.add(e["db_name"])
                break
    return picked[:k]


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
    def __init__(self, model_name: str = DEFAULT_MODEL, max_new_tokens: int = 128,
                 fewshot: list = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.device import get_device

        self.device = get_device()
        self.max_new_tokens = max_new_tokens
        self.fewshot = fewshot or []
        print(f"Loading {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # If the prompt (few-shot exemplars + target) is too long, truncate from the
        # LEFT (drop the earliest exemplars) rather than the default right-truncation,
        # which silently cuts off the target question itself -- confirmed to happen in
        # practice: a 3-shot prompt exceeded max_length and the model was never shown
        # the real question at all, so it just kept extrapolating the last exemplar.
        self.tokenizer.truncation_side = "left"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, schema: str, question: str) -> str:
        import torch

        prompt = build_prompt(schema, question, self.fewshot)
        input_max_length = MODEL_MAX_CONTEXT - self.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=input_max_length).to(self.device)
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
    parser.add_argument("--k_shot", type=int, default=0,
                        help="Prepend k structurally-diverse worked examples from "
                             "--fewshot_data before the target question. Default 0 "
                             "(zero-shot) measured 0.0%% EX -- see module docstring.")
    parser.add_argument("--fewshot_data", default="Data/cot_data/sql_cot_train.json",
                        help="Spider train-split CoT file to draw exemplars from "
                             "(disjoint from a held-out --data eval set).")
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        entries = json.load(f)
    if args.n and args.n < len(entries):
        random.seed(args.seed)
        entries = random.sample(entries, args.n)

    fewshot = _pick_fewshot_examples(args.fewshot_data, args.k_shot)
    if args.k_shot and not args.quiet:
        print(f"Few-shot: {len(fewshot)}/{args.k_shot} exemplars picked "
              f"from {args.fewshot_data}:")
        for ex in fewshot:
            print(f"  [{ex['db_name']}] {ex['question']} -> {ex['sql']}")

    config = _load_config(args.config)
    scorer = _SqlScorer(config)
    model  = BaselineModel(args.model, max_new_tokens=args.max_new_tokens, fewshot=fewshot)

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

    print(f"\n=== CP1 baseline — {args.model} — k_shot={args.k_shot} — {len(results)} questions ===")
    ex = f"{summary['ex']:.1%}" if summary["ex"] is not None else "n/a"
    print(f"EX          : {ex}  (scored on {summary['n_scored']}/{summary['n_total']})")
    print(f"exact-match : {summary['exact_match']:.1%}")
    if summary["n_unscorable"]:
        print(f"[!] {summary['n_unscorable']} questions excluded from EX "
              f"(gold query did not execute — usually a missing local database).")
    if summary["ex"] is not None:
        print(f"\nPlan expects roughly 45–55% here. The full pipeline targets >82% "
              f"(run scripts/run_eval.py on the same --data for a paired comparison).")

    report = {"track": "sql", "strategy": f"baseline:{args.model}:k_shot={args.k_shot}",
              "n_questions": len(results), "ablations": ["baseline"],
              "results": results, "summary": {"baseline": summary}}
    out = args.out or f"evaluation/results/cp1_baseline_{datetime.now():%Y%m%d_%H%M%S}.json"
    print(f"\nFull per-question report -> {save_report(report, out)}")


if __name__ == "__main__":
    main()
