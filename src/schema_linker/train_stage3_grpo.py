"""
SchemaLinker Stage 3 — GRPO Reinforcement Learning.
Adapted from SchemaRAG train_SchemaLinker_GRPO_peft.py:
- Reward function: format check + TP/FP/FN scoring on schema link sets.
- RewardConfig: TP=+2, FP=-0.5, FN=-3, F1 bonus=+0.5, format_fail=-1000.
- modelscope → transformers; uses TRL GRPOTrainer.
- Input data: same as Stage 2 (question, database, key_fields as true_links).

Run on Colab A100 (GRPO needs more memory than T4):
    python -m src.schema_linker.train_stage3_grpo \
        --data  Data/cot_data/mtl_train.json \
        --model models/schema_linker_mtl \
        --out   models/schema_linker_grpo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Set

import torch
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    reward_correct:   float = 2.0
    reward_wrong:     float = -0.5
    reward_missed:    float = -3.0
    f1_bonus_weight:  float = 0.5
    format_penalty:   float = -1000.0


REWARD_CFG = RewardConfig()


def check_format(text: str) -> bool:
    if "<think>" not in text or "</think>" not in text:
        return False
    ts, te = text.find("<think>"), text.find("</think>")
    if te <= ts:
        return False
    think = text[ts + 7 : te]
    if not all(f"{i}." in think for i in range(1, 4)):
        return False
    return "The key field" in text[te + 8:]


def extract_key_fields_set(text: str) -> Set[str]:
    last = text.rfind("The key field")
    if last == -1:
        return set()
    rem   = text[last:]
    start = rem.find("[")
    end   = rem.find("]")
    if start == -1 or end == -1 or end < start:
        return set()
    return {item.strip() for item in rem[start + 1 : end].split(",") if item.strip()}


def grpo_reward_func(completions, **kwargs):
    rewards    = []
    true_links = kwargs["true_links"]

    for i, gen_text in enumerate(completions):
        true_link = set(true_links[i]) if true_links[i] else set()

        if not check_format(gen_text):
            rewards.append(REWARD_CFG.format_penalty)
            continue

        pred  = extract_key_fields_set(gen_text)
        tp    = len(pred & true_link)
        fp    = len(pred - true_link)
        fn    = len(true_link - pred)
        prec  = tp / max(1, len(pred))
        rec   = tp / max(1, len(true_link))
        f1    = 2 * prec * rec / max(1e-8, prec + rec)
        score = (
            REWARD_CFG.reward_correct  * tp
            + REWARD_CFG.reward_wrong  * fp
            + REWARD_CFG.reward_missed * fn
            + REWARD_CFG.f1_bonus_weight * f1
        )
        rewards.append(score)

    return rewards


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GRPODataset(Dataset):
    def __init__(self, data: list, tokenizer):
        self.data = []
        for item in data:
            prompt = (
                "<|im_start|>system\nYou are a Schema Linking Expert<|im_end|>\n"
                f"<|im_start|>user\n# Question:{item['question']}\n"
                f"# Database:{item['schema']}<|im_end|>\n<|im_start|>assistant\n"
            )
            self.data.append({"prompt": prompt, "true_links": item.get("key_fields", [])})

    def __len__(self):        return len(self.data)
    def __getitem__(self, i): return self.data[i]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--model",  default="models/schema_linker_mtl")
    parser.add_argument("--out",    default="models/schema_linker_grpo")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
        r=64, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    with open(args.data, encoding="utf-8") as f:
        all_data = json.load(f)

    train_ds = GRPODataset(all_data, tokenizer)

    grpo_config = GRPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        max_new_tokens=512,
        num_generations=4,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_ds,
        reward_funcs=grpo_reward_func,
        peft_config=peft_config,
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Stage 3 GRPO model saved to {args.out}")


if __name__ == "__main__":
    main()
