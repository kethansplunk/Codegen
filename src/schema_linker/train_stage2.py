"""
SchemaLinker Stage 2 — Multi-Task Learning (MTL) PEFT.
Adapted from SchemaRAG train_SchemaLinker_MTL_peft.py:
- 3 tasks trained jointly: error_detection, correction, generation.
- WeightedRandomSampler balances task distribution.
- Input data format: each item must have:
    question, database (schema text), think (correct reasoning),
    answer (correct key fields), think_pre (wrong reasoning),
    schema_links_pred (wrong prediction), error_explanation.
  This data is produced by Phase 10A (error mining script).
- modelscope → transformers.
- deepspeed import removed (not needed for LoRA on T4).

Run on Colab T4:
    python -m src.schema_linker.train_stage2 \
        --data  Data/cot_data/mtl_train.json \
        --model models/schema_linker_cot \
        --out   models/schema_linker_mtl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TASK_MAP = {"error_detection": 0, "correction": 1, "generation": 2}

class MTLDataset(Dataset):
    def __init__(self, data: list, tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data: list = []
        for item in data:
            self.data.extend(self._encode_all_tasks(item))

    def _encode_one(self, input_text: str, target_text: str, task_id: int) -> dict:
        task_prefix = f"<task:{task_id}>"
        inp = self.tokenizer(task_prefix + input_text, add_special_tokens=False)
        tgt = self.tokenizer(target_text,              add_special_tokens=False)

        input_ids      = inp["input_ids"] + tgt["input_ids"]
        attention_mask = inp["attention_mask"] + tgt["attention_mask"]
        labels         = [-100] * len(inp["input_ids"]) + tgt["input_ids"]

        if len(input_ids) > self.max_length:
            input_ids      = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels         = labels[:self.max_length]
        else:
            pad = self.max_length - len(input_ids)
            input_ids      += [self.tokenizer.pad_token_id] * pad
            attention_mask += [0] * pad
            labels         += [-100] * pad

        return {
            "input_ids":      torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels":         torch.tensor(labels),
            "task_id":        task_id,
        }

    def _encode_all_tasks(self, ex: dict) -> list:
        sys = "<|im_start|>system\nYou are a Schema Linking Expert<|im_end|>\n"
        q, db, tp, sl = ex["question"], ex["schema"], ex.get("think_pre", ""), ex.get("schema_links_pred", "")
        err = ex.get("error_explanation", "")
        think, answer = ex.get("think", ""), ex.get("answer", "")

        tasks = {
            "error_detection": (
                f"{sys}<|im_start|>user\n# Question:{q}\n# Database:{db}\n"
                f"# Wrong Reasoning:{tp}\n# Schema Links:{sl}\nPlease analyse the error:<|im_end|>\n<|im_start|>assistant\n",
                f"{err}<|im_end|>",
            ),
            "correction": (
                f"{sys}<|im_start|>user\n# Question:{q}\n# Database:{db}\n"
                f"# Wrong Reasoning:{tp}\n# Wrong Answer:{sl}\n# Error Analysis:{err}\n"
                f"Please provide the correct reasoning and answer:<|im_end|>\n<|im_start|>assistant\n",
                f"<think>{think}</think>{answer}<|im_end|>",
            ),
            "generation": (
                f"{sys}<|im_start|>user\n# Question:{q}\n# Database:{db}\n"
                f"<|im_end|>\n<|im_start|>assistant\n",
                f"<think>{think}</think>{answer}<|im_end|>",
            ),
        }
        return [self._encode_one(inp, tgt, TASK_MAP[name]) for name, (inp, tgt) in tasks.items()]

    def __len__(self):      return len(self.data)
    def __getitem__(self, i): return self.data[i]


# ---------------------------------------------------------------------------
# Trainer with task-balanced sampling
# ---------------------------------------------------------------------------

class MTLTrainer(Trainer):
    def __init__(self, *args, w_error=0.3, w_correction=0.4, w_generation=1.0, **kwargs):
        self.w_error      = w_error
        self.w_correction = w_correction
        self.w_generation = w_generation
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v for k, v in inputs.items() if k != "task_id"}
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def _get_train_sampler(self):
        if self.train_dataset is None or not hasattr(self.train_dataset, "data"):
            return super()._get_train_sampler()

        task_counts = {}
        for item in self.train_dataset.data:
            tid = item.get("task_id", 0)
            task_counts[tid] = task_counts.get(tid, 0) + 1

        total = len(self.train_dataset)
        task_weights = {tid: total / (len(task_counts) * cnt) for tid, cnt in task_counts.items()}
        multipliers  = {0: self.w_error, 1: self.w_correction, 2: self.w_generation}

        sample_weights = [
            task_weights[item.get("task_id", 0)] * multipliers.get(item.get("task_id", 0), 1.0)
            for item in self.train_dataset.data
        ]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def split_dataset(data: list, train_ratio: float = 0.9, seed: int = 42):
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True, help="Path to mtl_train.json")
    parser.add_argument("--model", default="models/schema_linker_cot", help="Stage-1 checkpoint")
    parser.add_argument("--out",   default="models/schema_linker_mtl")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    with open(args.data, encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"Total MTL samples: {len(all_data)}")

    train_data, val_data = split_dataset(all_data)
    train_ds = MTLDataset(train_data, tokenizer)
    val_ds   = MTLDataset(val_data,   tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=False,
        fp16=False,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_steps=30,
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
    )

    existing = sorted(glob.glob(os.path.join(args.out, "checkpoint-*")))
    resume_from = existing[-1] if existing else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    else:
        print("No checkpoint found — training from scratch.")

    trainer = MTLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        w_error=0.3, w_correction=0.4, w_generation=1.0,
    )
    trainer.train(resume_from_checkpoint=resume_from)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Stage 2 MTL model saved to {args.out}")


if __name__ == "__main__":
    main()
