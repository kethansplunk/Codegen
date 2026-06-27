"""
SchemaLinker Stage 1 — CoT SFT (Supervised Fine-Tuning).
Adapted from SchemaRAG train_SchemaLinker_CoT_peft.py:
- Input format updated to our sql_cot_train.json schema:
    {question, sql, db_name, schema, cot, key_fields}
  The 'think' portion is inside 'cot' between <think>...</think>.
  The 'answer' portion is cot content after </think>.
- Removed hardcoded /path/to/ paths.
- modelscope → transformers.
- LoRA r raised to 64 (matches SchemaRAG MTL config, better for complex reasoning).

Run on Colab T4 (16GB):
    python -m src.schema_linker.train_stage1 \
        --data  Data/cot_data/sql_cot_train.json \
        --model Qwen/Qwen3-8B \
        --out   models/schema_linker_cot
"""

from __future__ import annotations

import argparse
import json
import random

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CoTDataset(Dataset):
    def __init__(self, data: list, tokenizer, max_length: int = 2048):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = [self._encode(item) for item in data]

    def _encode(self, ex: dict) -> dict:
        cot = ex["cot"]
        # Split <think> and answer portions
        if "</think>" in cot:
            think_part, answer_part = cot.split("</think>", 1)
            think_part  = think_part.replace("<think>", "").strip()
            answer_part = answer_part.strip()
        else:
            think_part  = cot
            answer_part = ""

        instruction = self.tokenizer(
            f"<|im_start|>system\nYou are a Schema Linking Expert<|im_end|>\n"
            f"<|im_start|>user\n# Question:{ex['question']}\n"
            f"# Database:{ex['schema']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        response = self.tokenizer(
            f"<think>{think_part}</think>{answer_part}",
            add_special_tokens=False,
        )

        input_ids      = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels         = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]

        if len(input_ids) > self.max_length:
            input_ids      = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels         = labels[:self.max_length]
        else:
            pad = self.max_length - len(input_ids)
            input_ids      += [self.tokenizer.pad_token_id] * pad
            attention_mask += [0] * pad
            labels         += [-100] * pad

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]


def split_dataset(data: list, train_ratio: float = 0.9, seed: int = 42):
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True, help="Path to sql_cot_train.json")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model name or path")
    parser.add_argument("--out",   default="models/schema_linker_cot", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=2048)
    args = parser.parse_args()

    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    with open(args.data, encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"Total CoT samples: {len(all_data)}")

    train_data, val_data = split_dataset(all_data)
    train_ds = CoTDataset(train_data, tokenizer, args.max_len)
    val_ds   = CoTDataset(val_data,   tokenizer, args.max_len)

    training_args = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,   # effective batch = 16
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Stage 1 model saved to {args.out}")


if __name__ == "__main__":
    main()
