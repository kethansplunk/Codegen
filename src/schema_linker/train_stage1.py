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
- save_strategy="steps" (every 200 steps ≈ every 20 min) so Colab disconnects
  lose at most 20 min of work; auto-resumes from latest checkpoint on restart.

Run on Colab T4 (16GB) — point --out at Google Drive for persistence:
    python -m src.schema_linker.train_stage1 \
        --data  Data/cot_data/sql_cot_train.json \
        --model Qwen/Qwen3-8B \
        --out   /content/drive/MyDrive/codegen/checkpoints/sl_sql_stage1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
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
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    # QLoRA: 4-bit NF4 quantization keeps Qwen3-8B at ~4.5GB on T4 (vs ~16.7GB bf16).
    # LoRA adapters remain in bf16 — training quality is unchanged vs full bf16 LoRA.
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
    # use_gradient_checkpointing recomputes activations during backward pass
    # instead of storing them — trades ~30% compute for ~60% activation memory saving.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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

    # ~1520 optimizer steps per epoch (6073 examples / batch 1 / accum 4 = 1518).
    # batch=1 + accum=16 keeps effective batch=16 while using 4x less activation memory.
    # gradient_checkpointing saves another ~60% activation memory at ~30% compute cost.
    # Save and eval every 500 steps ≈ every 20-25 min on T4.
    # Early stopping patience=4 ≈ 2000 steps ≈ ~1.3 epochs of no improvement.
    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,    # effective batch = 16
        gradient_checkpointing=True,
        learning_rate=2e-4,
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
        save_total_limit=3,               # keep last 3 checkpoints on Drive
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
    )

    # Auto-resume from latest checkpoint if output dir already has checkpoints.
    # On first run: no checkpoints → trains from scratch.
    # After Colab disconnect: reconnect, re-run same command → resumes automatically.
    existing = sorted(glob.glob(os.path.join(args.out, "checkpoint-*")))
    resume_from = existing[-1] if existing else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    else:
        print("No checkpoint found — training from scratch.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )
    trainer.train(resume_from_checkpoint=resume_from)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Stage 1 model saved to {args.out}")


if __name__ == "__main__":
    main()
