"""
Phase 14A — SQL Generator fine-tuning (Qwen2.5-Coder-7B-Instruct)

Fine-tunes Qwen2.5-Coder-7B-Instruct via LoRA (SFT) on the combined
schema + key_fields + SAR examples + question → SQL training data built
by scripts/build_generator_training_data.py.

T4 path  (--use_a100 omitted): 4-bit QLoRA NF4, batch=1, accum=16
A100 path (--use_a100):        bf16 LoRA, batch=4, accum=4, max_len=2048

Run on Colab:
    python -m src.generator.train \
        --data   Data/generator_data/sql_generator_train.jsonl \
        --out    models/generator_sql \
        --use_a100
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# NOTE: This intentionally uses plain transformers.Trainer (not trl.SFTTrainer).
# TRL's API churns fast — recent versions removed DataCollatorForCompletionOnlyLM
# and changed the SFTTrainer signature. transformers.Trainer is stable, and we do
# the prompt-masking ourselves (see _tokenize below), so training works with
# whatever transformers/peft Colab happens to install.

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


class ProgressCallback(TrainerCallback):
    """
    Prints training progress at every 10% and announces each checkpoint save.

    Emits:
      [TRAIN] plan   — once, up front: total steps, steps/epoch, when ckpts save
      [TRAIN] NN%    — at 10, 20, ... 100% with elapsed + ETA (real, measured)
      [CKPT]  saved  — each time the Trainer writes a checkpoint
    """

    def __init__(self):
        self._start = None
        self._next_pct = 10

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = time.time()
        total = state.max_steps
        epochs = args.num_train_epochs
        steps_per_epoch = max(1, round(total / epochs))
        print(f"[TRAIN] plan | {total} total steps | ~{steps_per_epoch} steps/epoch "
              f"| {epochs} epochs", flush=True)
        print(f"[TRAIN] plan | save_strategy=epoch → checkpoint 1 at ~step "
              f"{steps_per_epoch} (end of epoch 1, ~1/{epochs} of total time)", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        total = state.max_steps
        if not total:
            return
        pct = 100 * state.global_step / total
        if pct >= self._next_pct:
            elapsed = time.time() - self._start
            per_step = elapsed / max(1, state.global_step)
            eta = per_step * (total - state.global_step)
            print(f"[TRAIN] {int(self._next_pct)}% | step {state.global_step}/{total} "
                  f"| elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m "
                  f"| {per_step:.2f}s/step", flush=True)
            self._next_pct += 10

    def on_save(self, args, state, control, **kwargs):
        elapsed = time.time() - self._start if self._start else 0.0
        print(f"[CKPT] saved | step {state.global_step} | epoch {state.epoch:.2f} "
              f"| elapsed {elapsed/60:.1f}m", flush=True)

LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def train(
    data_path: str,
    output_dir: str,
    use_a100: bool = False,
    epochs: int = 3,
    lr: float = 2e-4,
):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if use_a100:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.enable_input_require_grads()
        max_seq_length = 2048
        batch_size     = 4
        grad_accum     = 4
    else:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
        max_seq_length = 1024
        batch_size     = 1
        grad_accum     = 16

    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Dataset — tokenize and mask the prompt so loss is only on the SQL
    # ------------------------------------------------------------------
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Training examples: {len(dataset)}")

    def _tokenize(example):
        """
        Split each example at the assistant marker, tokenize the prompt and the
        completion separately, and mask the prompt tokens with -100 so the model
        only learns to predict the SQL. Truncates to max_seq_length; if the prompt
        alone overflows, the whole example is masked out (contributes no loss).
        """
        text = example["text"]
        if RESPONSE_TEMPLATE in text:
            prompt_text, completion_text = text.split(RESPONSE_TEMPLATE, 1)
            prompt_text += RESPONSE_TEMPLATE
        else:
            prompt_text, completion_text = text, ""

        prompt_ids     = tokenizer(prompt_text,     add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + completion_ids)[:max_seq_length]
        labels    = ([-100] * len(prompt_ids) + completion_ids)[:max_seq_length]
        return {
            "input_ids":      input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels":         labels,
        }

    tokenized = dataset.map(_tokenize, remove_columns=dataset.column_names,
                            desc="Tokenizing + masking prompts")

    # Step math (so the plan is visible before the first step runs)
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, -(-len(tokenized) // effective_batch))  # ceil div
    total_steps     = steps_per_epoch * epochs
    print(f"Effective batch: {effective_batch} ({batch_size} x {grad_accum} accum)")
    print(f"Steps/epoch: {steps_per_epoch} | total steps: {total_steps} "
          f"| checkpoint after each epoch ({epochs} checkpoints)")

    # Pads input_ids/attention_mask/labels (labels padded with -100) per batch.
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding="longest", label_pad_token_id=-100,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=use_a100,
        fp16=not use_a100,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=not use_a100,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[ProgressCallback()],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Generator saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     required=True, help="sql_generator_train.jsonl")
    parser.add_argument("--out",      default="models/generator_sql")
    parser.add_argument("--epochs",   type=int,   default=3)
    parser.add_argument("--lr",       type=float, default=2e-4)
    parser.add_argument("--use_a100", action="store_true",
                        help="A100 path: bf16 full LoRA, larger batch")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.out,
        use_a100=args.use_a100,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
