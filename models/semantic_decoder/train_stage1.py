#!/usr/bin/env python3
"""
STAGE 1: Adapter Alignment via Hindi ASR

PURPOSE:
  Teach the SpeechAdapter to produce embeddings that the FROZEN LLM can
  decode as Hindi text. After this stage, the adapter "speaks the LLM's
  language" — it maps Hindi semantic units to a region of embedding space
  the LLM already understands.

WHAT TRAINS:  SpeechAdapter only (~2M params)
WHAT'S FROZEN: Entire LLM (~3B params)
DATA:          Hindi ASR samples (speech units → Hindi text)

USAGE:
  CUDA_VISIBLE_DEVICES=0 python train_stage1.py --config config_adapter.yaml
"""

import os
import glob
import json
import argparse

import torch
from transformers import TrainingArguments

from adapter_model import (
    log, load_config, build_model_stage1, load_jsonl_by_task,
    build_dataset, make_collator, save_adapter,
    AdapterTrainer, SaveAllCallback,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    s1 = cfg["stage1"]

    log("=" * 70)
    log("STAGE 1: ADAPTER ALIGNMENT VIA ASR")
    log("=" * 70)
    log(f"  Goal: adapter learns to produce LLM-compatible embeddings")
    log(f"  Trainable: SpeechAdapter only")
    log(f"  LLM: completely frozen")
    log(f"  Task: {s1['task']}")
    log(f"  Samples: {s1['samples']}")
    log(f"  Epochs: {s1['epochs']}  |  LR: {s1['lr']}")
    log("=" * 70)

    # --- Model ---
    model, tokenizer, sem_start, sem_end, sep_id = build_model_stage1(cfg, args.device)

    # --- Data ---
    log("\n[DATA] Loading ASR samples")
    task_limits = {s1["task"]: s1["samples"]}
    samples = load_jsonl_by_task(cfg["train_jsonl"], task_limits, seed=cfg["seed"])
    ds = build_dataset(samples, cfg["max_seq_len"], sem_start, sem_end, sep_id)

    # Validate
    max_id = max(max(ex) for ex in ds["input_ids"])
    assert max_id < len(tokenizer), f"Token {max_id} >= vocab {len(tokenizer)}"
    log(f"  Max token ID: {max_id} OK")

    # Verify speech_mask has some True values
    total_speech = sum(sum(m) for m in ds["speech_mask"])
    log(f"  Total speech-masked tokens: {total_speech:,}")
    assert total_speech > 0, "No speech tokens found — check data format"

    # Split
    eval_ds = None
    if s1.get("eval_split", 0) > 0:
        split = ds.train_test_split(test_size=s1["eval_split"], seed=cfg["seed"])
        train_ds, eval_ds = split["train"], split["test"]
        log(f"  Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}")
    else:
        train_ds = ds
        log(f"  Train: {len(train_ds):,}")

    # --- Training ---
    out_dir = cfg["out_dir_stage1"]
    os.makedirs(out_dir, exist_ok=True)
    bf16_ok = torch.cuda.get_device_capability()[0] >= 8

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=s1["epochs"],
        per_device_train_batch_size=s1["per_device_bs"],
        per_device_eval_batch_size=s1["per_device_bs"],
        gradient_accumulation_steps=s1.get("grad_accum", 2),
        learning_rate=s1["lr"],
        weight_decay=s1.get("weight_decay", 0.0),
        warmup_ratio=s1.get("warmup_ratio", 0.1),
        max_grad_norm=s1.get("max_grad_norm", 1.0),
        optim="adamw_torch",
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=False,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25,
        save_strategy="steps",
        save_steps=s1.get("save_steps", 1000),
        save_total_limit=s1.get("save_total_limit", 3),
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=s1.get("save_steps", 1000) if eval_ds else None,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        seed=cfg["seed"],
    )

    # No task weighting for stage 1 (single task)
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(tokenizer.pad_token_id),
        task_weights={},  # No weighting — single ASR task
        lr_groups={"adapter": s1["lr"]},
    )
    trainer.add_callback(SaveAllCallback())

    eff_bs = s1["per_device_bs"] * s1.get("grad_accum", 2)
    steps_per_epoch = len(train_ds) // eff_bs
    total_steps = int(steps_per_epoch * s1["epochs"])
    log(f"\n  ~{steps_per_epoch:,} steps/epoch  |  ~{total_steps:,} total")

    log("\n" + "=" * 70)
    log("STAGE 1 TRAINING STARTS")
    log("=" * 70 + "\n")
    trainer.train()
    log("\n" + "=" * 70)
    log("STAGE 1 COMPLETE")
    log("=" * 70)

    # --- Find best checkpoint ---
    if eval_ds:
        best_loss, best_ckpt = float("inf"), None
        for ckpt in sorted(glob.glob(os.path.join(out_dir, "checkpoint-*"))):
            sp = os.path.join(ckpt, "trainer_state.json")
            if not os.path.exists(sp):
                continue
            with open(sp) as f:
                state = json.load(f)
            for e in state.get("log_history", []):
                if "eval_loss" in e and e["eval_loss"] < best_loss:
                    best_loss = e["eval_loss"]
                    best_ckpt = ckpt
        if best_ckpt:
            log(f"\n  Best checkpoint: {best_ckpt} (loss={best_loss:.4f})")
            with open(os.path.join(out_dir, "best_checkpoint.txt"), "w") as f:
                f.write(f"path: {best_ckpt}\neval_loss: {best_loss:.4f}\n")

    # --- Save final ---
    # trainer.save_model(out_dir)
    # save_adapter(model, out_dir)
    tokenizer.save_pretrained(out_dir)
    log(f"\n  Saved to: {out_dir}")
    log(f"\n  Next: python train_stage2.py --config {args.config}")


if __name__ == "__main__":
    main()