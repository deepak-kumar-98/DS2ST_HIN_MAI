#!/usr/bin/env python3
"""
STAGE 2: S2ST Fine-tuning with Adapter + LoRA

PURPOSE:
  Train the full system for Hindi speech → Maithili semantic unit translation.
  The adapter (from stage 1) provides aligned speech embeddings.
  LoRA adapts the LLM for the translation task.
  Multi-task training prevents catastrophic forgetting.

WHAT TRAINS:
  - SpeechAdapter (loaded from stage 1, continues training)
  - LoRA adapters on all attention + MLP layers
  - embed_tokens new-token rows (for Maithili teacher forcing)
  - lm_head new-token rows (for generating Maithili semantic tokens)

WHAT'S FROZEN:
  - LLM base weights
  - embed_tokens original text rows

USAGE:
  CUDA_VISIBLE_DEVICES=0 python train_stage2.py --config config_adapter.yaml

  If you want to load from a specific stage 1 checkpoint:
  CUDA_VISIBLE_DEVICES=0 python train_stage2.py --config config_adapter.yaml \
    --stage1_dir /path/to/adapter_stage1/checkpoint-XXXX
"""

import os
import glob
import json
import argparse
from typing import Dict

import torch
from transformers import TrainingArguments

from adapter_model import (
    log, load_config, build_model_stage2, load_jsonl_by_task,
    build_dataset, make_collator, get_sem_token_range, get_sep_id,
    AdapterTrainer, SaveAllCallback, load_adapter,load_embed_weights
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage1_dir", type=str, default=None,
                        help="Override stage 1 output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    s2 = cfg["stage2"]
    stage1_dir = args.stage1_dir or cfg["out_dir_stage1"]

    log("=" * 70)
    log("STAGE 2: S2ST FINE-TUNING (ADAPTER + LoRA)")
    log("=" * 70)
    log(f"  Stage 1 adapter: {stage1_dir}")
    log(f"  Tasks: S2ST ({s2['s2st_samples']}), ASR ({s2['asr_hi_samples']}), MT ({s2['mt_samples']})")
    log(f"  Epochs: {s2['epochs']}  |  LR: adapter={s2['adapter_lr']}, lora={s2['lora_lr']}, embed={s2['embed_lr']}")
    log("=" * 70)

    # --- Model ---
    model, tokenizer, sem_start, sem_end, sep_id = build_model_stage2(
        cfg, stage1_dir, args.device)

    # --- Data ---
    log("\n[DATA] Loading multi-task data")
    task_limits = {
        "S2ST":   s2["s2st_samples"],
        "ASR_HI": s2["asr_hi_samples"],
        "MT":     s2.get("mt_samples", 30000),
    }
    samples = load_jsonl_by_task(cfg["train_jsonl"], task_limits, seed=cfg["seed"])
    ds = build_dataset(samples, cfg["max_seq_len"], sem_start, sem_end, sep_id)

    # Validate
    max_id = max(max(ex) for ex in ds["input_ids"])
    assert max_id < len(tokenizer), f"Token {max_id} >= vocab {len(tokenizer)}"

    # Task distribution
    counts: Dict[str, int] = {}
    for t in ds["task"]:
        counts[t] = counts.get(t, 0) + 1
    log("  Task distribution:")
    for t, c in sorted(counts.items()):
        log(f"    {t:10s}: {c:>7,} ({c / len(ds) * 100:.1f}%)")

    # Split
    eval_ds = None
    if s2.get("eval_split", 0) > 0:
        split = ds.train_test_split(test_size=s2["eval_split"], seed=cfg["seed"])
        train_ds, eval_ds = split["train"], split["test"]
        log(f"  Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}")
    else:
        train_ds = ds
        log(f"  Train: {len(train_ds):,}")

    # --- Training ---
    out_dir = cfg["out_dir_stage2"]
    os.makedirs(out_dir, exist_ok=True)
    bf16_ok = torch.cuda.get_device_capability()[0] >= 8

    task_weights = {
        "S2ST":   s2.get("s2st_weight", 3.0),
        "ASR_HI": s2.get("asr_weight", 1.0),
        "MT":     s2.get("mt_weight", 0.5),
    }

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=s2["epochs"],
        per_device_train_batch_size=s2["per_device_bs"],
        per_device_eval_batch_size=s2["per_device_bs"],
        gradient_accumulation_steps=s2.get("grad_accum", 4),
        learning_rate=s2["lora_lr"],  # Base LR (overridden per group)
        weight_decay=s2.get("weight_decay", 0.01),
        warmup_ratio=s2.get("warmup_ratio", 0.05),
        max_grad_norm=s2.get("max_grad_norm", 1.0),
        optim="adamw_torch",
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=False,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25,
        save_strategy="steps",
        save_steps=s2.get("save_steps", 500),
        save_total_limit=s2.get("save_total_limit", 3),
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=s2.get("save_steps", 500) if eval_ds else None,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        seed=cfg["seed"],
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(tokenizer.pad_token_id),
        task_weights=task_weights,
        lr_groups={
            "adapter": s2["adapter_lr"],
            "embed":   s2["embed_lr"],
            "lora":    s2["lora_lr"],
        },
    )
    trainer.add_callback(SaveAllCallback())

    eff_bs = s2["per_device_bs"] * s2.get("grad_accum", 4)
    steps_per_epoch = len(train_ds) // eff_bs
    total_steps = int(steps_per_epoch * s2["epochs"])
    log(f"\n  ~{steps_per_epoch:,} steps/epoch  |  ~{total_steps:,} total")
    log(f"  Checkpoints: every {s2.get('save_steps', 500)} steps")

    log("\n" + "=" * 70)
    log("STAGE 2 TRAINING STARTS")
    log("=" * 70 + "\n")
    ckpt_path = "/mnt/storage/aditya/app2_adapter/adapter_stage2/checkpoint-6000"

    # if os.path.isdir(ckpt_path):
    #     log(f"\nResuming from {ckpt_path}")

    #     from peft import PeftModel

    #     # Load LoRA weights
    #     model.llm = PeftModel.from_pretrained(model.llm, ckpt_path, is_trainable=True)

    #     # Load speech adapter weights
    #     load_adapter(model.speech_adapter, ckpt_path)

    #     # Load embedding weights
    #     load_embed_weights(model, ckpt_path)

    #     trainer.create_optimizer()
    #     trainer.create_scheduler(
    #         num_training_steps=trainer.args.max_steps
    #         if trainer.args.max_steps > 0
    #         else trainer.args.num_train_epochs
    #     )

    #     # 5️⃣ Load optimizer + scheduler state
    #     optimizer_state = torch.load(os.path.join(ckpt_path, "optimizer.pt"), map_location="cpu")
    #     scheduler_state = torch.load(os.path.join(ckpt_path, "scheduler.pt"), map_location="cpu")

    #     trainer.optimizer.load_state_dict(optimizer_state)
    #     trainer.lr_scheduler.load_state_dict(scheduler_state)

    #     # 6️⃣ Restore trainer state (global_step etc.)
    #     trainer.state = trainer.state.load_from_json(
    #         os.path.join(ckpt_path, "trainer_state.json")
    #     )


    #     trainer.train()
    # else:
    #     trainer.train()
    

    # --------------------------------------------------
    # Auto-resume from latest checkpoint if available
    # --------------------------------------------------

    # resume_checkpoint = None

    # checkpoints = glob.glob(os.path.join("/mnt/storage/aditya/app2_adapter/adapter_stage2", "checkpoint-*"))

    # if checkpoints:
    #     resume_checkpoint = max(
    #         checkpoints,
    #         key=lambda x: int(x.split("-")[-1])
    #     )
    #     log(f"\n🔄 Resuming from checkpoint: {resume_checkpoint}")
    # else:
    #     log("\n🆕 Starting fresh training")


    # if resume_checkpoint:
    #     trainer.train(resume_from_checkpoint=resume_checkpoint)
    # else:
    #     trainer.train()

    resume_checkpoint = None
    checkpoints = glob.glob(os.path.join(out_dir, "checkpoint-*"))

    if checkpoints:
        resume_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        log(f"\n🔄 Resuming from: {resume_checkpoint}")
    else:
        log("\n🆕 Starting fresh")


    # if resume_checkpoint:

    #     from peft import PeftModel

    #     log(f"\n🔄 Resuming from: {resume_checkpoint}")

    #     from peft import set_peft_model_state_dict
    #     from safetensors.torch import load_file

    #     adapter_path = os.path.join(resume_checkpoint, "adapter_model.safetensors")

    #     adapter_state = load_file(adapter_path)

    #     set_peft_model_state_dict(model.llm, adapter_state)

    #     # 2️⃣ Load speech adapter
    #     load_adapter(model.speech_adapter, resume_checkpoint)

    #     # 3️⃣ Load embedding rows
    #     load_embed_weights(model, resume_checkpoint)

    #     # 4️⃣ NOW recreate trainer optimizer & scheduler
    #     trainer.create_optimizer()
    #     trainer.create_scheduler(
    #         num_training_steps=trainer.state.max_steps
    #     )

    #     # 5️⃣ Load optimizer state
    #     trainer.optimizer.load_state_dict(
    #         torch.load(os.path.join(resume_checkpoint, "optimizer.pt"), map_location="cpu")
    #     )

    #     trainer.lr_scheduler.load_state_dict(
    #         torch.load(os.path.join(resume_checkpoint, "scheduler.pt"), map_location="cpu")
    #     )

    #     # 6️⃣ Restore trainer state
    #     from transformers.trainer_callback import TrainerState

    #     trainer.state = TrainerState.load_from_json(
    #         os.path.join(resume_checkpoint, "trainer_state.json")
    #     )

    #     print("Restored global_step:", trainer.state.global_step)
    #     print("Restored epoch:", trainer.state.epoch)
    #     print("Restored LR:", trainer.optimizer.param_groups[0]["lr"])

    #     trainer.train()
    # else:
    #     trainer.train()


    if resume_checkpoint:

        log(f"\n🔄 Resuming from: {resume_checkpoint}")

        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        from transformers.trainer_callback import TrainerState

        # 1️⃣ Load LoRA weights
        adapter_state = load_file(
            os.path.join(resume_checkpoint, "adapter_model.safetensors")
        )
        set_peft_model_state_dict(model.llm, adapter_state)

        # 2️⃣ Load speech adapter
        load_adapter(model.speech_adapter, resume_checkpoint)

        # 3️⃣ Load embedding rows
        load_embed_weights(model, resume_checkpoint)

        # 4️⃣ Restore trainer state FIRST
        trainer.state = TrainerState.load_from_json(
            os.path.join(resume_checkpoint, "trainer_state.json")
        )

        print("Restored global_step:", trainer.state.global_step)
        print("Restored epoch:", trainer.state.epoch)

        # 5️⃣ Create optimizer
        trainer.create_optimizer()

        # 6️⃣ Load optimizer state
        trainer.optimizer.load_state_dict(
            torch.load(os.path.join(resume_checkpoint, "optimizer.pt"), map_location="cpu")
        )

        print("Optimizer LR after load:",
            trainer.optimizer.param_groups[0]["lr"])

        # 7️⃣ Now create scheduler with CORRECT total steps
        # Let HF compute total steps using train dataloader
        train_dataloader = trainer.get_train_dataloader()
        total_steps = (
            len(train_dataloader)
            // trainer.args.gradient_accumulation_steps
            * trainer.args.num_train_epochs
        )

        trainer.create_scheduler(num_training_steps=total_steps)

        # 8️⃣ Load scheduler state
        trainer.lr_scheduler.load_state_dict(
            torch.load(os.path.join(resume_checkpoint, "scheduler.pt"), map_location="cpu")
        )

        print("Scheduler last_epoch:", trainer.lr_scheduler.last_epoch)
        print("Scheduler LR:", trainer.lr_scheduler.get_last_lr())

        trainer._created_optimizer = True
        trainer._created_scheduler = True


        trainer.train()

    else:
        trainer.train()

    # trainer.train(resume_from_checkpoint=True)

    # from peft import PeftModel

    # ckpt_path = "/mnt/storage/aditya/app2_adapter/adapter_stage2/checkpoint-6000"

    # print(f"\n🔄 Resuming from: {ckpt_path}")

    # # 1️⃣ Load LoRA (trainable!)
    # model.llm = PeftModel.from_pretrained(
    #     model.llm,
    #     ckpt_path,
    #     is_trainable=True
    # )

    # # 2️⃣ Load speech adapter
    # load_adapter(model.speech_adapter, ckpt_path)

    # # 3️⃣ Load embed rows
    # load_embed_weights(model, ckpt_path)

    # # 4️⃣ Start training normally (DO NOT pass resume_from_checkpoint)
    # trainer.train()





    # trainer.train()
    log("\n" + "=" * 70)
    log("STAGE 2 COMPLETE")
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
            log(f"\n  Best: {best_ckpt} (loss={best_loss:.4f})")
            with open(os.path.join(out_dir, "best_checkpoint.txt"), "w") as f:
                f.write(f"path: {best_ckpt}\neval_loss: {best_loss:.4f}\n")

    # --- Save ---
    # trainer.save_model(out_dir)
    # model.save_trainable(out_dir)
    tokenizer.save_pretrained(out_dir)
    log(f"\n  Saved to: {out_dir}")
    log(f"  Contents: speech_adapter.pt + LoRA + embed_weights.pt + tokenizer")
    log(f"\n  Inference: python infer_adapter.py --config {args.config} --checkpoint {out_dir}")


if __name__ == "__main__":
    main()