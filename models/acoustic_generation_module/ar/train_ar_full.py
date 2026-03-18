"""
train_stage2_full.py

Full fine-tuning of Qwen2.5-3B for Stage 2 (S2ST) on Hindi→Maithili data.
Starts from either a Stage 1 checkpoint or fresh Qwen2.5-3B weights.

Memory optimizations used:
  - bfloat16 throughout
  - 8-bit AdamW optimizer (bitsandbytes)
  - Gradient checkpointing
  - Flash Attention 2
  - Batch size 1 + gradient accumulation

Usage:
  # From scratch (no Stage 1 checkpoint)
  python train_stage2_full.py

  # From Stage 1 checkpoint
  python train_stage2_full.py --resume_from /mnt/storage/aditya/checkpoints/stage1/best_model

  # Resume interrupted Stage 2 training
  python train_stage2_full.py --resume_from /mnt/storage/aditya/checkpoints/stage2/checkpoint_step_10000
"""

import os
import json
import math
import random
import shutil
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from transformers import (
    Qwen2ForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

TRAIN_FILE  = "/mnt/storage/aditya/SLM_acoustic_info/train_ar_stage2.jsonl"
VAL_FILE    = "/mnt/storage/aditya/SLM_acoustic_info/val_ar_stage2.jsonl"
OUTPUT_DIR  = "/mnt/storage/aditya/checkpoints/stage2_full_ft"
BASE_MODEL  = "Qwen/Qwen2.5-3B"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LR               = 2e-4
TOTAL_STEPS      = 300_000
WARMUP_STEPS     = 2_000
GRAD_ACCUM       = 16
BATCH_SIZE       = 1
MAX_SEQ_LEN      = 2048
WEIGHT_DECAY     = 0.01
MAX_GRAD_NORM    = 1.0
VAL_EVERY        = 1_000
SAVE_EVERY       = 2_000
SAVE_TOTAL_LIMIT = 4
LOG_EVERY        = 25
SEED             = 42

# ── Vocab extension ────────────────────────────────────────────────────────────

N_SEMANTIC = 500
N_ACOUSTIC = 1024


def extend_vocab(model, tokenizer):
    """
    Extends Qwen vocabulary with speech tokens.
    If tokens already exist (loading from checkpoint), just rebuilds token_map.
    """
    semantic_tokens = [f"<|sem_{i}|>"  for i in range(N_SEMANTIC)]
    acoustic_tokens = [f"<|acou_{i}|>" for i in range(N_ACOUSTIC)]
    special_tokens  = [
        "<|sem_bos|>",
        "<|acou_bos|>",
        "<|tgt_bos|>",
        "<|speech_sep|>",
        "<|task_mono|>",
        "<|task_tts|>",
        "<|task_s2st|>",
    ]

    all_new_tokens = semantic_tokens + acoustic_tokens + special_tokens

    # Add tokens — if already present, add_tokens returns 0 for existing ones
    old_vocab_size = len(tokenizer)
    num_added      = tokenizer.add_tokens(all_new_tokens, special_tokens=False)

    if num_added > 0:
        print(f"Added {num_added} new tokens. Resizing embeddings...")
        model.resize_token_embeddings(len(tokenizer))

        # Initialize new embeddings with mean of existing embeddings + small noise
        # This gives new tokens a reasonable starting point
        with torch.no_grad():
            embed_weight = model.model.embed_tokens.weight
            mean_embed   = embed_weight[:old_vocab_size].mean(dim=0, keepdim=True)
            embed_weight[old_vocab_size:] = (
                mean_embed +
                0.01 * torch.randn(
                    num_added,
                    embed_weight.shape[1],
                    dtype  = embed_weight.dtype,
                    device = embed_weight.device,
                )
            )

            # Also initialize lm_head if not tied
            lm_weight = model.lm_head.weight
            if lm_weight.data_ptr() != embed_weight.data_ptr():
                mean_lm = lm_weight[:old_vocab_size].mean(dim=0, keepdim=True)
                lm_weight[old_vocab_size:] = (
                    mean_lm +
                    0.01 * torch.randn(
                        num_added,
                        lm_weight.shape[1],
                        dtype  = lm_weight.dtype,
                        device = lm_weight.device,
                    )
                )

        model.config.vocab_size = len(tokenizer)
        print(f"New vocab size: {len(tokenizer)}")
    else:
        print(f"Vocab already extended. Size: {len(tokenizer)}")

    # Build token_map regardless
    token_map = {
        "sem_offset" : tokenizer.convert_tokens_to_ids("<|sem_0|>"),
        "acou_offset": tokenizer.convert_tokens_to_ids("<|acou_0|>"),
        "n_semantic" : N_SEMANTIC,
        "n_acoustic" : N_ACOUSTIC,
        "SEM_BOS"    : tokenizer.convert_tokens_to_ids("<|sem_bos|>"),
        "ACOU_BOS"   : tokenizer.convert_tokens_to_ids("<|acou_bos|>"),
        "TGT_BOS"    : tokenizer.convert_tokens_to_ids("<|tgt_bos|>"),
        "SEP"        : tokenizer.convert_tokens_to_ids("<|speech_sep|>"),
        "TASK_MONO"  : tokenizer.convert_tokens_to_ids("<|task_mono|>"),
        "TASK_TTS"   : tokenizer.convert_tokens_to_ids("<|task_tts|>"),
        "TASK_S2ST"  : tokenizer.convert_tokens_to_ids("<|task_s2st|>"),
        "EOS"        : tokenizer.eos_token_id,
        "PAD"        : tokenizer.pad_token_id,
    }

    return model, tokenizer, token_map


# ── Dataset ────────────────────────────────────────────────────────────────────

class SpeechDataset(Dataset):

    def __init__(self, jsonl_path: str, max_seq_len: int = MAX_SEQ_LEN):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item["total_len"] <= max_seq_len:
                    self.samples.append(item)
        print(f"Loaded {len(self.samples)} samples from {os.path.basename(jsonl_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels"   : torch.tensor(item["labels"],    dtype=torch.long),
            "total_len": item["total_len"],
        }


def collate_fn(batch: list, pad_token_id: int) -> dict:
    max_len    = max(item["total_len"] for item in batch)
    batch_size = len(batch)

    input_ids_padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels_padded    = torch.full((batch_size, max_len), -100,         dtype=torch.long)
    attention_mask   = torch.zeros((batch_size, max_len),              dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["total_len"]
        input_ids_padded[i, :L] = item["input_ids"]
        labels_padded[i, :L]    = item["labels"]
        attention_mask[i, :L]   = 1

    return {
        "input_ids"     : input_ids_padded,
        "labels"        : labels_padded,
        "attention_mask": attention_mask,
    }


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, optimizer, scheduler, step, val_loss, is_best=False):
    ckpt_name = "best_model" if is_best else f"checkpoint_step_{step}"
    ckpt_dir  = os.path.join(OUTPUT_DIR, ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    torch.save({
        "step"     : step,
        "val_loss" : val_loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, os.path.join(ckpt_dir, "training_state.pt"))

    print(f"  Saved: {ckpt_name}  (val_loss={val_loss:.4f})")

    # Keep only last SAVE_TOTAL_LIMIT step checkpoints
    if not is_best:
        step_ckpts = []
        for name in os.listdir(OUTPUT_DIR):
            if name.startswith("checkpoint_step_"):
                s = int(name.replace("checkpoint_step_", ""))
                step_ckpts.append((s, os.path.join(OUTPUT_DIR, name)))
        step_ckpts.sort(key=lambda x: x[0])
        for _, old_path in step_ckpts[:-SAVE_TOTAL_LIMIT]:
            shutil.rmtree(old_path)
            print(f"  Deleted old checkpoint: {os.path.basename(old_path)}")


def load_training_state(optimizer, scheduler, ckpt_dir) -> int:
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    if not os.path.exists(state_path):
        print("  No training_state.pt found — starting from step 0")
        return 0
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"  Resumed from step {state['step']} (val_loss={state['val_loss']:.4f})")
    return state["step"]


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, max_batches=50) -> float:
    model.eval()
    total_loss = 0.0
    n          = 0
    for batch in val_loader:
        if n >= max_batches:
            break
        out = model(
            input_ids      = batch["input_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device),
            labels         = batch["labels"].to(device),
        )
        total_loss += out.loss.item()
        n          += 1
    model.train()
    return total_loss / max(n, 1)


# ── Sequence length stats ──────────────────────────────────────────────────────

def print_length_stats(dataset: SpeechDataset, label: str):
    lengths = [s["total_len"] for s in dataset.samples]
    sorted_l = sorted(lengths)
    print(f"\nSequence lengths ({label}):")
    print(f"  Count  : {len(lengths)}")
    print(f"  Mean   : {sum(lengths)/len(lengths):.0f}")
    print(f"  Median : {sorted_l[len(sorted_l)//2]}")
    print(f"  95th % : {sorted_l[int(len(sorted_l)*0.95)]}")
    print(f"  Max    : {max(lengths)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Checkpoint directory to resume from (Stage 1 best_model or interrupted Stage 2)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    load_path = args.resume_from if args.resume_from else BASE_MODEL
    print(f"\nLoading model from: {load_path}")

    model = Qwen2ForCausalLM.from_pretrained(
        load_path,
        torch_dtype         = torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(load_path)

    # ── Extend vocabulary ──────────────────────────────────────────────────
    print("\nExtending vocabulary...")
    model, tokenizer, token_map = extend_vocab(model, tokenizer)

    # ── Enable gradient checkpointing ──────────────────────────────────────
    model.gradient_checkpointing_enable()

    # ── Move to device ─────────────────────────────────────────────────────
    model = model.to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total_params/1e9:.2f}B")
    print(f"Trainable parameters : {trainable_params/1e9:.2f}B")

    # ── Datasets ───────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = SpeechDataset(TRAIN_FILE, max_seq_len=MAX_SEQ_LEN)
    val_ds   = SpeechDataset(VAL_FILE,   max_seq_len=MAX_SEQ_LEN)

    print_length_stats(train_ds, "train")

    _collate = partial(collate_fn, pad_token_id=token_map["PAD"])

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = _collate,
        num_workers = 2,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        collate_fn  = _collate,
        num_workers = 2,
        pin_memory  = True,
    )

    # ── Optimizer — 8-bit AdamW ────────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr           = LR,
            weight_decay = WEIGHT_DECAY,
            betas        = (0.9, 0.95),
        )
        print("\nUsing 8-bit AdamW optimizer")
    except ImportError:
        print("\nWARNING: bitsandbytes not found, falling back to standard AdamW")
        print("Install with: pip install bitsandbytes")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = LR,
            weight_decay = WEIGHT_DECAY,
            betas        = (0.9, 0.95),
        )

    # ── Scheduler ──────────────────────────────────────────────────────────
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = WARMUP_STEPS,
        num_training_steps = TOTAL_STEPS,
    )

    # ── Resume training state ──────────────────────────────────────────────
    start_step = 0
    if args.resume_from:
        start_step = load_training_state(optimizer, scheduler, args.resume_from)

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Stage 2 Full Fine-tuning")
    print(f"  Total steps     : {TOTAL_STEPS}")
    print(f"  LR              : {LR}")
    print(f"  Warmup steps    : {WARMUP_STEPS}")
    print(f"  Grad accum      : {GRAD_ACCUM}")
    print(f"  Effective batch : {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Max seq len     : {MAX_SEQ_LEN}")
    print(f"  Device          : {device}")
    print(f"  Output          : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    log_path = os.path.join(OUTPUT_DIR, "training_log.jsonl")

    model.train()
    optimizer.zero_grad()

    train_iter    = iter(train_loader)
    best_val_loss = float("inf")
    running_loss  = 0.0

    for step in range(start_step, TOTAL_STEPS):

        # ── Get next batch ─────────────────────────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # ── Forward ────────────────────────────────────────────────────
        out  = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )
        loss = out.loss / GRAD_ACCUM

        # ── NaN guard ──────────────────────────────────────────────────
        if torch.isnan(loss):
            print(f"Step {step+1}: NaN loss — skipping batch")
            optimizer.zero_grad()
            continue

        # ── OOM guard ──────────────────────────────────────────────────
        try:
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            print(f"Step {step+1}: OOM — skipping batch")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue

        running_loss += loss.item() * GRAD_ACCUM

        # ── Optimizer step ─────────────────────────────────────────────
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Log every LOG_EVERY steps ──────────────────────────────────
        if (step + 1) % LOG_EVERY == 0:
            avg_loss     = running_loss / LOG_EVERY
            running_loss = 0.0
            lr_current   = scheduler.get_last_lr()[0]
            allocated    = torch.cuda.memory_allocated() / 1e9
            print(
                f"Step {step+1:6d} | "
                f"loss={avg_loss:.4f} | "
                f"lr={lr_current:.2e} | "
                f"mem={allocated:.1f}GB"
            )

        # ── Validation ─────────────────────────────────────────────────
        if (step + 1) % VAL_EVERY == 0:
            val_loss   = validate(model, val_loader, device)
            lr_current = scheduler.get_last_lr()[0]
            print(
                f"Step {step+1:6d} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={lr_current:.2e}"
            )

            # Log to file
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step"    : step + 1,
                    "val_loss": val_loss,
                    "lr"      : lr_current,
                }) + "\n")

            # Save best
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(model, tokenizer, optimizer, scheduler,
                                step + 1, val_loss, is_best=True)

        # ── Periodic checkpoint ────────────────────────────────────────
        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, tokenizer, optimizer, scheduler,
                            step + 1, best_val_loss, is_best=False)

        # ── Periodic cache clear ───────────────────────────────────────
        if (step + 1) % 100 == 0:
            torch.cuda.empty_cache()

    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best model    : {os.path.join(OUTPUT_DIR, 'best_model')}")


if __name__ == "__main__":
    main()