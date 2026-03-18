"""
train_nar.py — NAR model training with multi-GPU DDP support.

Architecture: Encoder-Decoder Transformer
  Encoder : processes source context (semantics + src_cb0)
  Decoder : processes target cb0 with cross-attention to source
  Heads   : 7 × Linear(D, 1024) predicting cb1-cb7

Single GPU:
  CUDA_VISIBLE_DEVICES=1 python train_nar.py

Multiple GPUs (slots 1, 2, 3):
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29600 train_nar.py

Resume:
  CUDA_VISIBLE_DEVICES=1 python train_nar.py --resume_from /mnt/storage/aditya/checkpoints/nar/checkpoint_step_5000
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29600 train_nar.py --resume_from /mnt/storage/aditya/checkpoints/nar/checkpoint_step_5000
"""

import os
import sys
import json
import time
import shutil
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from contextlib import contextmanager

@contextmanager
def maybe_autocast(device_type, dtype):
    """autocast when on GPU, no-op otherwise."""
    if device_type == "cuda" and dtype is not None:
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield
from transformers import get_cosine_schedule_with_warmup

import config as cfg
from model import NARTransformer
from dataset import NARDataset, collate_fn, print_dataset_stats


# ── Distributed helpers ────────────────────────────────────────────────────────

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def is_master() -> bool:
    return get_rank() == 0


# ── GPU capability check ───────────────────────────────────────────────────────

def supports_bf16(device) -> bool:
    """Check if the GPU supports bfloat16 (Ampere and newer)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8   # Ampere = compute capability 8.x


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, scaler, step, val_loss, is_best=False):
    if not is_master():
        return

    raw_model = model.module if hasattr(model, "module") else model
    ckpt_name = "best_model" if is_best else f"checkpoint_step_{step}"
    ckpt_dir  = os.path.join(cfg.OUTPUT_DIR, ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    torch.save({
        "step"     : step,
        "val_loss" : val_loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler"   : scaler.state_dict() if scaler is not None else None,
    }, os.path.join(ckpt_dir, "training_state.pt"))

    print(f"  [SAVED] {ckpt_name}  (val_loss={val_loss:.4f})")

    # Keep only last SAVE_TOTAL_LIMIT step checkpoints
    if not is_best:
        step_ckpts = []
        for name in os.listdir(cfg.OUTPUT_DIR):
            if name.startswith("checkpoint_step_"):
                try:
                    s = int(name.replace("checkpoint_step_", ""))
                    step_ckpts.append((s, os.path.join(cfg.OUTPUT_DIR, name)))
                except ValueError:
                    pass
        step_ckpts.sort(key=lambda x: x[0])
        for _, old_path in step_ckpts[:-cfg.SAVE_TOTAL_LIMIT]:
            shutil.rmtree(old_path)
            print(f"  [DELETED] {os.path.basename(old_path)}")


def load_checkpoint(model, optimizer, scheduler, scaler, ckpt_dir: str) -> int:
    raw_model  = model.module if hasattr(model, "module") else model
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    model_path = os.path.join(ckpt_dir, "model.pt")

    if not os.path.exists(state_path) or not os.path.exists(model_path):
        if is_master():
            print(f"  Checkpoint not found in {ckpt_dir} — starting from step 0")
        return 0

    raw_model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    # scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    if is_master():
        print(f"  Resumed from step {state['step']}  (val_loss={state['val_loss']:.4f})")

    return state["step"]


# ── Loss ───────────────────────────────────────────────────────────────────────

def compute_loss(logits_list: list, labels_dict: dict) -> tuple:
    """
    Computes mean cross entropy loss across all 7 codebooks.
    Returns (total_loss, list of per-codebook losses).
    """
    criterion = nn.CrossEntropyLoss(
        ignore_index    = -100,
        label_smoothing = cfg.LABEL_SMOOTHING,
    )
    per_cb_losses = []

    for k in range(cfg.N_CODEBOOKS):
        logits = logits_list[k]              # [B, max_tgt_len, 1024]
        labels = labels_dict[f"cb{k+1}"]    # [B, max_tgt_len]
        loss_k = criterion(
            logits.permute(0, 2, 1).float(),  # [B, 1024, T]  cast to fp32 for loss stability
            labels,
        )
        per_cb_losses.append(loss_k)

    total_loss = torch.stack(per_cb_losses).mean()
    return total_loss, per_cb_losses


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, amp_dtype, max_batches=50) -> tuple:
    model.eval()
    total_loss    = 0.0
    per_cb_totals = [0.0] * cfg.N_CODEBOOKS
    n             = 0

    for batch in val_loader:
        if n >= max_batches:
            break

        input_ids        = batch["input_ids"].to(device)
        attention_mask   = batch["attention_mask"].to(device)
        target_start_idx = batch["target_start_idx"].to(device)
        target_len       = batch["target_len"].to(device)
        labels_dict      = {f"cb{k}": batch[f"cb{k}"].to(device) for k in range(1, 8)}

        with maybe_autocast(device.type, amp_dtype):
            logits_list = model(input_ids, attention_mask, target_start_idx, target_len)

        loss, per_cb = compute_loss(logits_list, labels_dict)

        total_loss += loss.item()
        for k in range(cfg.N_CODEBOOKS):
            per_cb_totals[k] += per_cb[k].item()
        n += 1

    model.train()
    avg_loss   = total_loss / max(n, 1)
    avg_per_cb = [x / max(n, 1) for x in per_cb_totals]
    return avg_loss, avg_per_cb


# ── Logger ─────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_path: str):
        self.log_path    = log_path
        self.train_start = time.time()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def elapsed(self) -> str:
        secs = int(time.time() - self.train_start)
        return str(datetime.timedelta(seconds=secs))

    def log_step(self, step, loss, lr, epoch, step_time, mem_gb, world_size, grad_norm):
        msg = (
            f"Step {step:7d} | "
            f"loss={loss:.4f} | "
            f"lr={lr:.2e} | "
            f"epoch={epoch:.2f} | "
            f"iter={step_time*1000:.0f}ms | "
            f"elapsed={self.elapsed()} | "
            f"mem={mem_gb:.1f}GB | "
            f"gnorm={grad_norm:.3f} | "
            f"gpus={world_size}"
        )
        print(msg, flush=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "step"      : step,
                "train_loss": loss,
                "lr"        : lr,
                "epoch"     : epoch,
                "iter_ms"   : step_time * 1000,
                "elapsed"   : self.elapsed(),
                "mem_gb"    : mem_gb,
                "grad_norm" : grad_norm,
            }) + "\n")

    def log_val(self, step, val_loss, per_cb, lr):
        cb_str = " | ".join(f"cb{k+1}={v:.4f}" for k, v in enumerate(per_cb))
        msg = (
            f"\n{'─'*80}\n"
            f"[VAL] Step {step:7d} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={lr:.2e} | "
            f"elapsed={self.elapsed()}\n"
            f"       {cb_str}\n"
            f"{'─'*80}\n"
        )
        print(msg, flush=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "step"    : step,
                "val_loss": val_loss,
                "per_cb"  : per_cb,
                "lr"      : lr,
                "type"    : "val",
            }) + "\n")


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Distributed init ───────────────────────────────────────────────────
    use_ddp    = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_size = get_world_size()

    # ── AMP dtype — bf16 for Ampere+, fp16 for older GPUs, None for CPU ─────
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if supports_bf16(device) else torch.float16
    else:
        amp_dtype = None
    scaler = GradScaler() if amp_dtype == torch.float16 else None

    # ── Seed ──────────────────────────────────────────────────────────────
    random.seed(cfg.SEED + get_rank())
    np.random.seed(cfg.SEED + get_rank())
    torch.manual_seed(cfg.SEED + get_rank())
    torch.cuda.manual_seed_all(cfg.SEED + get_rank())

    # ── Output dir ────────────────────────────────────────────────────────
    if is_master():
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    logger = Logger(os.path.join(cfg.OUTPUT_DIR, "train_log.jsonl")) if is_master() else None

    # ── Model ─────────────────────────────────────────────────────────────
    if is_master():
        print(f"\n{'='*80}")
        print(f"NAR Model Training  (Encoder-Decoder)")
        print(f"{'='*80}")
        print(f"  World size   : {world_size}")
        print(f"  Device       : {device}")
        print(f"  AMP dtype    : {amp_dtype}")
        print(f"  Output dir   : {cfg.OUTPUT_DIR}")

    model = NARTransformer().to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Datasets ──────────────────────────────────────────────────────────
    if is_master():
        print("\nLoading datasets...")

    train_ds = NARDataset(cfg.TRAIN_FILE)
    val_ds   = NARDataset(cfg.VAL_FILE)

    if is_master():
        print_dataset_stats(train_ds, "Train")

    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas = world_size,
            rank         = get_rank(),
            shuffle      = True,
        )
        shuffle_train = False
    else:
        train_sampler = None
        shuffle_train = True

    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = shuffle_train,
        sampler     = train_sampler,
        collate_fn  = collate_fn,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = use_pin_memory,
        drop_last   = True,   # avoids uneven batches in DDP
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = use_pin_memory,
    ) if is_master() else None

    # ── Optimizer ─────────────────────────────────────────────────────────
    # LR scaled linearly with world size (linear scaling rule)
    effective_lr = cfg.LR * world_size

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = effective_lr,
        weight_decay = cfg.WEIGHT_DECAY,
        betas        = (0.9, 0.95),
    )

    # ── Scheduler — defined in OPTIMIZER steps, not training steps ────────
    # This fixes the warmup bug where scheduler never reached peak LR
    total_optimizer_steps  = cfg.TOTAL_STEPS // cfg.GRAD_ACCUM
    warmup_optimizer_steps = cfg.WARMUP_STEPS   # already in optimizer steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_optimizer_steps,
        num_training_steps = total_optimizer_steps,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, scheduler, scaler, args.resume_from)
        # Reset LR to config value — ignores loaded optimizer LR
        for pg in optimizer.param_groups:
            pg['lr'] = cfg.LR * world_size
        if is_master():
            print(f"  LR reset to {cfg.LR * world_size:.2e}")

    # ── Print training config ──────────────────────────────────────────────
    if is_master():
        effective_batch = cfg.BATCH_SIZE * cfg.GRAD_ACCUM * world_size
        print(f"\n{'='*80}")
        print(f"Training config:")
        print(f"  Total train steps    : {cfg.TOTAL_STEPS}")
        print(f"  Total optimizer steps: {total_optimizer_steps}")
        print(f"  Warmup optimizer steps: {warmup_optimizer_steps}")
        print(f"  LR (scaled)          : {effective_lr:.2e}")
        print(f"  Batch per GPU        : {cfg.BATCH_SIZE}")
        print(f"  Grad accum           : {cfg.GRAD_ACCUM}")
        print(f"  Effective batch size : {effective_batch}")
        print(f"  Label smoothing      : {cfg.LABEL_SMOOTHING}")
        print(f"  Resume from step     : {start_step}")
        print(f"  Log every            : {cfg.LOG_EVERY} steps")
        print(f"  Val every            : {cfg.VAL_EVERY} steps")
        print(f"  Save every           : {cfg.SAVE_EVERY} steps")
        print(f"{'='*80}\n")

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    train_iter    = iter(train_loader)
    current_epoch = 0
    best_val_loss = float("inf")
    running_loss  = 0.0
    last_grad_norm = 0.0
    step_start    = time.time()
    n_samples     = len(train_ds)

    for step in range(start_step, cfg.TOTAL_STEPS):

        # ── Get batch ──────────────────────────────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            current_epoch += 1
            if use_ddp and train_sampler is not None:
                train_sampler.set_epoch(current_epoch)
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        input_ids        = batch["input_ids"].to(device)
        attention_mask   = batch["attention_mask"].to(device)
        target_start_idx = batch["target_start_idx"].to(device)
        target_len       = batch["target_len"].to(device)
        labels_dict      = {f"cb{k}": batch[f"cb{k}"].to(device) for k in range(1, 8)}

        # ── Forward with AMP ───────────────────────────────────────────
        with maybe_autocast(device.type, amp_dtype):
            logits_list = model(input_ids, attention_mask, target_start_idx, target_len)

        loss, _ = compute_loss(logits_list, labels_dict)
        loss    = loss / cfg.GRAD_ACCUM

        # ── NaN guard ──────────────────────────────────────────────────
        if torch.isnan(loss) or torch.isinf(loss):
            if is_master():
                print(f"Step {step+1}: NaN/Inf loss — skipping batch")
            optimizer.zero_grad()
            continue

        # ── Backward ───────────────────────────────────────────────────
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item() * cfg.GRAD_ACCUM

        # ── Optimizer step (every GRAD_ACCUM steps) ────────────────────
        # if (step + 1) % cfg.GRAD_ACCUM == 0:
        #     if scaler is not None:
        #         scaler.unscale_(optimizer)
        #         last_grad_norm = torch.nn.utils.clip_grad_norm_(
        #             model.parameters(), cfg.MAX_GRAD_NORM
        #         ).item()
        #         scaler.step(optimizer)
        #         scaler.update()
        #     else:
        #         last_grad_norm = torch.nn.utils.clip_grad_norm_(
        #             model.parameters(), cfg.MAX_GRAD_NORM
        #         ).item()
        #         optimizer.step()

        #     scheduler.step()   # step scheduler ONLY here — after optimizer step
        #     optimizer.zero_grad()

        if (step + 1) % cfg.GRAD_ACCUM == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.MAX_GRAD_NORM
            ).item()

            # Guard — skip update if gradients exploded despite clipping
            if grad_norm > 10.0:
                if is_master():
                    print(f'  WARNING: grad_norm={grad_norm:.1f} after clipping at step {step+1} — skipping')
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.update()
                scheduler.step()
                last_grad_norm = grad_norm
                continue

            last_grad_norm = grad_norm

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        # ── Log every LOG_EVERY steps ──────────────────────────────────
        if (step + 1) % cfg.LOG_EVERY == 0 and is_master():
            step_time    = (time.time() - step_start) / cfg.LOG_EVERY
            step_start   = time.time()
            avg_loss     = running_loss / cfg.LOG_EVERY
            running_loss = 0.0
            lr_current   = scheduler.get_last_lr()[0]
            epoch        = ((step + 1) * cfg.BATCH_SIZE * world_size) / n_samples
            mem_gb       = torch.cuda.memory_allocated(device) / 1e9

            logger.log_step(
                step       = step + 1,
                loss       = avg_loss,
                lr         = lr_current,
                epoch      = epoch,
                step_time  = step_time,
                mem_gb     = mem_gb,
                world_size = world_size,
                grad_norm  = last_grad_norm,
            )

        # ── Validation ─────────────────────────────────────────────────
        if (step + 1) % cfg.VAL_EVERY == 0 and is_master():
            val_loss, per_cb = validate(model, val_loader, device, amp_dtype)
            lr_current       = scheduler.get_last_lr()[0]

            logger.log_val(
                step     = step + 1,
                val_loss = val_loss,
                per_cb   = per_cb,
                lr       = lr_current,
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    step + 1, val_loss, is_best=True
                )

        # ── Periodic checkpoint ────────────────────────────────────────
        if (step + 1) % cfg.SAVE_EVERY == 0:
            if is_master():
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    step + 1, best_val_loss, is_best=False
                )
            if use_ddp:
                dist.barrier()   # sync all processes after save

        # ── Periodic memory cleanup ────────────────────────────────────
        if (step + 1) % 500 == 0:
            torch.cuda.empty_cache()

    # ── Training complete ──────────────────────────────────────────────────
    if is_master():
        # Final validation
        val_loss, per_cb = validate(model, val_loader, device, amp_dtype)
        logger.log_val(step=cfg.TOTAL_STEPS, val_loss=val_loss, per_cb=per_cb,
                       lr=scheduler.get_last_lr()[0])

        # Save final checkpoint
        save_checkpoint(model, optimizer, scheduler, scaler,
                        cfg.TOTAL_STEPS, val_loss, is_best=False)

        print(f"\n{'='*80}")
        print(f"Training complete.")
        print(f"  Best val loss : {best_val_loss:.4f}")
        print(f"  Best model    : {os.path.join(cfg.OUTPUT_DIR, 'best_model')}")
        print(f"  Total time    : {logger.elapsed()}")
        print(f"{'='*80}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()