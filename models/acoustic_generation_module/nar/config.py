"""
config.py — All hyperparameters for NAR model training.
Edit this file to change any training setting.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────

TRAIN_FILE  = "/mnt/storage/aditya/SLM_acoustic_info_nar/train_nar.jsonl"
VAL_FILE    = "/mnt/storage/aditya/SLM_acoustic_info_nar/val_nar.jsonl"
OUTPUT_DIR  = "/mnt/storage/aditya/SLM_acoustic_info_nar/checkpoints/nar_02"

# ── NAR Vocabulary ─────────────────────────────────────────────────────────────
# Must match nar_dataset_prep.py exactly

VOCAB_SIZE   = 1529
SEM_OFFSET   = 0
ACOU_OFFSET  = 500
N_SEMANTIC   = 500
N_ACOUSTIC   = 1024
PAD_TOKEN    = 1524
SEM_BOS      = 1525
ACOU_BOS     = 1526
TGT_BOS      = 1527
SEP_TOKEN    = 1528
N_CODEBOOKS  = 7       # cb1 through cb7 (cb0 is input, not predicted)

# ── Model Architecture ─────────────────────────────────────────────────────────
# Sized for ~100-200K samples of speech data

D_MODEL      = 512     # hidden dimension
N_HEADS      = 8       # attention heads (D_MODEL must be divisible by N_HEADS)
D_FFN        = 2048    # feed-forward dimension (4 × D_MODEL)
N_LAYERS     = 6       # transformer encoder layers
                       # 8 layers gives ~32M params — good balance for your data size
DROPOUT      = 0.15
MAX_SEQ_LEN  = 2048    # must match dataset prep

# ── Training ───────────────────────────────────────────────────────────────────

LR               = 1e-4
TOTAL_STEPS      = 250_000
WARMUP_STEPS     = 0
BATCH_SIZE       = 32      # per GPU — NAR model is small, can afford larger batch
GRAD_ACCUM       = 2      # effective batch = BATCH_SIZE × GRAD_ACCUM × N_GPUS
WEIGHT_DECAY     = 0.01
MAX_GRAD_NORM    = 0.3
LABEL_SMOOTHING  = 0.1     # smoothing for cross entropy — helps generalization

# ── Logging and checkpointing ──────────────────────────────────────────────────

LOG_EVERY        = 25
VAL_EVERY        = 1_000
SAVE_EVERY       = 5_000
SAVE_TOTAL_LIMIT = 4

# ── Misc ───────────────────────────────────────────────────────────────────────

SEED         = 42
NUM_WORKERS  = 4       # dataloader workers per GPU