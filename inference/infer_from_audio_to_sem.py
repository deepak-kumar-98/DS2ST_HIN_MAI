#!/usr/bin/env python3

"""
Usage:

python3 -m app2_adapter.infer_from_audio_to_sem \
    --audio /mnt/storage/aditya/raw_data/hindi_audio_data/audios_slot1/audio_slot1_3.wav
"""

import argparse
import torch
from fastdtw import fastdtw

from extractor_new.token_ft import audio_to_semantic_tokens

from app2_adapter.adapter_model import (
    load_config,
    build_model_stage2,
    compute_speech_mask,
    load_adapter,
    load_embed_weights
)

from safetensors.torch import load_file
from peft import set_peft_model_state_dict


# ------------------------------------------------------------
# Hardcoded paths
# ------------------------------------------------------------

CONFIG_PATH = "/home/aditya/extraxtor_LLM/src/app2_adapter/config_adapter.yaml"
CHECKPOINT_DIR = "/mnt/storage/aditya/app2_adapter/adapter_stage2"


# ------------------------------------------------------------
# Hardcoded special tokens
# ------------------------------------------------------------

BOS_ID  = 152165
EOS_ID  = 152166
SEP_ID  = 152167
S2ST_ID = 152168


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
def load_model(device):

    cfg = load_config(CONFIG_PATH)

    print("Building model...")

    model, tokenizer, sem_start, sem_end, _ = build_model_stage2(
        cfg,
        stage1_dir=CHECKPOINT_DIR,
        device=device
    )

    print("Loading LoRA...")
    lora_state = load_file(f"{CHECKPOINT_DIR}/adapter_model.safetensors")
    set_peft_model_state_dict(model.llm, lora_state)

    print("Loading Speech Adapter...")
    load_adapter(model.speech_adapter, CHECKPOINT_DIR)

    print("Loading embedding rows...")
    load_embed_weights(model, CHECKPOINT_DIR)

    model.eval()

    print("Model ready\n")

    return model, sem_start, sem_end


def edit_similarity(a, b):
    """Normalized edit similarity (0–1)"""
    if not a or not b:
        return 0.0

    dist, _ = fastdtw(
        a, b,
        dist=lambda x, y: 0 if x == y else 1
    )
    return 1.0 - dist / max(len(a), len(b))


# ------------------------------------------------------------
# Build input sequence
# ------------------------------------------------------------
def build_input_sequence(src_sem_units, sem_start):

    """
    Prefix format:

    <BOS> <S2ST> <HI_SEM_UNITS> <SEP>
    """

    shifted_units = [u + sem_start for u in src_sem_units]

    return [BOS_ID, S2ST_ID] + shifted_units + [SEP_ID]


# ------------------------------------------------------------
# Core inference
# ------------------------------------------------------------
def run_infer_sem(audio_path, device="cuda"):

    model, sem_start, sem_end = load_model(device)

    device = model.device

    # ------------------------------------------------
    # Audio → semantic tokens
    # ------------------------------------------------
    src_sem_units = audio_to_semantic_tokens(audio_path)

    print("Source semantic units:", len(src_sem_units))
    print(f"Example units: {src_sem_units[:10]}\n")

    # ------------------------------------------------
    # Build prefix
    # ------------------------------------------------
    prefix = build_input_sequence(src_sem_units, sem_start)

    input_ids = torch.tensor([prefix], dtype=torch.long).to(device)

    attention_mask = torch.ones_like(input_ids)

    speech_mask = compute_speech_mask(
        input_ids,
        sem_start,
        sem_end,
        SEP_ID
    )

    # ------------------------------------------------
    # Generate
    # ------------------------------------------------
    with torch.no_grad():

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speech_mask=speech_mask,
            max_new_tokens=400,
            eos_token_id=EOS_ID,
            do_sample=False
        )

    full = output[0].tolist()

    generated = full[len(prefix):]

    # ------------------------------------------------
    # Convert back to original semantic units
    # ------------------------------------------------
    tgt_sem_units = []

    for tok in generated:

        if tok == EOS_ID:
            break

        if sem_start <= tok <= sem_end:
            tgt_sem_units.append(tok - sem_start)

    print("\nTarget semantic units:", len(tgt_sem_units))
    print("Example units:", tgt_sem_units[:10])

    return tgt_sem_units


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    tgt_sem_pred = run_infer_sem(args.audio)
    TGT_PATH = args.audio.replace("hindi", "maithili")
    tgt_ground_truth = audio_to_semantic_tokens(TGT_PATH)

    edit_sim = edit_similarity(tgt_sem_pred, tgt_ground_truth)
    print(f"\nEdit similarity with ground truth: {edit_sim:.4f}")


if __name__ == "__main__":
    main()