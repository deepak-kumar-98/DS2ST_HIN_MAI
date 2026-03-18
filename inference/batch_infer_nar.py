"""
batch_infer_nar.py — Batch inference over a folder of Hindi audios.

All models are loaded once and reused for every sample.

Usage:
    python nar_training/batch_infer_nar.py \
        --src_dir   /mnt/storage/aditya/test_hi_audios \
        --out_dir   /mnt/storage/aditya/outputs/test_mai_audios \
        --nar_ckpt  /mnt/storage/aditya/SLM_acoustic_info_nar/checkpoints/nar_02/best_model \
        --ar_ckpt   /mnt/storage/aditya/checkpoints/stage2_full_ft/best_model \
        --device    cuda
"""

import os
import sys
import argparse
import types
import torch
import torchaudio
from tqdm import tqdm
from transformers import Qwen2ForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app2_adapter')))

import config as cfg
from model import NARTransformer

from encodec import EncodecModel
from encodec.utils import convert_audio
from extractor_new.token_ft import audio_to_semantic_tokens
from vocoder.infer_vocoder import extract_acoustic_units

from app2_adapter.adapter_model import (
    load_config, build_model_stage2, compute_speech_mask,
    load_adapter, load_embed_weights,
)
from safetensors.torch import load_file
from peft import set_peft_model_state_dict


# ── Constants matching infer_from_audio_to_sem.py ─────────────────────────────
CONFIG_PATH    = "/home/aditya/extraxtor_LLM/src/app2_adapter/config_adapter.yaml"
SEM_CKPT_DIR   = "/mnt/storage/aditya/app2_adapter/adapter_stage2"
BOS_ID  = 152165
EOS_ID  = 152166
SEP_ID  = 152167
S2ST_ID = 152168

# ── Constants matching infer_full_ft.py ───────────────────────────────────────
N_SEMANTIC = 500
N_ACOUSTIC = 1024


# ==============================================================================
# Load all models once
# ==============================================================================

def load_all_models(ar_ckpt_dir, nar_ckpt_dir, device):

    # ── EnCodec ───────────────────────────────────────────────────────────────
    encodec = EncodecModel.encodec_model_24khz()
    encodec.set_target_bandwidth(6.0)
    encodec.eval().to(device)

    # ── Semantic model (app2_adapter) ─────────────────────────────────────────
    sem_cfg = load_config(CONFIG_PATH)
    sem_model, tokenizer_sem, sem_start, sem_end, _ = build_model_stage2(
        sem_cfg, stage1_dir=SEM_CKPT_DIR, device=device
    )
    lora_state = load_file(f"{SEM_CKPT_DIR}/adapter_model.safetensors")
    set_peft_model_state_dict(sem_model.llm, lora_state)
    load_adapter(sem_model.speech_adapter, SEM_CKPT_DIR)
    load_embed_weights(sem_model, SEM_CKPT_DIR)
    sem_model.eval()

    # ── AR model (Qwen2) ──────────────────────────────────────────────────────
    ar_model = Qwen2ForCausalLM.from_pretrained(
        ar_ckpt_dir, torch_dtype=torch.bfloat16, use_cache=True
    ).to(device).eval()
    ar_tokenizer = AutoTokenizer.from_pretrained(ar_ckpt_dir)
    ar_token_map = {
        "sem_offset" : ar_tokenizer.convert_tokens_to_ids("<|sem_0|>"),
        "acou_offset": ar_tokenizer.convert_tokens_to_ids("<|acou_0|>"),
        "n_semantic" : N_SEMANTIC,
        "n_acoustic" : N_ACOUSTIC,
        "SEM_BOS"    : ar_tokenizer.convert_tokens_to_ids("<|sem_bos|>"),
        "ACOU_BOS"   : ar_tokenizer.convert_tokens_to_ids("<|acou_bos|>"),
        "TGT_BOS"    : ar_tokenizer.convert_tokens_to_ids("<|tgt_bos|>"),
        "SEP"        : ar_tokenizer.convert_tokens_to_ids("<|speech_sep|>"),
        "TASK_S2ST"  : ar_tokenizer.convert_tokens_to_ids("<|task_s2st|>"),
        "EOS"        : ar_tokenizer.eos_token_id,
        "PAD"        : ar_tokenizer.pad_token_id,
    }

    # ── NAR model ─────────────────────────────────────────────────────────────
    nar_model = NARTransformer().to(device)
    nar_model.load_state_dict(
        torch.load(
            os.path.join(nar_ckpt_dir, "model.pt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    nar_model.eval()

    return encodec, sem_model, sem_start, sem_end, ar_model, ar_token_map, nar_model


# ==============================================================================
# Per-sample inference helpers
# ==============================================================================

def infer_tgt_sem(src_audio_path, sem_model, sem_start, sem_end, device):
    """Predict target semantic units from source audio."""
    src_sem_units = audio_to_semantic_tokens(src_audio_path)

    shifted = [u + sem_start for u in src_sem_units]
    prefix  = [BOS_ID, S2ST_ID] + shifted + [SEP_ID]

    input_ids      = torch.tensor([prefix], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)
    speech_mask    = compute_speech_mask(input_ids, sem_start, sem_end, SEP_ID)

    with torch.no_grad():
        output = sem_model.generate(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            speech_mask    = speech_mask,
            max_new_tokens = 400,
            eos_token_id   = EOS_ID,
            do_sample      = False,
        )

    generated = output[0].tolist()[len(prefix):]
    tgt_sem   = []
    for tok in generated:
        if tok == EOS_ID:
            break
        if sem_start <= tok <= sem_end:
            tgt_sem.append(tok - sem_start)

    return tgt_sem


def infer_tgt_cb0(src_audio_path, tgt_sem, ar_model, ar_token_map, device,
                  max_new_tokens=1000, temperature=1.0, top_p=0.9):
    """Predict target cb0 from source audio + target semantics using AR model."""
    tm = ar_token_map

    # Extract source cb0
    units    = extract_acoustic_units(src_audio_path)[0]
    src_cb0  = [int(u[0]) for u in units]

    sem_ids = [tm["sem_offset"]  + s for s in tgt_sem]
    cb0_ids = [tm["acou_offset"] + c for c in src_cb0]

    prefix = (
        [tm["TASK_S2ST"]] +
        [tm["SEM_BOS"]]   +
        sem_ids           +
        [tm["SEP"]]       +
        [tm["ACOU_BOS"]]  +
        cb0_ids           +
        [tm["TGT_BOS"]]
    )

    prefix_tensor = torch.tensor(prefix, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        generated = ar_model.generate(
            prefix_tensor,
            max_new_tokens = max_new_tokens,
            do_sample      = True,
            temperature    = temperature,
            top_p          = top_p,
            eos_token_id   = tm["EOS"],
            pad_token_id   = tm["PAD"],
        )

    new_ids = generated[0, len(prefix):].tolist()
    pred_cb0 = []
    for tid in new_ids:
        if tid == tm["EOS"]:
            break
        if tm["acou_offset"] <= tid < tm["acou_offset"] + tm["n_acoustic"]:
            pred_cb0.append(tid - tm["acou_offset"])

    return pred_cb0


def infer_nar(tgt_sem, src_cb0_list, pred_tgt_cb0_list, nar_model, device):
    """Predict cb1-cb7 using NAR model."""
    tgt_sem_t    = torch.tensor(tgt_sem,          dtype=torch.long)
    src_cb0_t    = torch.tensor(src_cb0_list,     dtype=torch.long)
    pred_tgt_cb0 = torch.tensor(pred_tgt_cb0_list,dtype=torch.long)

    src_cb0_ids      = src_cb0_t    + cfg.ACOU_OFFSET
    pred_tgt_cb0_ids = pred_tgt_cb0 + cfg.ACOU_OFFSET

    input_ids = torch.cat([
        torch.tensor([cfg.SEM_BOS],   dtype=torch.long),
        tgt_sem_t,
        torch.tensor([cfg.SEP_TOKEN], dtype=torch.long),
        torch.tensor([cfg.ACOU_BOS],  dtype=torch.long),
        src_cb0_ids,
        torch.tensor([cfg.TGT_BOS],   dtype=torch.long),
        pred_tgt_cb0_ids,
    ])

    target_start_idx = (
        1 + len(tgt_sem_t) + 1 + 1 + len(src_cb0_ids) + 1
    )
    target_len     = len(pred_tgt_cb0_ids)
    attention_mask = torch.ones(len(input_ids), dtype=torch.long)

    # Truncate if needed
    if len(input_ids) > cfg.MAX_SEQ_LEN:
        input_ids      = input_ids[:cfg.MAX_SEQ_LEN]
        attention_mask = attention_mask[:cfg.MAX_SEQ_LEN]
        target_len     = min(target_len, cfg.MAX_SEQ_LEN - target_start_idx)
        pred_tgt_cb0   = pred_tgt_cb0[:target_len]

    with torch.no_grad():
        logits_list = nar_model(
            input_ids.unsqueeze(0).to(device),
            attention_mask.unsqueeze(0).to(device),
            torch.tensor([target_start_idx], device=device),
            torch.tensor([target_len],       device=device),
        )

    pred_cbs = [logits_list[k][0].argmax(dim=-1).cpu() for k in range(7)]
    pred_cb1_7 = torch.stack(pred_cbs, dim=1)   # [T, 7]

    return pred_tgt_cb0[:target_len], pred_cb1_7


def reconstruct_audio(pred_tgt_cb0, pred_cb1_7, encodec, device):
    """Reconstruct waveform from cb0 + cb1-cb7. Returns [1, N] float32 tensor."""
    T = pred_tgt_cb0.shape[0]
    all_codes = torch.cat([
        pred_tgt_cb0.unsqueeze(1),
        pred_cb1_7,
    ], dim=1)   # [T, 8]

    codes = all_codes.transpose(0, 1).unsqueeze(0).to(device)  # [1, 8, T]
    scale = torch.ones(1, 1, device=device)

    with torch.no_grad():
        wav = encodec.decode([(codes, scale)])

    wav = wav.detach().cpu()
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    wav = wav.to(torch.float32).clamp(-1.0, 1.0)
    wav = wav / (wav.abs().max() + 1e-8)
    return wav.contiguous()


# ==============================================================================
# Extract src_cb0 as a list (reused in NAR)
# ==============================================================================

def get_src_cb0_list(src_audio_path):
    units = extract_acoustic_units(src_audio_path)[0]
    return [int(u[0]) for u in units]


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir",  required=True,  help="Folder of Hindi .wav files")
    parser.add_argument("--out_dir",  required=True,  help="Folder to save output .wav files")
    parser.add_argument("--nar_ckpt", required=True,  help="NAR model checkpoint directory")
    parser.add_argument("--ar_ckpt",  default="/mnt/storage/aditya/checkpoints/stage2_full_ft/best_model")
    parser.add_argument("--device",   default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Collect all wav files ─────────────────────────────────────────────────
    wav_files = sorted([
        f for f in os.listdir(args.src_dir)
        if f.lower().endswith(".wav")
    ])
    assert len(wav_files) > 0, f"No .wav files found in {args.src_dir}"

    # ── Load all models once ──────────────────────────────────────────────────
    (encodec, sem_model, sem_start, sem_end,
     ar_model, ar_token_map, nar_model) = load_all_models(
        args.ar_ckpt, args.nar_ckpt, device
    )

    # ── Batch inference ───────────────────────────────────────────────────────
    failed = []

    for fname in tqdm(wav_files, desc="Generating", unit="audio"):
        src_path = os.path.join(args.src_dir, fname)
        out_path = os.path.join(args.out_dir, fname)

        # Skip if already generated
        if os.path.exists(out_path):
            continue

        try:
            # 1. Predict target semantic units
            tgt_sem = infer_tgt_sem(src_path, sem_model, sem_start, sem_end, device)
            if not tgt_sem:
                raise ValueError("Empty target semantic units")

            # 2. Extract source cb0 (needed for both AR and NAR)
            src_cb0_list = get_src_cb0_list(src_path)

            # 3. Predict target cb0 via AR model
            pred_cb0_list = infer_tgt_cb0(
                src_path, tgt_sem, ar_model, ar_token_map, device
            )
            if not pred_cb0_list:
                raise ValueError("AR model returned empty cb0")

            # 4. Predict cb1-cb7 via NAR model
            pred_tgt_cb0, pred_cb1_7 = infer_nar(
                tgt_sem, src_cb0_list, pred_cb0_list, nar_model, device
            )

            # 5. Reconstruct and save audio
            wav = reconstruct_audio(pred_tgt_cb0, pred_cb1_7, encodec, device)
            torchaudio.save(out_path, wav, sample_rate=24000)

        except Exception as e:
            failed.append((fname, str(e)))
            continue

    # ── Summary ───────────────────────────────────────────────────────────────
    total    = len(wav_files)
    success  = total - len(failed)
    print(f"\nDone: {success}/{total} files generated → {args.out_dir}")

    if failed:
        print(f"Failed ({len(failed)}):")
        for fname, reason in failed:
            print(f"  {fname}: {reason}")


if __name__ == "__main__":
    main()