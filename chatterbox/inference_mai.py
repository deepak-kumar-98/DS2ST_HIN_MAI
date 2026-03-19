#!/usr/bin/env python3
"""
Batch Maithili TTS inference using FULL fine-tuned Chatterbox T3 model.
MEMORY-OPTIMIZED VERSION - Prevents OOM crashes

Reads from JSON file containing text and reference audio mappings.

Output structure:
  <OUTPUT_DIR>/
      audios_<filename>/
          audio_<filename>_<line_idx>.wav
"""

import sys
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import json
import re
import gc  # ADDED: For memory cleanup

# ------------------------------------------------------------------
# USER INPUT PATHS
# ------------------------------------------------------------------
JSON_FILE_PATH = "<path to audio data JSON>"
OUTPUT_DIR = "<path to output dir>"

# ------------------------------------------------------------------
# FIXED PATHS
# ------------------------------------------------------------------
T3_CKPT_PATH = "<path to T3 checkpoint>"
MAP_JSON_PATH = "<path to map JSON>"
LANGUAGE_ID = "mai"

# ADDED: Memory cleanup interval
CLEANUP_INTERVAL = 10  # clean every N samples

# ------------------------------------------------------------------
# Fix PYTHONPATH → OSTTS_LR/src
# ------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent / "OSTTS_LR" / "src"
sys.path.insert(0, str(SRC_DIR))
print(f"Added to PYTHONPATH: {SRC_DIR}")

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_char_mapping(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_audio_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_text(text, char_map):
    """
    1. Replace "-" with "से" ONLY when between digits
    2. Apply character mapping
    """
    text = re.sub(r"(\d)-(\d)", r"\1 से \2", text)

    out = []
    for ch in text:
        out.append(char_map.get(ch, ch))
    return "".join(out)


def extract_filename_from_key(key):
    if key.startswith("audio_"):
        key = key[6:]
    parts = key.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else key


# ------------------------------------------------------------------
# Memory cleanup utilities (ADDED)
# ------------------------------------------------------------------
def cleanup_memory(device):
    """Aggressively free memory to prevent OOM"""
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ------------------------------------------------------------------
# Audio utilities
# ------------------------------------------------------------------
def save_audio(path: Path, wav, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav).squeeze()
    sf.write(str(path), wav, sr)
    print(f"Saved: {path} ({len(wav)/sr:.2f}s)")


def trim_tail(wav, threshold=1e-4, keep_ms=80, sr=24000):
    wav = np.asarray(wav).astype(np.float32)
    idx = np.where(np.abs(wav) > threshold)[0]
    if len(idx) == 0:
        return wav
    last = idx[-1]
    keep = int(sr * (keep_ms / 1000))
    return wav[: min(len(wav), last + keep)]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("Maithili batch inference from JSON started (MEMORY OPTIMIZED)")

    json_file = Path(JSON_FILE_PATH)
    if not json_file.exists():
        raise FileNotFoundError(json_file)

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    ckpt = Path(T3_CKPT_PATH)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    char_map = load_char_mapping(MAP_JSON_PATH)
    audio_data = load_audio_data(json_file)

    print(f"Loaded {len(audio_data)} entries")

    device = pick_device()
    print(f"Using device: {device}")

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------
    print("Loading base Chatterbox pipeline...")
    tts = ChatterboxMultilingualTTS.from_pretrained(device=device)

    print(f"Loading Maithili T3 checkpoint: {ckpt}")
    state = torch.load(str(ckpt), map_location=device)
    tts.t3.load_state_dict(state, strict=True)
    tts.t3.eval()
    print("✅ Maithili fine-tuned T3 loaded")

    cleanup_memory(device)
    print(f"Memory cleanup interval: every {CLEANUP_INTERVAL} samples")

    # --------------------------------------------------------------
    # Decoding params
    # --------------------------------------------------------------
    base_gen_kwargs = dict(
        language_id=LANGUAGE_ID,
        exaggeration=0.0,
        temperature=0.6,
        top_p=0.9,
        min_p=0.08,
        repetition_penalty=1.3,
        cfg_weight=0.7,
    )

    total = len(audio_data)
    processed = 0
    skipped = 0

    # --------------------------------------------------------------
    # Inference loop
    # --------------------------------------------------------------
    for idx, (key, entry) in enumerate(audio_data.items(), start=1):
        mai_text = entry.get("mai_text", "")
        ref_audio = entry.get("ref_audio_path", "")

        if not mai_text:
            skipped += 1
            continue
        if not ref_audio or not Path(ref_audio).exists():
            skipped += 1
            continue

        filename = extract_filename_from_key(key)
        out_dir = output_root / f"audios_{filename}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_wav = out_dir / f"{key}.wav"
        if out_wav.exists():
            skipped += 1
            continue

        processed_text = preprocess_text(mai_text, char_map)

        print(f"\n[{idx}/{total}] {key}")
        print(f"  Text: {mai_text[:60]}")
        print(f"  Ref: {Path(ref_audio).name}")

        gen_kwargs = base_gen_kwargs.copy()
        gen_kwargs["audio_prompt_path"] = ref_audio

        try:
            with torch.no_grad():
                wav = tts.generate(text=processed_text, **gen_kwargs)
                wav = trim_tail(wav, sr=tts.sr)

            save_audio(out_wav, wav, tts.sr)
            processed += 1

            del wav  # IMPORTANT

            if processed % CLEANUP_INTERVAL == 0:
                print("  🧹 Memory cleanup...")
                cleanup_memory(device)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            skipped += 1

    cleanup_memory(device)

    print("\n" + "=" * 80)
    print("✅ Maithili batch inference complete")
    print(f"Total: {total}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
