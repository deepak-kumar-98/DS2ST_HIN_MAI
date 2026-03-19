#!/usr/bin/env python3
"""
Hindi TTS inference from plain text file using fine-tuned Chatterbox T3 model.
Each line in the input file is a Hindi sentence.
Reference audio is assigned sequentially: line N → utt_000N.wav
CUDA_VISIBLE_DEVICES=1 python inf_mai_from_text.py --rank 0 --world_size 6 & CUDA_VISIBLE_DEVICES=2 python inf_mai_from_text.py --rank 1 --world_size 6 & CUDA_VISIBLE_DEVICES=3 python inf_mai_from_text.py --rank 2 --world_size 6 & CUDA_VISIBLE_DEVICES=4 python inf_mai_from_text.py --rank 3 --world_size 6 & CUDA_VISIBLE_DEVICES=5 python inf_mai_from_text.py --rank 4 --world_size 6 & CUDA_VISIBLE_DEVICES=6 python inf_mai_from_text.py --rank 5 --world_size 6 & wait

"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import json
import re
import gc

# ------------------------------------------------------------------
# USER PATHS
# ------------------------------------------------------------------
INPUT_TXT_PATH = "<path to input text file>"
OUTPUT_DIR = "<path to output dir>"
T3_CKPT_PATH = "<path to T3 checkpoint>"
MAP_JSON_PATH = "<path to map JSON>"
REF_AUDIO_DIR = "<path to reference audio dir>"  # Sequential reference audios
LANGUAGE_ID = "mai"

# ------------------------------------------------------------------
# PYTHONPATH FIX (CRITICAL)
# ------------------------------------------------------------------
CHATTERBOX_SRC = Path("<path to chatterbox src>")
sys.path.insert(0, str(CHATTERBOX_SRC))
print(f"Added to PYTHONPATH: {CHATTERBOX_SRC}")

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_text(text, char_map):
    text = re.sub(r"(\d)-(\d)", r"\1 से \2", text)
    return "".join(char_map.get(c, c) for c in text)

def save_audio(path, wav, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    wav = np.asarray(wav).squeeze()
    sf.write(str(path), wav, sr)

def trim_tail(wav, threshold=1e-4, keep_ms=80, sr=24000):
    wav = np.asarray(wav)
    idx = np.where(np.abs(wav) > threshold)[0]
    if len(idx) == 0:
        return wav
    return wav[: idx[-1] + int(sr * keep_ms / 1000)]

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    # Set CUDA device
    torch.cuda.set_device(0)
    device = "cuda"

    # Load char map
    char_map = load_json(MAP_JSON_PATH)

    # Prepare output directory
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    # Read input sentences
    with open(INPUT_TXT_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Total sentences: {len(lines)}")

    # Build sequential reference audio list: line N (1-indexed) → utt_000N.wav
    ref_audio_dir = Path(REF_AUDIO_DIR)
    ref_audio_list = []
    missing = []
    for idx in range(1, len(lines) + 1):
        ref_path = ref_audio_dir / f"utt_{idx:05d}.wav"
        if not ref_path.exists():
            missing.append(str(ref_path))
        ref_audio_list.append(str(ref_path))

    if missing:
        print(f"WARNING: {len(missing)} reference audio(s) not found:")
        for m in missing[:10]:  # Print first 10 missing
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Load TTS model
    print("Loading base Chatterbox model")
    tts = ChatterboxMultilingualTTS.from_pretrained(device=device)

    print("Loading Hindi fine-tuned T3 weights")
    state = torch.load(T3_CKPT_PATH, map_location=device)
    tts.t3.load_state_dict(state, strict=True)
    tts.t3.eval()

    cleanup()

    gen_kwargs = dict(
        language_id=LANGUAGE_ID,
        exaggeration=1.0,
        temperature=0.6,
        top_p=0.9,
        min_p=0.08,
        repetition_penalty=1.3,
        cfg_weight=0.7,
    )

    for idx, (line, ref_audio) in enumerate(zip(lines, ref_audio_list), 1):
        text = preprocess_text(line, char_map)
        out_wav = output_root / f"utt_{idx:05d}.wav"

        if out_wav.exists():
            continue

        if not Path(ref_audio).exists():
            print(f"SKIP [{idx}/{len(lines)}]: Reference audio not found: {ref_audio}")
            continue

        try:
            with torch.no_grad():
                wav = tts.generate(
                    text=text,
                    audio_prompt_path=ref_audio,
                    **gen_kwargs,
                )
                wav = trim_tail(wav, sr=tts.sr)

            save_audio(out_wav, wav, tts.sr)
            print(f"[{idx}/{len(lines)}] Saved: {out_wav} (ref: {os.path.basename(ref_audio)})")

            del wav

            if idx % 100 == 0:
                cleanup()
                print(f"Processed {idx} samples")

        except Exception as e:
            print(f"ERROR on line {idx}: {e}")
            continue

    cleanup()
    print("Inference complete.")

if __name__ == "__main__":
    main()