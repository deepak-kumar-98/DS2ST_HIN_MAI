#!/usr/bin/env python3
"""
Hindi TTS inference from plain text file using fine-tuned Chatterbox T3 model.
Each line in the input file is a Hindi sentence.
A reference audio is randomly assigned to each sentence, ensuring all are used equally.
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
import random
from collections import deque

# ------------------------------------------------------------------
# USER PATHS
# ------------------------------------------------------------------
INPUT_TXT_PATH = "<path to input text file>"
OUTPUT_DIR = "<path to output dir>"
T3_CKPT_PATH = "<path to T3 checkpoint>"
MAP_JSON_PATH = "<path to map JSON>"
REF_AUDIO_DIR = "<path to reference audio dir>"
LANGUAGE_ID = "hi"

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

    # Prepare reference audios
    ref_audio_dir = Path(REF_AUDIO_DIR)
    ref_audio_paths = sorted([str(p) for p in ref_audio_dir.glob("*.wav")])
    if len(ref_audio_paths) != 40:
        raise RuntimeError("Expected 10 reference audios in ref_audios folder.")

    # Read input sentences
    with open(INPUT_TXT_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Total sentences: {len(lines)}")

    # Prepare a list that cycles through all reference audios equally
    num_sentences = len(lines)
    repeats = num_sentences // len(ref_audio_paths)
    remainder = num_sentences % len(ref_audio_paths)
    ref_audio_list = ref_audio_paths * repeats + random.sample(ref_audio_paths, remainder)
    random.shuffle(ref_audio_list)
    ref_audio_queue = deque(ref_audio_list)

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

    for idx, line in enumerate(lines, 1):
        text = preprocess_text(line, char_map)
        out_wav = output_root / f"utt_{idx:05d}.wav"

        if out_wav.exists():
            continue

        ref_audio = ref_audio_queue.popleft()

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