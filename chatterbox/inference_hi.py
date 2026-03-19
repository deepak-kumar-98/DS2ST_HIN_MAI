#!/usr/bin/env python3
"""
Batch Hindi TTS inference using FULL fine-tuned Chatterbox T3 model.
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
import gc  # ADDED: For garbage collection

# ------------------------------------------------------------------
# USER INPUT PATHS
# ------------------------------------------------------------------
JSON_FILE_PATH = "<path to audio data JSON>"   # <-- Path to the JSON file we generated
OUTPUT_DIR = "<path to output dir>"        # <-- Output directory for audio files

# ------------------------------------------------------------------
# FIXED PATHS (as provided)
# ------------------------------------------------------------------
T3_CKPT_PATH = "<path to T3 checkpoint>"
MAP_JSON_PATH = "<path to map JSON>"
LANGUAGE_ID = "hi"

# ADDED: Memory cleanup interval - adjust if needed (lower = more frequent cleanup)
CLEANUP_INTERVAL = 10  # Clean memory every 10 samples

# ------------------------------------------------------------------
# Fix PYTHONPATH → OSTTS_LR/src
# ------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent / "OSTTS_LR" / "src"
sys.path.insert(0, str(SRC_DIR))
print(f"Added to PYTHONPATH: {SRC_DIR}")

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def load_char_mapping(json_path):
    """Load character mapping from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_audio_data(json_path):
    """Load audio data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_text(text, char_map):
    """
    Preprocess text by applying character mappings.
    
    Rules:
    1. Replace "-" with "से" ONLY when it appears between two digits
    2. Replace all other characters according to char_map
    """
    # Step 1: Handle special case - replace "-" between numbers with "से"
    text = re.sub(r'(\d)-(\d)', r'\1 से \2', text)
    
    # Step 2: Replace all other characters from the mapping
    result = []
    for char in text:
        if char in char_map:
            result.append(char_map[char])
        else:
            result.append(char)
    
    return ''.join(result)


def extract_filename_from_key(key):
    """
    Extract filename from key like 'audio_slot1_1' -> 'slot1'
    Pattern: audio_<filename>_<line_count>
    """
    # Remove 'audio_' prefix
    if key.startswith('audio_'):
        key = key[6:]
    
    # Split by '_' and take all parts except the last one (which is line_count)
    parts = key.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0]
    return key


# ADDED: Memory cleanup function
def cleanup_memory(device):
    """Aggressively clean up memory to prevent OOM"""
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def save_audio(path: Path, wav, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.squeeze()
    sf.write(str(path), wav, sr)
    print(f"Saved: {path} ({len(wav)/sr:.2f}s)")


def trim_tail(wav, threshold=1e-4, keep_ms=80, sr=24000):
    """
    Removes trailing garbage / musical tail after EOS.
    Keeps a small buffer to avoid abrupt cutoff.
    """
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
    print("Hindi batch inference from JSON started (MEMORY OPTIMIZED)")

    json_file = Path(JSON_FILE_PATH)
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    ckpt = Path(T3_CKPT_PATH)
    if not ckpt.exists():
        raise FileNotFoundError(f"T3 checkpoint not found: {ckpt}")

    # Load mappings and data
    char_map = load_char_mapping(MAP_JSON_PATH)
    audio_data = load_audio_data(json_file)
    
    print(f"Loaded {len(audio_data)} entries from JSON")

    device = pick_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load Chatterbox pipeline + Hindi fine-tuned T3
    # ------------------------------------------------------------------
    print("Loading base Chatterbox pipeline...")
    tts = ChatterboxMultilingualTTS.from_pretrained(device=device)

    print(f"Loading Hindi T3 checkpoint: {ckpt}")
    state = torch.load(str(ckpt), map_location=device)
    tts.t3.load_state_dict(state, strict=True)
    tts.t3.eval()
    print("✅ Hindi fine-tuned T3 loaded")

    # ADDED: Initial memory cleanup
    cleanup_memory(device)
    print(f"Memory cleanup interval: every {CLEANUP_INTERVAL} samples")

    # ------------------------------------------------------------------
    # Stable decoding parameters (important)
    # ------------------------------------------------------------------
    base_gen_kwargs = dict(
        language_id=LANGUAGE_ID,
        exaggeration=1.0,          # Hindi benefits from higher articulation
        temperature=0.6,
        top_p=0.9,
        min_p=0.08,
        repetition_penalty=1.3,
        cfg_weight=0.7,
    )

    # ------------------------------------------------------------------
    # Process each entry in JSON
    # ------------------------------------------------------------------
    total_entries = len(audio_data)
    processed = 0
    skipped = 0
    
    for idx, (key, entry) in enumerate(audio_data.items(), start=1):
        # Extract information from entry
        hi_text = entry.get('hi_text', '')
        ref_audio_path = entry.get('ref_audio_path', '')
        
        if not hi_text:
            print(f"[{idx}/{total_entries}] Skipping {key}: no hi_text")
            skipped += 1
            continue
        
        if not ref_audio_path or not Path(ref_audio_path).exists():
            print(f"[{idx}/{total_entries}] Skipping {key}: ref_audio not found at {ref_audio_path}")
            skipped += 1
            continue
        
        # Extract filename to determine subfolder
        filename = extract_filename_from_key(key)
        out_folder = output_root / f"audios_{filename}"
        out_folder.mkdir(parents=True, exist_ok=True)
        
        # Output audio file path
        out_wav = out_folder / f"{key}.wav"
        
        if out_wav.exists():
            print(f"[{idx}/{total_entries}] Exists, skipping: {out_wav.name}")
            skipped += 1
            continue
        
        # Preprocess text
        processed_text = preprocess_text(hi_text, char_map)
        
        # Print progress
        print(f"\n[{idx}/{total_entries}] Processing: {key}")
        print(f"  Text: {hi_text[:50]}..." if len(hi_text) > 50 else f"  Text: {hi_text}")
        print(f"  Ref audio: {Path(ref_audio_path).name}")
        
        # Generate audio with specific reference audio
        gen_kwargs = base_gen_kwargs.copy()
        gen_kwargs['audio_prompt_path'] = str(ref_audio_path)
        
        try:
            # MODIFIED: Use torch.no_grad() to prevent gradient accumulation
            with torch.no_grad():
                wav = tts.generate(text=processed_text, **gen_kwargs)
                wav = trim_tail(wav, sr=tts.sr)
            
            save_audio(out_wav, wav, tts.sr)
            processed += 1
            
            # ADDED: Explicitly delete wav tensor
            del wav
            
            # ADDED: Periodic aggressive memory cleanup
            if processed % CLEANUP_INTERVAL == 0:
                print(f"  🧹 Memory cleanup at {processed} samples...")
                cleanup_memory(device)
                
        except Exception as e:
            print(f"  ❌ Error generating audio: {e}")
            skipped += 1
            continue

    # ADDED: Final cleanup
    cleanup_memory(device)
    
    print("\n" + "=" * 80)
    print("✅ Hindi batch inference complete.")
    print(f"Total entries: {total_entries}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()