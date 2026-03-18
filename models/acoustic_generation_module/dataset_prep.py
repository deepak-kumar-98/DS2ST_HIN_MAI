"""
dataset_prep.py

Prepares 4 dataset files for AR model training:
  - train_ar_stage1.jsonl  : Monolingual (Task1) + TTS (Task2) samples
  - val_ar_stage1.jsonl    : Validation for Stage 1
  - train_ar_stage2.jsonl  : S2ST (Task3) samples for Stage 2
  - val_ar_stage2.jsonl    : Validation for Stage 2

Stage 1 data sources:
  1. Hindi→Maithili parallel data (DATA_JSON)
       - Maithili audio grouped by ref_audio_path (same ref = same speaker)
       - Task1: tgt_sem  + ref_cb0 → tgt_cb0
       - Task2: tgt_text + ref_cb0 → tgt_cb0
  2. IISc Maithili Male + Female speakers
       - All utterances from same speaker → any utterance can be ref
       - Task1: tgt_sem  + ref_cb0 → tgt_cb0
       - Task2: tgt_text + ref_cb0 → tgt_cb0

Stage 2 data source:
  - Hindi→Maithili parallel data (DATA_JSON)
  - Task3: src_sem + src_cb0 → tgt_cb0

Sequence formats:
  Task1: [TASK_MONO][SEM_BOS] tgt_sem  [SEP][ACOU_BOS] ref_cb0 [TGT_BOS] → tgt_cb0 [EOS]
  Task2: [TASK_TTS] [SEM_BOS] txt_toks [SEP][ACOU_BOS] ref_cb0 [TGT_BOS] → tgt_cb0 [EOS]
  Task3: [TASK_S2ST][SEM_BOS] src_sem  [SEP][ACOU_BOS] src_cb0 [TGT_BOS] → tgt_cb0 [EOS]

Notes:
  - Only codebook 0 is used. Shape is [T, 8] → unit[0] per frame
  - Labels are -100 for prefix, actual token IDs for target
  - Reference audio for Task1/Task2 is a DIFFERENT utterance from the same speaker
"""

import os
import json
import random
import torch
from tqdm import tqdm
from transformers import Qwen2ForCausalLM, AutoTokenizer

from helper_func import (
    extend_qwen_vocabulary,
    semantic_unit_to_id,
    acoustic_unit_to_id,
    get_special_token_id,
)
from extractor_new.token_ft import audio_to_semantic_tokens
from vocoder.infer_vocoder import extract_acoustic_units

# ── Paths ──────────────────────────────────────────────────────────────────────

# Parallel Hindi→Maithili data
DATA_JSON  = "/home/aditya/extraxtor_LLM/data/audio_data.json"
SOURCE_DIR = "/mnt/storage/aditya/raw_data/hindi_audio_data"
TARGET_DIR = "/mnt/storage/aditya/raw_data/maithili_audio_data"

# IISc Maithili monolingual data
IISC_SPEAKERS = [
    {
        "name"      : "IISc_Male_Spk001",
        "wav_dir"   : "/media/backup/IISC_mai_data/IISc_SYSPIN_Data/IISc_SYSPINProject_Maithili_Male_Spk001_HC/wav",
        "transcript": "/media/backup/IISC_mai_data/IISc_SYSPIN_Data/IISc_SYSPINProject_Maithili_Male_Spk001_HC/IISc_SYSPINProject_Maithili_Male_Spk001_HC_Transcripts.json",
    },
    {
        "name"      : "IISc_Female_Spk001",
        "wav_dir"   : "/media/backup/IISC_mai_data/IISc_SYSPIN_Data/IISc_SYSPINProject_Maithili_Female_Spk001_HC/wav",
        "transcript": "/media/backup/IISC_mai_data/IISc_SYSPIN_Data/IISc_SYSPINProject_Maithili_Female_Spk001_HC/IISc_SYSPINProject_Maithili_Female_Spk001_HC_Transcripts.json",
    },
]

# Output
OUTPUT_DIR = "/mnt/storage/aditya/SLM_acoustic_info"

# ── Config ─────────────────────────────────────────────────────────────────────
EVAL_SPLIT  = 0.05
MAX_SEQ_LEN = 2048
RANDOM_SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_folder_from_filename(filename: str) -> str:
    """
    audio_<folder>_<number>.wav → <folder>
    Example: audio_slot3_1234.wav → slot3
    """
    name  = os.path.basename(filename).replace(".wav", "")
    parts = name.split("_")
    if len(parts) < 3 or parts[0] != "audio":
        raise ValueError(f"Unexpected filename format: {filename}")
    return parts[1]


def compress_acoustic_units(units) -> list:
    """
    Extract codebook 0 from acoustic units.
    Shape is [T, 8] → take unit[0] for each frame (first codebook).
    """
    compressed = []
    for unit in units:
        compressed.append(int(unit[0]))
    return compressed


def get_tgt_path(file_name: str) -> str:
    """Get full path to Maithili (target) audio."""
    sub_folder = extract_folder_from_filename(os.path.basename(file_name))
    return os.path.join(TARGET_DIR, f"audios_{sub_folder}", file_name)


def get_src_path(file_name: str) -> str:
    """Get full path to Hindi (source) audio."""
    sub_folder = extract_folder_from_filename(os.path.basename(file_name))
    return os.path.join(SOURCE_DIR, f"audios_{sub_folder}", file_name)


def group_files_by_ref_audio(data_json: dict) -> list:
    """
    Groups file names by ref_audio_path.
    Files with the same ref_audio_path = same speaker.
    Returns list of groups (only groups with >1 file).
    """
    ref_audio_map = {}
    for file_name, info in data_json.items():
        ref_path = info["ref_audio_path"]
        ref_audio_map.setdefault(ref_path, []).append(file_name)

    grouped = [files for files in ref_audio_map.values() if len(files) > 1]
    return grouped


def build_sequence(
    task        : str,
    content_ids : list,
    ref_cb0_ids : list,
    tgt_cb0_ids : list,
    token_map   : dict,
) -> dict:
    """
    Builds input_ids and labels for a given task.

    Returns dict with input_ids, labels, prefix_len, total_len.
    Labels are -100 for prefix, actual token IDs for target.
    """
    tm = token_map

    task_token = {
        "mono" : tm["TASK_MONO"],
        "tts"  : tm["TASK_TTS"],
        "s2st" : tm["TASK_S2ST"],
    }[task]

    prefix = (
        [task_token]     +
        [tm["SEM_BOS"]]  +
        content_ids      +
        [tm["SEP"]]      +
        [tm["ACOU_BOS"]] +
        ref_cb0_ids      +
        [tm["TGT_BOS"]]
    )

    target    = tgt_cb0_ids + [tm["EOS"]]
    input_ids = prefix + target
    labels    = [-100] * len(prefix) + target

    return {
        "prefix_len": len(prefix),
        "total_len" : len(input_ids),
        "input_ids" : input_ids,
        "labels"    : labels,
    }


def is_valid(seq_dict: dict) -> bool:
    """Validity checks for a built sequence."""
    if seq_dict["total_len"] > MAX_SEQ_LEN:
        return False
    if seq_dict["total_len"] != len(seq_dict["labels"]):
        return False
    if all(l == -100 for l in seq_dict["labels"]):
        return False
    if seq_dict["labels"][seq_dict["prefix_len"]] == -100:
        return False
    return True


def safe_extract(audio_path: str) -> dict | None:
    """
    Safely extract semantic tokens and acoustic cb0 from audio.
    Returns dict with keys 'semantic' and 'acoustic', or None on failure.
    """
    try:
        semantic = audio_to_semantic_tokens(audio_path)
        acoustic = compress_acoustic_units(extract_acoustic_units(audio_path)[0])
        return {"semantic": semantic, "acoustic": acoustic}
    except Exception as e:
        print(f"  Warning: Feature extraction failed for {audio_path}: {e}")
        return None


def to_sem_ids(semantic_units: list, token_map: dict) -> list:
    return [semantic_unit_to_id(s, token_map) for s in semantic_units]


def to_cb0_ids(acoustic_units: list, token_map: dict) -> list:
    return [acoustic_unit_to_id(c, token_map) for c in acoustic_units]


# ── Stage 1 — Parallel data (Maithili side) ────────────────────────────────────

def stage1_from_parallel_data(data_json: dict, token_map: dict) -> list:
    """
    Uses the Maithili (target) side of the parallel data.
    Files grouped by ref_audio_path ensure same-speaker pairing for reference audio.
    Generates Task1 (monolingual resynthesis) and Task2 (TTS) samples.
    """
    samples = []
    tm      = token_map
    grouped = group_files_by_ref_audio(data_json)

    print(f"  Speaker groups : {len(grouped)}")
    print(f"  Total files in groups: {sum(len(g) for g in grouped)}")
    print(f"  Files with pair: {sum(len(g) for g in grouped)}")

    # Feature cache: file_name → features dict or None
    feature_cache = {}

    def get_features(file_name: str) -> dict | None:
        if file_name not in feature_cache:
            tgt_path = get_tgt_path(file_name)+'.wav'
            if not os.path.exists(tgt_path):
                print(f"  Warning: Target audio not found: {tgt_path}")
                feature_cache[file_name] = None
            else:
                feature_cache[file_name] = safe_extract(tgt_path)
        return feature_cache[file_name]

    for group in tqdm(grouped, desc="  Parallel stage1"):
        for file_name in group:
            tgt_feats = get_features(file_name)
            if tgt_feats is None:
                print(f"  Warning: Feature extraction failed for {file_name}")
                continue

            # Pick a different file from same speaker group as reference
            other_files = [f for f in group if f != file_name]
            ref_file    = random.choice(other_files)
            ref_feats   = get_features(ref_file)
            if ref_feats is None:
                print(f"  Warning: Feature extraction failed for ref {ref_file}")
                continue

            tgt_sem_ids = to_sem_ids(tgt_feats["semantic"], tm)
            tgt_cb0_ids = to_cb0_ids(tgt_feats["acoustic"], tm)
            ref_cb0_ids = to_cb0_ids(ref_feats["acoustic"], tm)

            # Task 1 — Monolingual resynthesis
            task1 = build_sequence("mono", tgt_sem_ids, ref_cb0_ids, tgt_cb0_ids, tm)
            if is_valid(task1):
                samples.append({
                    "task"     : "mono",
                    "source"   : "parallel",
                    "file_name": file_name,
                    **task1,
                })

            # Task 2 — TTS (Maithili text → Maithili speech)
            mai_text = data_json[file_name].get("mai_text", "")
            if mai_text:
                text_ids = tm["tokenizer"].encode(mai_text, add_special_tokens=False)
                task2    = build_sequence("tts", text_ids, ref_cb0_ids, tgt_cb0_ids, tm)
                if is_valid(task2):
                    samples.append({
                        "task"      : "tts",
                        "source"    : "parallel",
                        "file_name" : file_name,
                        **task2,
                    })

    print(f"  Stage1 from parallel data: {len(samples)} samples")
    return samples


# ── Stage 1 — IISc Maithili data ──────────────────────────────────────────────

def stage1_from_iisc_data(token_map: dict) -> list:
    """
    Uses IISc Maithili Male and Female speaker data.
    All utterances within a speaker folder = same speaker.
    Generates Task1 and Task2 samples.
    """
    samples = []
    tm      = token_map

    for speaker in IISC_SPEAKERS:
        wav_dir    = speaker["wav_dir"]
        trans_path = speaker["transcript"]
        spk_name   = speaker["name"]

        print(f"\n  Processing: {spk_name}")

        with open(trans_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        transcripts = transcript_data["Transcripts"]

        # Build utterance list — only files that have transcripts
        utterances = []
        for wav_file in os.listdir(wav_dir):
            if not wav_file.endswith(".wav"):
                continue
            utt_id = wav_file.replace(".wav", "")
            if utt_id not in transcripts:
                continue
            utterances.append({
                "utt_id"    : utt_id,
                "wav_path"  : os.path.join(wav_dir, wav_file),
                "transcript": transcripts[utt_id]["Transcript"],
            })

        print(f"  Utterances with transcripts: {len(utterances)}")

        # Feature cache for this speaker
        feature_cache = {}

        def get_features(utt: dict) -> dict | None:
            uid = utt["utt_id"]
            if uid not in feature_cache:
                feature_cache[uid] = safe_extract(utt["wav_path"])
            return feature_cache[uid]

        for i, utt in enumerate(tqdm(utterances, desc=f"  {spk_name}")):

            tgt_feats = get_features(utt)
            if tgt_feats is None:
                continue

            # Reference = different utterance from same speaker
            other_utts = [u for j, u in enumerate(utterances) if j != i]
            if not other_utts:
                continue
            ref_utt   = random.choice(other_utts)
            ref_feats = get_features(ref_utt)
            if ref_feats is None:
                continue

            tgt_sem_ids = to_sem_ids(tgt_feats["semantic"], tm)
            tgt_cb0_ids = to_cb0_ids(tgt_feats["acoustic"], tm)
            ref_cb0_ids = to_cb0_ids(ref_feats["acoustic"], tm)

            # Task 1 — Monolingual resynthesis
            task1 = build_sequence("mono", tgt_sem_ids, ref_cb0_ids, tgt_cb0_ids, tm)
            if is_valid(task1):
                samples.append({
                    "task"      : "mono",
                    "source"    : "iisc",
                    "speaker"   : spk_name,
                    "utt_id"    : utt["utt_id"],
                    "ref_utt_id": ref_utt["utt_id"],
                    **task1,
                })

            # Task 2 — TTS
            text_ids = tm["tokenizer"].encode(
                utt["transcript"], add_special_tokens=False
            )
            task2 = build_sequence("tts", text_ids, ref_cb0_ids, tgt_cb0_ids, tm)
            if is_valid(task2):
                samples.append({
                    "task"      : "tts",
                    "source"    : "iisc",
                    "speaker"   : spk_name,
                    "utt_id"    : utt["utt_id"],
                    "ref_utt_id": ref_utt["utt_id"],
                    "transcript": utt["transcript"],
                    **task2,
                })

    print(f"\n  Stage1 from IISc: {len(samples)} samples")
    return samples


# ── Stage 2 — S2ST ────────────────────────────────────────────────────────────

def stage2_from_parallel_data(data_json: dict, token_map: dict) -> list:
    """
    Uses Hindi→Maithili parallel data for Stage 2.
    Source semantic + source cb0 (speaker prompt) → target cb0.
    """
    samples = []
    tm      = token_map

    print(f"  Total parallel entries: {len(data_json)}")

    for file_name, info in tqdm(data_json.items(), desc="  Stage2 S2ST"):
        src_path = get_src_path(file_name)+'.wav'
        tgt_path = get_tgt_path(file_name)+'.wav'

        if not os.path.exists(src_path):
            print(f"  Warning: Source not found: {src_path}")
            continue
        if not os.path.exists(tgt_path):
            print(f"  Warning: Target not found: {tgt_path}")
            continue

        # Source: need both semantic and acoustic cb0
        src_feats = safe_extract(src_path)
        tgt_feats = safe_extract(tgt_path)
        if src_feats is None:
            continue
        
        if tgt_feats is None:
            print(f"  Warning: Target feature extraction failed for {tgt_path}")
            continue

        # Target: only need acoustic cb0
        try:
            tgt_cb0 = compress_acoustic_units(extract_acoustic_units(tgt_path)[0])
        except Exception as e:
            print(f"  Warning: Target extraction failed for {file_name}: {e}")
            continue

        tgt_sem_ids = to_sem_ids(tgt_feats["semantic"], tm)
        src_sem_ids = to_sem_ids(src_feats["semantic"], tm)
        src_cb0_ids = to_cb0_ids(src_feats["acoustic"], tm)
        tgt_cb0_ids = to_cb0_ids(tgt_cb0, tm)

        task3 = build_sequence("s2st", tgt_sem_ids, src_cb0_ids, tgt_cb0_ids, tm)
        if is_valid(task3):
            samples.append({
                "task"     : "s2st",
                "file_name": file_name,
                **task3,
            })

    print(f"  Stage2 total: {len(samples)} samples")
    return samples


# ── Split and save ─────────────────────────────────────────────────────────────

def split_and_save(samples: list, train_path: str, val_path: str):
    random.shuffle(samples)
    eval_size = max(1, int(len(samples) * EVAL_SPLIT))
    val_set   = samples[:eval_size]
    train_set = samples[eval_size:]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Train : {len(train_set)} → {os.path.basename(train_path)}")
    print(f"  Val   : {len(val_set)}   → {os.path.basename(val_path)}")


def print_stats(samples: list, label: str):
    lengths = [s["total_len"] for s in samples]
    if not lengths:
        print(f"{label}: 0 samples")
        return
    task_counts = {}
    for s in samples:
        task_counts[s.get("task", "?")] = task_counts.get(s.get("task", "?"), 0) + 1
    print(f"\n{label}:")
    print(f"  Count  : {len(lengths)}")
    print(f"  Min len: {min(lengths)}")
    print(f"  Max len: {max(lengths)}")
    print(f"  Avg len: {sum(lengths)/len(lengths):.1f}")
    print(f"  Tasks  : {task_counts}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # ── Load model and extend vocabulary ──────────────────────────────────
    print("Loading Qwen2.5-3B...")
    model     = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    print(f"Original vocab size: {len(tokenizer)}")

    model, tokenizer, token_map = extend_qwen_vocabulary(model, tokenizer)
    token_map["tokenizer"] = tokenizer   # needed for Task2 text tokenization
    print(f"Extended vocab size: {len(tokenizer)}")

    # Load data JSON once — used by both Stage 1 and Stage 2
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data_json = json.load(f)
    print(f"Loaded data JSON: {len(data_json)} entries")

    # ── Stage 1 ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1: Monolingual + TTS samples")
    print("="*60)

    print("\n[1/2] Parallel data (Maithili side, grouped by speaker)...")
    s1_parallel = stage1_from_parallel_data(data_json, token_map)

    print("\n[2/2] IISc Maithili data...")
    s1_iisc     = stage1_from_iisc_data(token_map)

    print(f"s1_parallel : {len(s1_parallel)} samples")

    stage1_all = s1_parallel + s1_iisc
    print_stats(stage1_all, "Stage 1 combined")

    print("\nSaving Stage 1...")
    split_and_save(
        samples    = stage1_all,
        train_path = os.path.join(OUTPUT_DIR, "train_ar_stage1.jsonl"),
        val_path   = os.path.join(OUTPUT_DIR, "val_ar_stage1.jsonl"),
    )

    # ── Stage 2 ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2: S2ST samples")
    print("="*60)

    stage2_all = stage2_from_parallel_data(data_json, token_map)
    print_stats(stage2_all, "Stage 2")

    print("\nSaving Stage 2...")
    split_and_save(
        samples    = stage2_all,
        train_path = os.path.join(OUTPUT_DIR, "train_ar_stage2.jsonl"),
        val_path   = os.path.join(OUTPUT_DIR, "val_ar_stage2.jsonl"),
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("  train_ar_stage1.jsonl")
    print("  val_ar_stage1.jsonl")
    print("  train_ar_stage2.jsonl")
    print("  val_ar_stage2.jsonl")


if __name__ == "__main__":
    main()