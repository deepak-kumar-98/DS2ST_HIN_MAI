"""
nar_dataset_prep.py

Prepares dataset for NAR model training.

For each sample:
  Input sequence : [SEM_BOS] tgt_sem [SEP][ACOU_BOS] src_cb0 [TGT_BOS] tgt_cb0
  Labels         : cb1, cb2, cb3, cb4, cb5, cb6, cb7  (7 separate lists, raw codes 0-1023)
  target_len     : U  (number of target frames = length of tgt_cb0)
                   needed to extract the correct positions from encoder output

Token ranges (NAR vocab, independent of Qwen):
  Semantic units : 0    to 499   (500 tokens)
  Acoustic units : 500  to 1523  (1024 tokens)
  Special tokens:
    PAD          : 1524
    SEM_BOS      : 1525
    ACOU_BOS     : 1526
    TGT_BOS      : 1527
    SEP          : 1528

Note:
  - Labels (cb1-cb7) are raw codebook values in range [0, 1023]
    No offset needed — NAR heads classify directly over 1024 classes
  - cb0 in the input sequence DOES use the acoustic offset (500-1523)
    because it is part of the input token sequence
  - Source audio  : Hindi  (semantic units + cb0 as speaker prompt)
  - Target audio  : Maithili (cb0 as input, cb1-cb7 as labels)
  - Uses parallel data only (same as Stage 2 AR training)
"""

import os
import json
import random
from tqdm import tqdm

from token_ft import audio_to_semantic_tokens
from infer_vocoder import extract_acoustic_units

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_JSON  = "/Data/deepakkumar/aditya_btp/chatterbox_data_prep/audio_data.json"
SOURCE_DIR = "/mnt/storage/aditya/raw_data/hindi_audio_data"
TARGET_DIR = "/mnt/storage/aditya/raw_data/maithili_audio_data"

TRAIN_OUTPUT = "/mnt/storage/aditya/SLM_acoustic_info/train_nar.jsonl"
VAL_OUTPUT   = "/mnt/storage/aditya/SLM_acoustic_info/val_nar.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────────

EVAL_SPLIT  = 0.05
MAX_SEQ_LEN = 2048
RANDOM_SEED = 42

# ── NAR Token map ──────────────────────────────────────────────────────────────
# Independent of Qwen vocabulary

NAR_TOKEN_MAP = {
    "sem_offset"  : 0,
    "acou_offset" : 500,
    "n_semantic"  : 500,
    "n_acoustic"  : 1024,
    "PAD"         : 1524,
    "SEM_BOS"     : 1525,
    "ACOU_BOS"    : 1526,
    "TGT_BOS"     : 1527,
    "SEP"         : 1528,
    "VOCAB_SIZE"  : 1529,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_folder_from_filename(filename: str) -> str:
    name  = os.path.basename(filename).replace(".wav", "")
    parts = name.split("_")
    if len(parts) < 3 or parts[0] != "audio":
        raise ValueError(f"Unexpected filename format: {filename}")
    return parts[1]


def get_src_path(file_name: str) -> str:
    sub_folder = extract_folder_from_filename(file_name)
    return os.path.join(SOURCE_DIR, f"audios_{sub_folder}", file_name)


def get_tgt_path(file_name: str) -> str:
    sub_folder = extract_folder_from_filename(file_name)
    return os.path.join(TARGET_DIR, f"audios_{sub_folder}", file_name)


def extract_all_codebooks(audio_path: str) -> list | None:
    """
    Extracts all 8 codebooks from audio.
    Shape [T, 8] → returns list of 8 lists, each of length T.
    Returns None on failure.
    """
    try:
        units = extract_acoustic_units(audio_path)[0]
        print(f"  Extracted units shape: {units.shape} from {audio_path}")
        # units shape: [T, 8]
        all_cbs = []
        for k in range(8):
            cb_k = [int(frame[k]) for frame in units]
            all_cbs.append(cb_k)
        return all_cbs   # list of 8 lists, all_cbs[0]=cb0, all_cbs[1]=cb1 etc.
    except Exception as e:
        print(f"  Warning: Acoustic extraction failed for {audio_path}: {e}")
        return None


def extract_semantic(audio_path: str) -> list | None:
    try:
        return audio_to_semantic_tokens(audio_path)
    except Exception as e:
        print(f"  Warning: Semantic extraction failed for {audio_path}: {e}")
        return None


def build_nar_sequence(
    src_semantic_units : list,   # from Hindi audio
    src_cb0            : list,   # from Hindi audio (speaker prompt)
    tgt_cb0            : list,   # from Maithili audio (ground truth, teacher forcing)
    token_map          : dict,
) -> dict:
    """
    Builds the NAR input sequence and labels.

    Input sequence:
      [SEM_BOS] src_sem_1..T [SEP][ACOU_BOS] src_cb0_1..S [TGT_BOS] tgt_cb0_1..U

    Labels: raw codebook values for cb1-cb7 (NOT offset, range 0-1023)
    target_len: U — needed during training to extract target positions from encoder output

    Note: input_ids use acoustic offset (500+) for cb0 tokens since they are
    part of the input token sequence. Labels do NOT use offset since NAR heads
    classify directly over 1024 classes.
    """
    tm = token_map

    # Input tokens — semantic units use sem_offset, acoustic use acou_offset
    sem_ids      = [tm["sem_offset"]  + s for s in src_semantic_units]
    src_cb0_ids  = [tm["acou_offset"] + c for c in src_cb0]
    tgt_cb0_ids  = [tm["acou_offset"] + c for c in tgt_cb0]

    input_ids = (
        [tm["SEM_BOS"]]  +
        sem_ids          +
        [tm["SEP"]]      +
        [tm["ACOU_BOS"]] +
        src_cb0_ids      +
        [tm["TGT_BOS"]]  +
        tgt_cb0_ids
    )

    # target_start_idx: position in input_ids where tgt_cb0 begins
    # needed to slice encoder hidden states during training
    target_start_idx = (
        1              +   # SEM_BOS
        len(sem_ids)   +   # semantic tokens
        1              +   # SEP
        1              +   # ACOU_BOS
        len(src_cb0_ids) + # source cb0
        1                  # TGT_BOS
    )

    return {
        "input_ids"        : input_ids,
        "total_len"        : len(input_ids),
        "target_start_idx" : target_start_idx,
        "target_len"       : len(tgt_cb0),    # U — number of target frames
    }


def is_valid(sample: dict) -> bool:
    if sample["total_len"] > MAX_SEQ_LEN:
        return False
    if sample["target_len"] == 0:
        return False
    # Verify target_start_idx + target_len == total_len
    if sample["target_start_idx"] + sample["target_len"] != sample["total_len"]:
        return False
    return True


# ── Main preparation ───────────────────────────────────────────────────────────

def prepare_nar_dataset() -> list:
    """
    Processes Hindi→Maithili parallel data to build NAR training samples.

    For each pair:
      Source (Hindi)   : extract semantic units + cb0 (speaker prompt)
      Target (Maithili): extract all 8 codebooks
                         cb0 → input sequence (teacher forcing)
                         cb1-cb7 → labels
    """
    samples = []
    tm      = NAR_TOKEN_MAP

    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    print(f"Total parallel entries: {len(data_json)}")

    for file_name, info in tqdm(data_json.items(), desc="Building NAR samples"):

        src_path = get_src_path(file_name)+".wav"
        tgt_path = get_tgt_path(file_name)+".wav"

        if not os.path.exists(src_path):
            print(f"  Warning: Source not found: {src_path}")
            continue
        if not os.path.exists(tgt_path):
            print(f"  Warning: Target not found: {tgt_path}")
            continue

        # ── Source: semantic units + cb0 ──────────────────────────────────
        src_semantic = extract_semantic(src_path)
        if src_semantic is None:
            continue

        src_codebooks = extract_all_codebooks(src_path)
        if src_codebooks is None:
            continue
        src_cb0 = src_codebooks[0]

        # ── Target: all 8 codebooks ────────────────────────────────────────
        tgt_codebooks = extract_all_codebooks(tgt_path)
        if tgt_codebooks is None:
            continue

        tgt_cb0 = tgt_codebooks[0]   # input (teacher forcing)
        tgt_cb1 = tgt_codebooks[1]   # label
        tgt_cb2 = tgt_codebooks[2]   # label
        tgt_cb3 = tgt_codebooks[3]   # label
        tgt_cb4 = tgt_codebooks[4]   # label
        tgt_cb5 = tgt_codebooks[5]   # label
        tgt_cb6 = tgt_codebooks[6]   # label
        tgt_cb7 = tgt_codebooks[7]   # label

        # Sanity check: all codebooks must have same length
        cb_lengths = [len(tgt_codebooks[k]) for k in range(8)]
        if len(set(cb_lengths)) != 1:
            print(f"  Warning: Codebook length mismatch for {file_name}: {cb_lengths}")
            continue

        # ── Build sequence ─────────────────────────────────────────────────
        seq = build_nar_sequence(
            src_semantic_units = src_semantic,
            src_cb0            = src_cb0,
            tgt_cb0            = tgt_cb0,
            token_map          = tm,
        )

        if not is_valid(seq):
            continue

        # ── Labels: raw codebook values (0-1023), no offset ───────────────
        sample = {
            "file_name"        : file_name,
            "input_ids"        : seq["input_ids"],
            "total_len"        : seq["total_len"],
            "target_start_idx" : seq["target_start_idx"],
            "target_len"       : seq["target_len"],
            # Labels — raw values, range 0-1023
            "cb1"              : tgt_cb1,
            "cb2"              : tgt_cb2,
            "cb3"              : tgt_cb3,
            "cb4"              : tgt_cb4,
            "cb5"              : tgt_cb5,
            "cb6"              : tgt_cb6,
            "cb7"              : tgt_cb7,
        }
        samples.append(sample)
        # if len(samples) > 5:
        #     print(f"  Processed {len(samples)} valid samples so far...")
        #     break

    return samples


# ── Stats ──────────────────────────────────────────────────────────────────────

def print_stats(samples: list, label: str):
    lengths      = [s["total_len"]   for s in samples]
    target_lens  = [s["target_len"]  for s in samples]
    sorted_l     = sorted(lengths)

    print(f"\n{label}:")
    print(f"  Count        : {len(samples)}")
    print(f"  Seq len mean : {sum(lengths)/len(lengths):.0f}")
    print(f"  Seq len 95%  : {sorted_l[int(len(sorted_l)*0.95)]}")
    print(f"  Seq len max  : {max(lengths)}")
    print(f"  Target mean  : {sum(target_lens)/len(target_lens):.0f} frames")
    print(f"  Target max   : {max(target_lens)} frames")


# ── Verify one sample ──────────────────────────────────────────────────────────

def verify_sample(sample: dict):
    """Quick sanity checks on a single sample."""
    tm = NAR_TOKEN_MAP

    input_ids        = sample["input_ids"]
    target_start_idx = sample["target_start_idx"]
    target_len       = sample["target_len"]

    # Check total_len consistency
    assert sample["total_len"] == len(input_ids), \
        "total_len mismatch"

    # Check target region is at the end
    assert target_start_idx + target_len == len(input_ids), \
        f"target region mismatch: {target_start_idx} + {target_len} != {len(input_ids)}"

    # Check input tokens are in valid vocab range
    for tid in input_ids:
        assert 0 <= tid < tm["VOCAB_SIZE"], \
            f"Token {tid} out of vocab range [0, {tm['VOCAB_SIZE']})"

    # Check target region tokens are acoustic (offset 500-1523)
    target_tokens = input_ids[target_start_idx:]
    for tid in target_tokens:
        assert tm["acou_offset"] <= tid < tm["acou_offset"] + tm["n_acoustic"], \
            f"Target token {tid} not in acoustic range"

    # Check labels are in raw code range (0-1023)
    for cb_name in ["cb1", "cb2", "cb3", "cb4", "cb5", "cb6", "cb7"]:
        codes = sample[cb_name]
        assert len(codes) == target_len, \
            f"{cb_name} length {len(codes)} != target_len {target_len}"
        for code in codes:
            assert 0 <= code < tm["n_acoustic"], \
                f"{cb_name} code {code} out of range [0, {tm['n_acoustic']})"

    return True


# ── Split and save ─────────────────────────────────────────────────────────────

def split_and_save(samples: list):
    random.shuffle(samples)
    eval_size = max(1, int(len(samples) * EVAL_SPLIT))
    val_set   = samples[:eval_size]
    train_set = samples[eval_size:]

    os.makedirs(os.path.dirname(TRAIN_OUTPUT), exist_ok=True)

    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Train : {len(train_set)} → {TRAIN_OUTPUT}")
    print(f"  Val   : {len(val_set)}   → {VAL_OUTPUT}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    print("="*60)
    print("NAR Dataset Preparation")
    print("="*60)
    print(f"\nNAR Token Map:")
    for k, v in NAR_TOKEN_MAP.items():
        print(f"  {k:20s}: {v}")

    # Build samples
    samples = prepare_nar_dataset()

    if not samples:
        print("ERROR: No valid samples generated.")
        return

    # Verify first few samples
    print(f"\nVerifying samples...")
    for i, s in enumerate(samples[:5]):
        try:
            verify_sample(s)
        except AssertionError as e:
            print(f"  Sample {i} FAILED: {e}")
            raise
    print(f"  ✓ Verification passed on {min(5, len(samples))} samples")

    # Stats
    print_stats(samples, "Full dataset")

    # Split and save
    print("\nSaving...")
    split_and_save(samples)

    print("\nDone.")


if __name__ == "__main__":
    main()