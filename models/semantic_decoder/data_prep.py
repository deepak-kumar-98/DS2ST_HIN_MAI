#!/usr/bin/env python3
"""
Create multitask decoder dataset (JSONL) from a JSON dict like:

"audio_slot3_6730": {
  "hi_text": "...",
  "mai_text": "...",
  "ref_audio_path": "..."
}

Uses:
  from get_audio_path import get_hindi_audio_path, get_maithili_audio_path

Outputs JSONL with fields:
  utt_id, task, input_ids, labels, attention_mask, meta

CRITICAL CHANGE: Now uses Qwen's tokenizer for text instead of SentencePiece
to maintain compatibility with Qwen's pre-trained weights.



python <path to data_prep.py> \
  --qwen_name_or_path Qwen/Qwen2.5-3B \
  --json_path <path to audio_data.json> \
  --out_jsonl <path to data_tr_mh.jsonl> \
  --tokenizer_out_dir <path to tokenizer_mh> \
  --mt_sample_size 35000


"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from typing import Dict, List, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from transformers import AutoTokenizer

from src.extractor_new.token_ft import audio_to_semantic_tokens
from get_audio_path import get_hindi_audio_path, get_maithili_audio_path


IGNORE_INDEX = -100
NUM_SEM_UNITS = 500  # k-means clusters


def setup_extended_tokenizer(
    qwen_name_or_path: str,
    num_sem_units: int,
    tokenizer_out_dir: Optional[str] = None,
):
    """
    Extends Qwen tokenizer with:
      - <SEM_i> tokens for i in [0..num_sem_units-1]
      - task/control tokens: BOS/EOS/SEP + task ids
      - PAD token (if missing)
    
    CRITICAL: Does NOT replace Qwen's text tokenization. The original Qwen
    vocabulary and tokenization behavior remain intact. We only ADD new tokens.
    """
    # Load the original Qwen tokenizer with all its pre-trained vocabulary
    tok = AutoTokenizer.from_pretrained(qwen_name_or_path, trust_remote_code=True)
    
    print(f"Loaded Qwen tokenizer from: {qwen_name_or_path}")
    print(f"Original vocabulary size: {len(tok)}")
    
    # Add PAD token if the tokenizer doesn't have one
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<PAD>"})
        print("Added <PAD> token")
    
    # Add semantic tokens for speech representation
    # These map to the k-means cluster IDs from your semantic tokenizer
    sem_tokens = [f"<SEM_{i}>" for i in range(num_sem_units)]
    num_added_sem = tok.add_tokens(sem_tokens)
    print(f"Added {num_added_sem} semantic tokens: <SEM_0> through <SEM_{num_sem_units-1}>")
    
    # Add task-specific control tokens
    control_tokens = [
        "<S2ST_BOS>",    # Beginning of sequence for S2ST tasks
        "<S2ST_EOS>",    # End of sequence for S2ST tasks
        "<SEP>",         # Separator between input and output
        "<S2ST>",        # Task identifier for speech-to-speech translation
        "<ASR_HI>",      # Task identifier for Hindi ASR
        "<ASR_MA>",      # Task identifier for Maithili ASR
        "<MT>",          # Task identifier for machine translation
    ]
    num_added_control = tok.add_special_tokens({"additional_special_tokens": control_tokens})
    print(f"Added {num_added_control} control tokens")
    
    # Create lookup tables for semantic tokens
    # This maps semantic unit IDs (0..499) to model token IDs
    sem_id_table = tok.convert_tokens_to_ids(sem_tokens)
    
    # Create a dictionary of important token IDs for easy reference
    ids = {
        "PAD": tok.pad_token_id,
        "BOS": tok.convert_tokens_to_ids("<S2ST_BOS>"),
        "EOS": tok.convert_tokens_to_ids("<S2ST_EOS>"),
        "SEP": tok.convert_tokens_to_ids("<SEP>"),
        "S2ST": tok.convert_tokens_to_ids("<S2ST>"),
        "ASR_HI": tok.convert_tokens_to_ids("<ASR_HI>"),
        "ASR_MA": tok.convert_tokens_to_ids("<ASR_MA>"),
        "MT": tok.convert_tokens_to_ids("<MT>"),
    }
    
    print(f"Final vocabulary size: {len(tok)}")
    print(f"Vocabulary expansion: {num_added_sem + num_added_control} new tokens")
    print()
    print("Token ID mappings:")
    for name, tid in ids.items():
        print(f"  {name:10s} -> {tid}")
    print()
    
    # Save the extended tokenizer if requested
    # IMPORTANT: You MUST use this saved tokenizer during training
    # to ensure token IDs match between data preparation and training
    if tokenizer_out_dir:
        os.makedirs(tokenizer_out_dir, exist_ok=True)
        tok.save_pretrained(tokenizer_out_dir)
        
        # Also save metadata about the token ID mappings for reference
        metadata = {
            "ids": ids,
            "num_sem_units": num_sem_units,
            "original_qwen_model": qwen_name_or_path,
            "final_vocab_size": len(tok),
        }
        metadata_path = os.path.join(tokenizer_out_dir, "extended_tokenizer_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Extended tokenizer saved to: {tokenizer_out_dir}")
        print(f"✓ Metadata saved to: {metadata_path}")
        print()
    
    return tok, sem_id_table, ids


def sem_to_model_ids(units: List[int], sem_id_table: List[int]) -> List[int]:
    """
    Convert semantic unit IDs (from k-means clustering) to model token IDs.
    
    Args:
        units: List of semantic unit IDs in range [0, NUM_SEM_UNITS-1]
        sem_id_table: Lookup table mapping unit IDs to model token IDs
    
    Returns:
        List of model token IDs corresponding to the semantic units
    """
    if not isinstance(units, list):
        raise TypeError(f"Semantic units must be a list, got {type(units)}")
    
    if not all(isinstance(x, int) for x in units):
        non_ints = [type(x) for x in units if not isinstance(x, int)]
        raise TypeError(f"All semantic units must be integers, found types: {set(non_ints)}")
    
    # Validate that all unit IDs are in the valid range
    bad = [u for u in units if u < 0 or u >= NUM_SEM_UNITS]
    if bad:
        raise ValueError(
            f"Semantic unit IDs must be in range [0, {NUM_SEM_UNITS-1}]. "
            f"Found {len(bad)} invalid IDs. Examples: {bad[:10]}"
        )
    
    # Map each semantic unit ID to its corresponding model token ID
    return [sem_id_table[u] for u in units]


def text_to_model_ids(text: str, tokenizer: AutoTokenizer) -> List[int]:
    """
    Convert text to model token IDs using Qwen's tokenizer.
    
    This is CRITICAL: We use Qwen's original tokenizer to maintain compatibility
    with the pre-trained weights. This ensures the token IDs match what Qwen
    was trained on, allowing transfer learning to work properly.
    
    Args:
        text: Input text string (Hindi or Maithili)
        tokenizer: The Qwen tokenizer (with extended vocabulary)
    
    Returns:
        List of model token IDs corresponding to the text
    """
    if not isinstance(text, str):
        raise TypeError(f"Text must be a string, got {type(text)}")
    
    # Use Qwen's tokenizer to encode the text
    # add_special_tokens=False because we manually add BOS/EOS/SEP tokens
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    return token_ids


def make_example_sem2sem(
    task_id: int,
    src_sem: List[int],
    tgt_sem: List[int],
    sem_id_table: List[int],
    ids: Dict[str, int]
) -> Dict[str, Any]:
    """
    Create a speech-to-speech translation example.
    
    Format: [BOS] [TASK_ID] [source_semantic_tokens...] [SEP] [target_semantic_tokens...] [EOS]
    """
    prompt = (
        [ids["BOS"], task_id] + 
        sem_to_model_ids(src_sem, sem_id_table) + 
        [ids["SEP"]]
    )
    
    target = sem_to_model_ids(tgt_sem, sem_id_table) + [ids["EOS"]]
    
    input_ids = prompt + target
    labels = [IGNORE_INDEX] * len(prompt) + target
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


def make_example_sem2text(
    task_id: int,
    src_sem: List[int],
    tgt_text: str,
    sem_id_table: List[int],
    tokenizer: AutoTokenizer,
    ids: Dict[str, int]
) -> Dict[str, Any]:
    """
    Create a speech-to-text example (ASR or speech-to-text translation).
    
    Format: [BOS] [TASK_ID] [source_semantic_tokens...] [SEP] [target_text_tokens...] [EOS]
    """
    prompt = (
        [ids["BOS"], task_id] + 
        sem_to_model_ids(src_sem, sem_id_table) + 
        [ids["SEP"]]
    )
    
    target = text_to_model_ids(tgt_text, tokenizer) + [ids["EOS"]]
    
    input_ids = prompt + target
    labels = [IGNORE_INDEX] * len(prompt) + target
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


def make_example_text2text(
    task_id: int,
    src_text: str,
    tgt_text: str,
    tokenizer: AutoTokenizer,
    ids: Dict[str, int]
) -> Dict[str, Any]:
    """
    Create a text-to-text translation example (MT).
    
    Format: [BOS] [TASK_ID] [source_text_tokens...] [SEP] [target_text_tokens...] [EOS]
    """
    prompt = (
        [ids["BOS"], task_id] + 
        text_to_model_ids(src_text, tokenizer) + 
        [ids["SEP"]]
    )
    
    target = text_to_model_ids(tgt_text, tokenizer) + [ids["EOS"]]
    
    input_ids = prompt + target
    labels = [IGNORE_INDEX] * len(prompt) + target
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


def compute_semantic_tokens(audio_path: str) -> List[int]:
    """
    Compute semantic tokens from audio without caching.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        List of semantic unit IDs in range [0, NUM_SEM_UNITS-1]
    """
    # Extract semantic tokens using your k-means model (k=500)
    units = audio_to_semantic_tokens(audio_path)
    
    # Validate the output
    if not isinstance(units, list):
        raise TypeError(
            f"audio_to_semantic_tokens must return a list, got {type(units)} "
            f"for audio: {audio_path}"
        )
    
    if not all(isinstance(x, int) for x in units):
        non_ints = [type(x) for x in units if not isinstance(x, int)]
        raise TypeError(
            f"All semantic units must be integers. Found types: {set(non_ints)} "
            f"for audio: {audio_path}"
        )
    
    # Validate range
    bad = [u for u in units if u < 0 or u >= NUM_SEM_UNITS]
    if bad:
        raise ValueError(
            f"Semantic units must be in range [0, {NUM_SEM_UNITS-1}]. "
            f"Found {len(bad)} out-of-range values. Examples: {bad[:10]} "
            f"for audio: {audio_path}"
        )
    
    return units


def main():
    ap = argparse.ArgumentParser(
        description="Prepare multitask training dataset with Qwen tokenizer"
    )
    ap.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to dataset JSON with audio_file -> {hi_text, mai_text, ref_audio_path} mappings"
    )
    ap.add_argument(
        "--qwen_name_or_path",
        type=str,
        required=True,
        help="Qwen model name or local path (e.g., 'Qwen/Qwen2.5-3B-Instruct')"
    )
    ap.add_argument(
        "--out_jsonl",
        type=str,
        required=True,
        help="Output JSONL path for training data"
    )
    ap.add_argument(
        "--tokenizer_out_dir",
        type=str,
        required=True,
        help="Directory to save extended tokenizer (REQUIRED for training consistency)"
    )
    ap.add_argument(
        "--mt_sample_size",
        type=int,
        default=35000,
        help="Number of MT examples to sample from available parallel data (default: 35000)"
    )
    ap.add_argument(
        "--max_items",
        type=int,
        default=-1,
        help="Maximum number of items to process (-1 for all). Useful for debugging."
    )
    ap.add_argument(
        "--skip_mt",
        action="store_true",
        help="Skip machine translation (MT) task examples"
    )
    ap.add_argument(
        "--skip_asr",
        action="store_true",
        help="Skip ASR task examples"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MT sampling (default: 42)"
    )
    args = ap.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("=" * 80)
    print("MULTITASK DATASET PREPARATION WITH QWEN TOKENIZER")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Input JSON:        {args.json_path}")
    print(f"  Output JSONL:      {args.out_jsonl}")
    print(f"  Qwen model:        {args.qwen_name_or_path}")
    print(f"  Semantic units:    {NUM_SEM_UNITS} (k-means clusters)")
    print(f"  Tokenizer output:  {args.tokenizer_out_dir}")
    print(f"  MT sample size:    {args.mt_sample_size}")
    print(f"  Max items:         {args.max_items if args.max_items > 0 else 'ALL'}")
    print(f"  Skip MT:           {args.skip_mt}")
    print(f"  Skip ASR:          {args.skip_asr}")
    print(f"  Random seed:       {args.seed}")
    print()
    print("NOTE: No caching enabled - semantic tokens will be computed on-the-fly")
    print("      This may take longer but requires no additional disk space")
    print()
    
    # Extend Qwen tokenizer with semantic and control tokens
    print("Setting up extended tokenizer...")
    print()
    tokenizer, sem_id_table, ids = setup_extended_tokenizer(
        qwen_name_or_path=args.qwen_name_or_path,
        num_sem_units=NUM_SEM_UNITS,
        tokenizer_out_dir=args.tokenizer_out_dir,
    )
    
    # Load the dataset JSON
    print(f"Loading dataset from {args.json_path}...")
    with open(args.json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)
    print(f"✓ Loaded {len(data)} entries")
    print()
    
    # Determine which samples will get MT examples
    all_audio_files = list(data.keys())
    
    if not args.skip_mt:
        if args.mt_sample_size > 0 and args.mt_sample_size < len(all_audio_files):
            # Randomly select which samples get MT examples
            mt_selected = set(random.sample(all_audio_files, args.mt_sample_size))
            print(f"✓ Randomly selected {len(mt_selected)} samples for MT task")
        else:
            # Use all samples for MT
            mt_selected = set(all_audio_files)
            print(f"✓ Using all {len(all_audio_files)} samples for MT task")
    else:
        mt_selected = set()
        print("✓ MT task disabled (--skip_mt)")
    print()
    
    # Ensure output directory exists
    out_dir = os.path.dirname(args.out_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Counters
    written = 0
    read_n = 0
    skipped = 0
    count_s2st = 0
    count_asr = 0
    count_mt = 0
    
    print("Processing dataset...")
    print("(This may take a while since semantic tokens are computed on-the-fly)")
    print()
    
    # Process each audio file entry
    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        for audio_file, ex in data.items():
            read_n += 1
            
            # Print progress every 50 items
            if read_n % 50 == 0:
                print(f"Progress: {read_n}/{len(data)} entries | "
                      f"Written: {written} examples | "
                      f"Skipped: {skipped} | "
                      f"S2ST: {count_s2st} | ASR: {count_asr} | MT: {count_mt}")
            
            # Stop if we've reached max_items limit
            if args.max_items > 0 and read_n > args.max_items:
                print(f"\n✓ Reached max_items limit ({args.max_items}), stopping.")
                break
            
            # Extract metadata from JSON
            hi_text = ex.get("hi_text", "")
            mai_text = ex.get("mai_text", "")
            ref_audio_path = ex.get("ref_audio_path", "")
            
            # Get audio paths using your custom functions
            hi_audio_path = get_hindi_audio_path(audio_file) + ".wav"
            mai_audio_path = get_maithili_audio_path(audio_file) + ".wav"
            
            # Validate that audio files exist
            if not os.path.exists(hi_audio_path):
                print(f"[WARN] Missing Hindi audio for {audio_file}: {hi_audio_path}")
                skipped += 1
                continue
            
            if not os.path.exists(mai_audio_path):
                print(f"[WARN] Missing Maithili audio for {audio_file}: {mai_audio_path}")
                skipped += 1
                continue
            
            # Extract semantic tokens (no caching)
            try:
                hi_sem = compute_semantic_tokens(hi_audio_path)
                mai_sem = compute_semantic_tokens(mai_audio_path)
            except Exception as e:
                print(f"[ERROR] Failed to extract semantic tokens for {audio_file}: {e}")
                skipped += 1
                continue
            
            # ================================================================
            # TASK 1: Speech-to-Speech Translation (S2ST)
            # This is the main task: Hindi speech -> Maithili speech
            # ================================================================
            try:
                s2st = make_example_sem2sem(
                    task_id=ids["S2ST"],
                    src_sem=hi_sem,
                    tgt_sem=mai_sem,
                    sem_id_table=sem_id_table,
                    ids=ids
                )
                
                out_f.write(json.dumps({
                    "utt_id": audio_file,
                    "task": "S2ST",
                    **s2st,
                    "meta": {
                        "hi_audio": hi_audio_path,
                        "mai_audio": mai_audio_path,
                        "hi_text": hi_text,
                        "mai_text": mai_text,
                    }
                }, ensure_ascii=False) + "\n")
                written += 1
                count_s2st += 1
            except Exception as e:
                print(f"[ERROR] Failed to create S2ST example for {audio_file}: {e}")
                skipped += 1
                continue
            
            # ================================================================
            # TASK 2: Automatic Speech Recognition (ASR)
            # Hindi speech -> Hindi text and Maithili speech -> Maithili text
            # ================================================================
            if not args.skip_asr:
                # ASR for Hindi
                if hi_text:
                    try:
                        asr_hi = make_example_sem2text(
                            task_id=ids["ASR_HI"],
                            src_sem=hi_sem,
                            tgt_text=hi_text,
                            sem_id_table=sem_id_table,
                            tokenizer=tokenizer,
                            ids=ids
                        )
                        
                        out_f.write(json.dumps({
                            "utt_id": audio_file,
                            "task": "ASR_HI",
                            **asr_hi,
                            "meta": {
                                "audio": hi_audio_path,
                                "text": hi_text,
                            },
                        }, ensure_ascii=False) + "\n")
                        written += 1
                        count_asr += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to create ASR_HI example for {audio_file}: {e}")
                
                # ASR for Maithili
                if mai_text:
                    try:
                        asr_ma = make_example_sem2text(
                            task_id=ids["ASR_MA"],
                            src_sem=mai_sem,
                            tgt_text=mai_text,
                            sem_id_table=sem_id_table,
                            tokenizer=tokenizer,
                            ids=ids
                        )
                        
                        out_f.write(json.dumps({
                            "utt_id": audio_file,
                            "task": "ASR_MA",
                            **asr_ma,
                            "meta": {
                                "audio": mai_audio_path,
                                "text": mai_text,
                            },
                        }, ensure_ascii=False) + "\n")
                        written += 1
                        count_asr += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to create ASR_MA example for {audio_file}: {e}")
            
            # ================================================================
            # TASK 3: Machine Translation (MT)
            # Hindi text -> Maithili text
            # CRITICAL: Prevents catastrophic forgetting
            # ================================================================
            if (not args.skip_mt) and hi_text and mai_text and (audio_file in mt_selected):
                try:
                    mt = make_example_text2text(
                        task_id=ids["MT"],
                        src_text=hi_text,
                        tgt_text=mai_text,
                        tokenizer=tokenizer,
                        ids=ids
                    )
                    
                    out_f.write(json.dumps({
                        "utt_id": audio_file,
                        "task": "MT",
                        **mt,
                        "meta": {
                            "src_text": hi_text,
                            "tgt_text": mai_text,
                        },
                    }, ensure_ascii=False) + "\n")
                    written += 1
                    count_mt += 1
                except Exception as e:
                    print(f"[ERROR] Failed to create MT example for {audio_file}: {e}")
    
    # Print final statistics
    print()
    print("=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Statistics:")
    print(f"  Total entries read:      {read_n}")
    print(f"  Total examples written:  {written}")
    print(f"  Skipped entries:         {skipped}")
    print()
    print(f"Task breakdown:")
    print(f"  S2ST examples:           {count_s2st}")
    print(f"  ASR examples:            {count_asr}")
    print(f"  MT examples:             {count_mt}")
    print()
    print(f"Task distribution:")
    if written > 0:
        print(f"  S2ST:  {count_s2st/written*100:5.1f}%")
        print(f"  ASR:   {count_asr/written*100:5.1f}%")
        print(f"  MT:    {count_mt/written*100:5.1f}%")
    print()
    print(f"✓ Output written to: {args.out_jsonl}")
    print(f"✓ Extended tokenizer saved to: {args.tokenizer_out_dir}")
    print()
    print("=" * 80)
    print("IMPORTANT: Next Steps")
    print("=" * 80)
    print()
    print("1. Verify the output JSONL format:")
    print(f"   head -n 1 {args.out_jsonl} | python -m json.tool")
    print()
    print("2. Check the tokenizer was saved correctly:")
    print(f"   ls -la {args.tokenizer_out_dir}")
    print()
    print("3. Use the saved tokenizer in your training script:")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.tokenizer_out_dir}')")
    print()
    print("4. Begin Phase 1 training with this multitask dataset")
    print()


if __name__ == "__main__":
    main()