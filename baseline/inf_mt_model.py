"""
Batch Hindi -> Maithili translation using IndicTrans2 checkpoint.

Usage:
    python inf_mt_model.py \
        --checkpoint  /mnt/storage/aditya/bl/checkpoint-5000 \
        --input       /mnt/storage/aditya/bl/indicwav2vec_hindi_details.csv \
        --output      /mnt/storage/aditya/bl/output_bl.txt \
        [--batch-size 16] \
        [--max-new-tokens 256] \
        [--src-lang hin_Deva] \
        [--tgt-lang mai_Deva]
"""

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys

import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _dtype(device: str):
    return torch.float16 if "cuda" in device else torch.float32


def _safe_load_indic_processor():
    """
    Probe IndicProcessor in a child process first — Cython segfaults cannot
    be caught with try/except in-process. Returns instance or None.
    """
    probe = (
        "import sys\n"
        "for pkg in ['IndicTransToolkit', 'IndicTransTokenizer']:\n"
        "    try:\n"
        "        mod = __import__(pkg, fromlist=['IndicProcessor'])\n"
        "        mod.IndicProcessor(inference=True)\n"
        "        print(pkg)\n"
        "        sys.exit(0)\n"
        "    except Exception:\n"
        "        continue\n"
        "sys.exit(1)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            pkg_name = result.stdout.strip()
            mod = importlib.import_module(pkg_name)
            proc = mod.IndicProcessor(inference=True)
            logger.info(f"[MT] IndicProcessor loaded from '{pkg_name}' — Path A enabled.")
            return proc
    except Exception as e:
        logger.warning(f"[MT] IndicProcessor probe failed: {e}")

    logger.warning(
        "[MT] IndicProcessor unavailable — using Path B (raw tokenizer).\n"
        "     To fix: pip install 'numpy<2.0' then recompile IndicTransToolkit."
    )
    return None


# ─── tokenizer loading ────────────────────────────────────────────────────────

def _load_tokenizer(tokenizer_src: str):
    """
    Load the IndicTrans2 tokenizer. The tokenizer class (IndicTransTokenizer)
    is not registered with HuggingFace, so it must be fetched via
    trust_remote_code=True from the HF hub — exactly as in example.py.
    tokenizer_src should always be the HF base model ID, never a local
    checkpoint path (local checkpoints don't contain tokenization_indictrans.py).
    """
    from transformers import AutoTokenizer
    logger.info(f"[Tokenizer] Loading from: {tokenizer_src}")
    tok = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    logger.info("[Tokenizer] Loaded via AutoTokenizer (trust_remote_code).")
    return tok


# ─── model loading ────────────────────────────────────────────────────────────

def load_mt_model(checkpoint_path: str, device: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info(f"[MT] Loading checkpoint: {checkpoint_path} -> {device}")
    adapter_cfg_path = os.path.join(checkpoint_path, "adapter_config.json")

    if os.path.exists(adapter_cfg_path):
        from peft import PeftModel
        with open(adapter_cfg_path) as f:
            cfg = json.load(f)
        base_name = cfg.get("base_model_name_or_path", "ai4bharat/indictrans2-indic-indic-1B")
        logger.info(f"[MT] PEFT adapter detected. Base: {base_name}")
        base = AutoModelForSeq2SeqLM.from_pretrained(
            base_name, torch_dtype=_dtype(device), low_cpu_mem_usage=True
        )
        mt_model = PeftModel.from_pretrained(base, checkpoint_path)
        mt_model = mt_model.merge_and_unload()
        tokenizer_src = base_name
    else:
        mt_model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_path, torch_dtype=_dtype(device), low_cpu_mem_usage=True
        )
        # Full Trainer checkpoint — tokenizer was never saved into the
        # checkpoint dir. Always load it from the original HF base model.
        tokenizer_src = "ai4bharat/indictrans2-indic-indic-1B"

    # ── Spread across all available GPUs ──────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        logger.info(f"[MT] Wrapping model with DataParallel across {n_gpus} GPUs.")
        mt_model = mt_model.to(device)
        mt_model = nn.DataParallel(mt_model)
    else:
        mt_model = mt_model.to(device)

    mt_model.eval()

    mt_tokenizer = _load_tokenizer(tokenizer_src)
    indic_proc   = _safe_load_indic_processor()

    logger.info("[MT] Model ready.")
    return mt_model, mt_tokenizer, indic_proc


# ─── translation (batch-aware) ────────────────────────────────────────────────

def translate_batch(
    mt_model,
    mt_tokenizer,
    indic_proc,
    texts: list[str],
    device: str,
    src_lang: str = "hin_Deva",
    tgt_lang: str = "mai_Deva",
    max_new_tokens: int = 256,
) -> list[str]:
    """Translate a list of Hindi strings to Maithili in one forward pass."""

    # Unwrap DataParallel to call .generate() cleanly
    raw_model = mt_model.module if isinstance(mt_model, nn.DataParallel) else mt_model

    if indic_proc is not None:
        # ── Path A: IndicProcessor ────────────────────────────────────────────
        batch = indic_proc.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = mt_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            out_ids = raw_model.generate(**inputs, num_beams=5, max_new_tokens=max_new_tokens)

        raw_decoded = mt_tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return [s.strip() for s in indic_proc.postprocess_batch(raw_decoded, lang=tgt_lang)]

    else:
        # ── Path B: raw IndicTransTokenizer ──────────────────────────────────
        tagged = [f"{src_lang} {tgt_lang} {t}" for t in texts]
        inputs = mt_tokenizer(
            tagged,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        forced_bos = None
        if hasattr(mt_tokenizer, "tgt_encoder"):
            forced_bos = mt_tokenizer.tgt_encoder.get(tgt_lang)
        if forced_bos is None and hasattr(mt_tokenizer, "lang_code_to_id"):
            forced_bos = mt_tokenizer.lang_code_to_id.get(tgt_lang)

        gen_kwargs = dict(num_beams=5, max_new_tokens=max_new_tokens)
        if forced_bos is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos

        with torch.no_grad():
            out_ids = raw_model.generate(**inputs, **gen_kwargs)

        return [
            mt_tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in out_ids
        ]


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch Hindi->Maithili translation")
    parser.add_argument("--checkpoint",     required=True,  help="Path to model checkpoint")
    parser.add_argument("--input",          required=True,  help="Path to input CSV file")
    parser.add_argument("--output",         required=True,  help="Path to output .txt file")
    parser.add_argument("--batch-size",     type=int, default=16,  help="Sentences per batch (default: 16)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per sentence")
    parser.add_argument("--src-lang",       default="hin_Deva")
    parser.add_argument("--tgt-lang",       default="mai_Deva")
    parser.add_argument("--column",         default="hypothesis", help="CSV column to translate")
    args = parser.parse_args()

    # ── Device setup ──────────────────────────────────────────────────────────
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    logger.info(f"[Setup] Device: {device} | GPUs available: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {props.name}  ({props.total_memory / 1e9:.1f} GB)")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found. Available: {list(df.columns)}")

    sentences = df[args.column].astype(str).tolist()
    logger.info(f"[Data] {len(sentences)} sentences to translate.")

    # ── Load model ONCE ───────────────────────────────────────────────────────
    mt_model, mt_tokenizer, indic_proc = load_mt_model(args.checkpoint, device)

    # ── Translate in batches ──────────────────────────────────────────────────
    results = []
    total   = len(sentences)

    for start in range(0, total, args.batch_size):
        end   = min(start + args.batch_size, total)
        batch = sentences[start:end]

        logger.info(f"[Translate] Sentences {start+1}-{end} / {total}")

        translations = translate_batch(
            mt_model, mt_tokenizer, indic_proc,
            batch, device,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_new_tokens=args.max_new_tokens,
        )
        results.extend(translations)

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    logger.info(f"[Done] {len(results)} translations written to: {args.output}")


if __name__ == "__main__":
    main()