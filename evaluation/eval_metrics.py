"""
eval_metrics.py — Evaluation suite for generated Maithili audio.

Metrics:
  Text  : BLEU (char), chrF
  Speech: ASV Score, NISQA

Outputs:
  <out_dir>/translations_from_asr.txt  — all ASR transcriptions
  <out_dir>/results.txt                — tabular results

Usage:
    python3 nar_training/eval_bleu.py \
        --gen_dir     /mnt/storage/aditya/results_from_tts_metric/parler_results/audio \
        --ref_dir     /mnt/storage/aditya/Evaluation/mai_tts_outputs_gt \
        --transcripts /mnt/storage/aditya/results_from_tts_metric/test_translations_parler.json \
        --out_dir     /mnt/storage/aditya/results_from_tts_metric/outputs_parler \
        --device      cuda

Dependencies:
    pip install sacrebleu transformers torchaudio nisqa
"""

import os
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    AutoModel,
    AutoFeatureExtractor,
    AutoModelForAudioXVector,
)

warnings.filterwarnings("ignore")

# ── ASR ───────────────────────────────────────────────────────────────────────
ASR_MODEL_ID    = "ai4bharat/indic-conformer-600m-multilingual"
ASR_LANGUAGE    = "mai"
ASR_DECODE_MODE = "rnnt"
ASR_SAMPLE_RATE = 16000

# ── ASV ───────────────────────────────────────────────────────────────────────
ASV_MODEL_ID    = "microsoft/unispeech-sat-base-plus-sv"
ASV_SAMPLE_RATE = 16000


# ==============================================================================
# Audio loading helpers
# ==============================================================================

def load_wav_mono_np(path, target_sr):
    """Load audio -> mono numpy float32 array at target_sr."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).numpy().astype(np.float32)


# ==============================================================================
# ASR
# ==============================================================================

def load_asr_model(device):
    model = AutoModel.from_pretrained(
        ASR_MODEL_ID,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


def transcribe(asr_model, audio_path, device):
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != ASR_SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, ASR_SAMPLE_RATE)(wav)
    with torch.no_grad():
        result = asr_model(wav.to(device), ASR_LANGUAGE, ASR_DECODE_MODE)
    return result.strip()


# ==============================================================================
# ASV Score (Speaker Similarity)
# ==============================================================================

def load_asv_model(device):
    extractor = AutoFeatureExtractor.from_pretrained(ASV_MODEL_ID)
    model     = AutoModelForAudioXVector.from_pretrained(ASV_MODEL_ID).to(device)
    model.eval()
    return model, extractor


def compute_asv(ref_path, gen_path, asv_model, asv_extractor, device):
    """
    Cosine similarity between speaker embeddings of ref and gen audio.
    Range: -1 to 1. Higher = more similar speaker voice.
    Returns None if either file is missing or an error occurs.
    """
    ref_wav = load_wav_mono_np(ref_path, ASV_SAMPLE_RATE)
    gen_wav = load_wav_mono_np(gen_path, ASV_SAMPLE_RATE)

    inputs_ref = asv_extractor(
        ref_wav, sampling_rate=ASV_SAMPLE_RATE,
        return_tensors="pt", padding=True,
    )
    inputs_gen = asv_extractor(
        gen_wav, sampling_rate=ASV_SAMPLE_RATE,
        return_tensors="pt", padding=True,
    )

    with torch.no_grad():
        emb_ref = asv_model(**{k: v.to(device) for k, v in inputs_ref.items()}).embeddings
        emb_gen = asv_model(**{k: v.to(device) for k, v in inputs_gen.items()}).embeddings

    emb_ref = F.normalize(emb_ref, dim=-1)
    emb_gen = F.normalize(emb_gen, dim=-1)

    score = F.cosine_similarity(emb_ref, emb_gen).item()

    # Guard against nan/inf that can arise from silent or corrupt audio
    if not np.isfinite(score):
        return None

    return round(score, 4)


# ==============================================================================
# NISQA (Speech Naturalness / MOS prediction)
# ==============================================================================

def load_nisqa_model(dummy_audio_path):
    from nisqa.NISQA_model import nisqaModel
    args = {
        "mode"             : "predict_file",
        "pretrained_model" : "/mnt/storage/aditya/nisqa_weights/nisqa.tar",
        "deg"              : dummy_audio_path,
        "data_dir"         : None,
        "output_dir"       : None,
        "bs"               : 10,
        "ms_channel"       : None,
    }
    return nisqaModel(args)


def compute_nisqa(gen_path, nisqa_model):
    """
    Predicted MOS score. Range: 1-5. Higher = more natural speech.
    Returns None if prediction fails or result is non-finite.
    """
    nisqa_model.args["deg"] = gen_path
    nisqa_model.predict()

    # nisqaModel stores results in self.ds_val.df after predict()
    df = nisqa_model.ds_val.df
    if df is None or len(df) == 0 or "mos_pred" not in df.columns:
        return None

    mos = float(df["mos_pred"].values[0])
    if not np.isfinite(mos):
        return None

    return round(mos, 4)


# ==============================================================================
# Safe wrapper
# ==============================================================================

def safe(fn, label, fname, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  {label} error [{fname}]: {e}")
        return None


# ==============================================================================
# Aggregation helper — skips None and nan
# ==============================================================================

def safe_mean(vals):
    """Return mean of finite values, ignoring None and nan."""
    clean = [v for v in vals if v is not None and np.isfinite(v)]
    return round(float(np.mean(clean)), 4) if clean else None


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir",     required=True,
                        help="Folder of generated Maithili .wav files")
    parser.add_argument("--ref_dir",     required=True,
                        help="Folder of reference Maithili .wav files")
    parser.add_argument("--transcripts", required=True,
                        help="Path to test_transcripts.json")
    parser.add_argument("--out_dir",     default="/mnt/storage/aditya/outputs")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    device = "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    trans_path   = os.path.join(args.out_dir, "translations_from_asr.txt")
    results_path = os.path.join(args.out_dir, "results.txt")

    # ── Load transcripts ───────────────────────────────────────────────────────
    with open(args.transcripts, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    # ── Collect entries ────────────────────────────────────────────────────────
    entries = []
    for fname, info in sorted(transcripts.items()):
        gen_path = os.path.join(args.gen_dir, fname)
        ref_path = os.path.join(args.ref_dir, fname)
        if not os.path.exists(gen_path):
            print(f"  WARNING: generated {fname} not found — skipping")
            continue
        if not os.path.exists(ref_path):
            print(f"  WARNING: reference {fname} not found — ASV will be N/A")
            ref_path = None
        entries.append({
            "fname"    : fname,
            "gen_path" : gen_path,
            "ref_path" : ref_path,
            "mai_ref"  : info["mai_text"],
        })

    print(f"\nFound {len(entries)} files to evaluate")

    # ── Load all models once ───────────────────────────────────────────────────
    print("Loading ASR model...")
    asr = load_asr_model(device)
    print("  ASR loaded.")

    print("Loading ASV model...")
    asv_model, asv_extractor = load_asv_model(device)
    print("  ASV loaded.")

    print("Loading NISQA model...")
    nisqa = load_nisqa_model(entries[0]["gen_path"])
    print("  NISQA loaded.\n")

    # ── Per-file evaluation ────────────────────────────────────────────────────
    hypotheses = []
    references = []
    per_file   = []

    for entry in tqdm(entries, desc="Evaluating", unit="audio"):
        fname    = entry["fname"]
        gen_path = entry["gen_path"]
        ref_path = entry["ref_path"]

        hyp = transcribe(asr, gen_path, device)
        ref = entry["mai_ref"]
        hypotheses.append(hyp)
        references.append(ref)

        nisqa_score = safe(compute_nisqa, "NISQA", fname, gen_path, nisqa)

        if ref_path:
            asv_score = safe(compute_asv, "ASV", fname,
                             ref_path, gen_path, asv_model, asv_extractor, device)
        else:
            asv_score = None

        per_file.append({
            "fname" : fname,
            "hyp"   : hyp,
            "ref"   : ref,
            "asv"   : asv_score,
            "nisqa" : nisqa_score,
        })

    # ── Corpus-level text metrics ──────────────────────────────────────────────
    # tokenize="char" for character-level BLEU (correct for Maithili)
    bleu_scorer = BLEU(tokenize="none")
    chrf_scorer = CHRF()
    bleu_score  = bleu_scorer.corpus_score(hypotheses, [references])
    chrf_score  = chrf_scorer.corpus_score(hypotheses, [references])

    # ── Aggregate speech metrics (skip None / nan) ─────────────────────────────
    avg_asv   = safe_mean([r["asv"]   for r in per_file])
    avg_nisqa = safe_mean([r["nisqa"] for r in per_file])

    # Count how many samples contributed to each average
    asv_valid   = sum(1 for r in per_file if r["asv"]   is not None and np.isfinite(r["asv"]))
    nisqa_valid = sum(1 for r in per_file if r["nisqa"] is not None and np.isfinite(r["nisqa"]))

    def fv(v):
        return f"{v:.4f}" if v is not None else "N/A"

    # ── Save translations_from_asr.txt ─────────────────────────────────────────
    with open(trans_path, "w", encoding="utf-8") as f:
        f.write(f"{'FILE':<20}  ASR TRANSCRIPTION\n")
        f.write("─" * 100 + "\n")
        for r in per_file:
            f.write(f"{r['fname']:<20}  {r['hyp']}\n")

    # ── Save results.txt ───────────────────────────────────────────────────────
    cw = [20, 10, 10, 10, 10]   # FILE  BLEU  chrF  ASV  NISQA

    def make_row(fname, bleu, chrf, asv, nisqa):
        return (
            f"{fname:<{cw[0]}}"
            f"{fv(bleu):>{cw[1]}}"
            f"{fv(chrf):>{cw[2]}}"
            f"{fv(asv):>{cw[3]}}"
            f"{fv(nisqa):>{cw[4]}}"
        )

    header = (
        f"{'FILE':<{cw[0]}}"
        f"{'BLEU':>{cw[1]}}"
        f"{'chrF':>{cw[2]}}"
        f"{'ASV':>{cw[3]}}"
        f"{'NISQA':>{cw[4]}}"
    )
    sep = "─" * sum(cw)

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("═" * 60 + "\n")
        f.write("  EVALUATION RESULTS\n")
        f.write("═" * 60 + "\n")
        f.write(f"  Files evaluated : {len(per_file)}\n\n")

        f.write("  TEXT METRICS (corpus-level)\n")
        f.write(f"  {'BLEU (char)':<20}: {bleu_score.score:.2f}\n")
        f.write(f"  {'chrF':<20}: {chrf_score.score:.2f}\n\n")

        f.write("  SPEECH QUALITY METRICS (avg over valid files)\n")
        f.write(f"  {'ASV Score':<20}: {fv(avg_asv):<10} "
                f"higher is better (max 1.0)  [{asv_valid}/{len(per_file)} valid]\n")
        f.write(f"  {'NISQA (MOS)':<20}: {fv(avg_nisqa):<10} "
                f"higher is better (max 5.0)  [{nisqa_valid}/{len(per_file)} valid]\n")
        f.write("═" * 60 + "\n\n")

        f.write("  PER-FILE RESULTS\n\n")
        f.write(header + "\n")
        f.write(sep    + "\n")

        for r in per_file:
            try:
                pf_bleu = bleu_scorer.sentence_score(r["hyp"], [r["ref"]]).score
                pf_chrf = chrf_scorer.sentence_score(r["hyp"], [r["ref"]]).score
            except Exception:
                pf_bleu = pf_chrf = None

            print(f"{r['fname']}  BLEU: {fv(pf_bleu)}  REF: {r['ref']}  HYP: {r['hyp']}")

            f.write(make_row(r["fname"], pf_bleu, pf_chrf,
                             r["asv"], r["nisqa"]) + "\n")

        f.write(sep + "\n")
        f.write(make_row(
            "OVERALL",
            bleu_score.score, chrf_score.score,
            avg_asv, avg_nisqa,
        ) + "\n")
        f.write(sep + "\n")

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  BLEU (char)  : {bleu_score.score:.2f}")
    print(f"  chrF         : {chrf_score.score:.2f}")
    print(f"  ASV Score    : {fv(avg_asv)}  [{asv_valid}/{len(per_file)} valid samples]")
    print(f"  NISQA (MOS)  : {fv(avg_nisqa)}  [{nisqa_valid}/{len(per_file)} valid samples]")
    print(f"{'═'*60}")
    print(f"\n  Translations : {trans_path}")
    print(f"  Results      : {results_path}")


if __name__ == "__main__":
    main()