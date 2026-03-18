#!/usr/bin/env python3
"""
ASR Evaluation Script
Runs multiple ASR models on Hindi audio files and computes WER.
Models:
  1. ai4bharat/indicwav2vec-hindi
  2. ai4bharat/indic-conformer-600m-multilingual
  3. theainerd/Wav2Vec2-large-xlsr-hindi
  4. openai/whisper-large-v3
"""

import os
import re
import csv
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import librosa
import numpy as np
from jiwer import wer, cer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("asr_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
AUDIO_DIR   = "/mnt/storage/aditya/Evaluation/hi_tts_outputs"
TEXT_FILE   = "/mnt/storage/aditya/Evaluation/Benchmarking_dataset/Hindi_test_set.txt"
OUTPUT_DIR  = "./asr_results"
SAMPLE_RATE = 16000

MODELS = {
    # "indicwav2vec_hindi": {
    #     "model_id": "ai4bharat/indicwav2vec-hindi",
    #     "type": "wav2vec2",
    #     "description": "Wav2Vec2-based, trained on large Hindi speech corpora",
    # },
    "indic_conformer_600m": {
        "model_id": "ai4bharat/indic-conformer-600m-multilingual",
        "type": "nemo_conformer",
        "description": "Conformer architecture covering 22 Indic languages",
        "lang": "hi",
    },
    # "wav2vec2_xlsr_hindi": {
    #     "model_id": "theainerd/Wav2Vec2-large-xlsr-hindi",
    #     "type": "wav2vec2",
    #     "description": "XLS-R pretrained on Hindi, robust for noisy audio",
    # },
    # "whisper_large_v3": {
    #     "model_id": "openai/whisper-large-v3",
    #     "type": "whisper",
    #     "description": "OpenAI Whisper large-v3",
    #     "language": "hi",
    # },
}

# ─────────────────────────────────────────────
# TEXT NORMALISATION (Hindi)
# ─────────────────────────────────────────────
def normalize_hindi(text: str) -> str:
    """Minimal normalisation: strip punctuation, collapse whitespace, lowercase."""
    if not text:
        return ""
    # Remove punctuation (keep Devanagari characters, spaces, digits)
    text = re.sub(r"[।॥,\.\!\?\"\'\(\)\[\]\{\}\:\;\-–—]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_references(text_file: str) -> Dict[str, str]:
    """
    Load reference transcriptions.
    Expected format (one per line):
        utt_00001  <reference text>
    OR plain lines (index-aligned with sorted audio files).
    """
    references = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)   # split on first whitespace
            if len(parts) == 2 and parts[0].startswith("utt_"):
                utt_id, text = parts[0], parts[1]
            else:
                # No explicit ID – use positional key
                utt_id = f"utt_{i:05d}"
                text = line
            references[utt_id] = normalize_hindi(text)
    logger.info(f"Loaded {len(references)} reference transcriptions from {text_file}")
    return references


def load_audio_files(audio_dir: str) -> List[Tuple[str, str]]:
    """Return sorted list of (utt_id, filepath) tuples."""
    exts = {".wav", ".flac", ".mp3", ".ogg"}
    files = sorted([
        p for p in Path(audio_dir).iterdir()
        if p.suffix.lower() in exts
    ])
    pairs = []
    for p in files:
        # Derive utt_id from filename stem
        stem = p.stem   # e.g. utt_00001
        pairs.append((stem, str(p)))
    logger.info(f"Found {len(pairs)} audio files in {audio_dir}")
    return pairs


def load_audio(filepath: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = librosa.load(filepath, sr=sample_rate, mono=True)
    return audio


# ─────────────────────────────────────────────
# MODEL RUNNERS
# ─────────────────────────────────────────────
class Wav2Vec2Runner:
    def __init__(self, model_id: str):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        logger.info(f"Loading Wav2Vec2 model: {model_id}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        logger.info(f"  → device: {self.device}")

    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return normalize_hindi(transcription)


class WhisperRunner:
    def __init__(self, model_id: str, language: str = "hi"):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        logger.info(f"Loading Whisper model: {model_id}")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.language = language
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )
        logger.info(f"  → device: {self.device}")

    def transcribe(self, audio: np.ndarray) -> str:
        input_features = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.to(self.device)

        if self.device == "cuda":
            input_features = input_features.half()

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids
            )

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return normalize_hindi(transcription)


class NemoConformerRunner:
    """
    Wrapper for ai4bharat/indic-conformer-600m-multilingual via NeMo.
    Falls back to HuggingFace pipeline if NeMo is not installed.
    """
    def __init__(self, model_id: str, lang: str = "hi"):
        self.lang = lang
        try:
            import nemo.collections.asr as nemo_asr
            logger.info(f"Loading NeMo Conformer model: {model_id}")
            # For HuggingFace-hosted NeMo models, download config + weights
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_id)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device).eval()
            self.backend = "nemo"
            logger.info(f"  → NeMo backend, device: {self.device}")
        except ImportError:
            logger.warning("NeMo not found – falling back to HuggingFace pipeline for indic-conformer.")
            from transformers import pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=0 if torch.cuda.is_available() else -1
            )
            self.backend = "hf_pipeline"

    def transcribe(self, audio: np.ndarray) -> str:
        if self.backend == "nemo":
            import tempfile, soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, SAMPLE_RATE)
                tmp_path = tmp.name
            result = self.model.transcribe([tmp_path])
            os.unlink(tmp_path)
            text = result[0] if isinstance(result[0], str) else result[0][0]
        else:
            result = self.pipe({"array": audio, "sampling_rate": SAMPLE_RATE})
            text = result["text"]
        return normalize_hindi(text)


# ─────────────────────────────────────────────
# EVALUATION CORE
# ─────────────────────────────────────────────
def evaluate_model(
    model_name: str,
    model_cfg: dict,
    audio_pairs: List[Tuple[str, str]],
    references: Dict[str, str],
    output_dir: str,
    skip_missing: bool = True,
) -> dict:

    os.makedirs(output_dir, exist_ok=True)

    # ── Instantiate runner ──
    mtype = model_cfg["type"]
    mid   = model_cfg["model_id"]

    try:
        if mtype == "wav2vec2":
            runner = Wav2Vec2Runner(mid)
        elif mtype == "whisper":
            runner = WhisperRunner(mid, language=model_cfg.get("language", "hi"))
        elif mtype == "nemo_conformer":
            runner = NemoConformerRunner(mid, lang=model_cfg.get("lang", "hi"))
        else:
            raise ValueError(f"Unknown model type: {mtype}")
    except Exception as e:
        logger.error(f"[{model_name}] Failed to load model: {e}")
        return {"model": model_name, "error": str(e)}

    all_refs, all_hyps = [], []
    per_utt_results = []
    total_time = 0.0
    errors = 0

    detail_path = os.path.join(output_dir, f"{model_name}_details.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["utt_id", "reference", "hypothesis", "utt_wer", "utt_cer", "elapsed_s"]
        )
        writer.writeheader()

        for utt_id, fpath in tqdm(audio_pairs, desc=f"[{model_name}]", unit="utt"):
            ref = references.get(utt_id, "")
            if skip_missing and not ref:
                logger.debug(f"  Skip {utt_id}: no reference found")
                continue

            # Load audio
            try:
                audio = load_audio(fpath)
            except Exception as e:
                logger.warning(f"  [{utt_id}] Audio load error: {e}")
                errors += 1
                continue

            # Transcribe
            t0 = time.time()
            try:
                hyp = runner.transcribe(audio)
            except Exception as e:
                logger.warning(f"  [{utt_id}] Transcription error: {e}")
                hyp = ""
                errors += 1
            elapsed = time.time() - t0
            total_time += elapsed

            # Per-utterance metrics
            utt_wer = wer(ref, hyp) if ref else float("nan")
            utt_cer = cer(ref, hyp) if ref else float("nan")

            all_refs.append(ref)
            all_hyps.append(hyp)

            writer.writerow({
                "utt_id": utt_id,
                "reference": ref,
                "hypothesis": hyp,
                "utt_wer": f"{utt_wer:.4f}",
                "utt_cer": f"{utt_cer:.4f}",
                "elapsed_s": f"{elapsed:.3f}",
            })

    # ── Corpus-level metrics ──
    corpus_wer = wer(all_refs, all_hyps) if all_refs else float("nan")
    corpus_cer = cer(all_refs, all_hyps) if all_refs else float("nan")
    rtf = total_time / max(len(audio_pairs), 1)

    summary = {
        "model": model_name,
        "model_id": mid,
        "description": model_cfg.get("description", ""),
        "num_utterances": len(all_refs),
        "errors": errors,
        "corpus_wer": round(corpus_wer * 100, 2),   # percent
        "corpus_cer": round(corpus_cer * 100, 2),
        "total_inference_s": round(total_time, 2),
        "avg_rtf_s_per_utt": round(rtf, 3),
        "detail_csv": detail_path,
    }

    logger.info(
        f"[{model_name}] WER: {summary['corpus_wer']:.2f}%  "
        f"CER: {summary['corpus_cer']:.2f}%  "
        f"Utterances: {summary['num_utterances']}  "
        f"Errors: {errors}"
    )
    return summary


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────
def generate_report(summaries: List[dict], output_dir: str):
    report_path = os.path.join(output_dir, "asr_evaluation_report.txt")
    json_path   = os.path.join(output_dir, "asr_evaluation_report.json")
    csv_path    = os.path.join(output_dir, "asr_evaluation_summary.csv")

    # Sort by WER ascending
    valid = [s for s in summaries if "corpus_wer" in s]
    valid.sort(key=lambda x: x["corpus_wer"])

    # ── Plain text report ──
    sep = "=" * 72
    lines = [
        sep,
        "  ASR EVALUATION REPORT – Hindi TTS Outputs",
        sep,
        "",
        f"{'Model':<35} {'WER %':>8} {'CER %':>8} {'Utts':>6} {'Errs':>6}",
        "-" * 72,
    ]
    for s in valid:
        lines.append(
            f"{s['model']:<35} {s['corpus_wer']:>8.2f} {s['corpus_cer']:>8.2f} "
            f"{s['num_utterances']:>6} {s['errors']:>6}"
        )
    lines += ["", sep, "DETAILS", sep, ""]
    for s in valid:
        lines += [
            f"Model       : {s['model']}",
            f"  HF ID     : {s['model_id']}",
            f"  Desc      : {s['description']}",
            f"  WER       : {s['corpus_wer']:.2f} %",
            f"  CER       : {s['corpus_cer']:.2f} %",
            f"  Utterances: {s['num_utterances']}",
            f"  Errors    : {s['errors']}",
            f"  Infer (s) : {s['total_inference_s']}",
            f"  Avg RTF   : {s['avg_rtf_s_per_utt']} s/utt",
            f"  Detail CSV: {s['detail_csv']}",
            "",
        ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report written to {report_path}")

    # ── JSON ──
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    # ── Summary CSV ──
    fieldnames = [
        "model", "model_id", "corpus_wer", "corpus_cer",
        "num_utterances", "errors", "total_inference_s", "avg_rtf_s_per_utt"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid)

    print("\n" + "\n".join(lines))
    return report_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hindi ASR Evaluation Pipeline")
    parser.add_argument("--audio_dir",  default=AUDIO_DIR)
    parser.add_argument("--text_file",  default=TEXT_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Subset of models to evaluate"
    )
    parser.add_argument(
        "--max_utts", type=int, default=None,
        help="Limit number of utterances (for quick testing)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──
    references  = load_references(args.text_file)
    audio_pairs = load_audio_files(args.audio_dir)

    if args.max_utts:
        audio_pairs = audio_pairs[:args.max_utts]
        logger.info(f"Limiting to {args.max_utts} utterances.")

    # ── Run each model ──
    summaries = []
    for model_name in args.models:
        model_cfg = MODELS[model_name]
        logger.info(f"\n{'='*60}\nRunning: {model_name}\n{'='*60}")
        summary = evaluate_model(
            model_name=model_name,
            model_cfg=model_cfg,
            audio_pairs=audio_pairs,
            references=references,
            output_dir=args.output_dir,
        )
        summaries.append(summary)

        # Free GPU memory between models
        torch.cuda.empty_cache()

    # ── Generate report ──
    generate_report(summaries, args.output_dir)


if __name__ == "__main__":
    main()