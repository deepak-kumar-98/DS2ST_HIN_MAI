import os
import re
import json
import unicodedata
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from jiwer import wer, cer
from transformers import AutoModel

AUDIO_DIR   = "/mnt/storage/aditya/Evaluation/hi_tts_outputs"
TEXT_FILE   = "/mnt/storage/aditya/Evaluation/Benchmarking_dataset/Hindi_test_set.txt"
OUTPUT_DIR  = "./asr_results"

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
LANG_ID = "hi"
DECODER = "rnnt"   # or "ctc"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
TARGET_SR = 16000


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.strip().lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_references(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def collect_audio_files(audio_dir):
    return [
        str(p) for p in sorted(Path(audio_dir).iterdir())
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]


def load_audio(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    return wav


def main():
    ensure_dir(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading model:", MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    audio_files = collect_audio_files(AUDIO_DIR)
    references = load_references(TEXT_FILE)

    if not audio_files:
        raise ValueError(f"No audio files found in {AUDIO_DIR}")

    n = min(len(audio_files), len(references))
    if len(audio_files) != len(references):
        print(f"Warning: {len(audio_files)} audio files vs {len(references)} references. Using first {n}.")

    audio_files = audio_files[:n]
    references = references[:n]

    rows = []
    norm_refs = []
    norm_hyps = []

    for wav_path, ref in tqdm(list(zip(audio_files, references)), total=n, desc="Transcribing"):
        wav = load_audio(wav_path).to(device)

        with torch.inference_mode():
            pred = model(wav, LANG_ID, DECODER)

        ref_n = normalize_text(ref)
        hyp_n = normalize_text(pred)

        norm_refs.append(ref_n)
        norm_hyps.append(hyp_n)

        rows.append({
            "audio_path": wav_path,
            "reference": ref,
            "prediction": pred,
            "reference_normalized": ref_n,
            "prediction_normalized": hyp_n,
        })

    summary = {
        "model_used": MODEL_ID,
        "language_id": LANG_ID,
        "decoder": DECODER,
        "num_samples": n,
        "wer": wer(norm_refs, norm_hyps),
        "cer": cer(norm_refs, norm_hyps),
    }

    with open(os.path.join(OUTPUT_DIR, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
