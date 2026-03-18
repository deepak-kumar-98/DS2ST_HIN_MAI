"""
facebook/mms-tts-mai  →  ASR  →  BLEU
=======================================
Steps:
  1. Read predicted text (one sentence per line)
  2. Synthesize audio with facebook/mms-tts-mai
  3. Transcribe audio with ai4bharat/indic-conformer-600m-multilingual
  4. Compute BLEU score (tokenize="none") against ground truth

Usage:
  python eval_mms.py \
      --ground_truth /mnt/storage/aditya/Evaluation/Benchmarking_dataset/mai_gt_bl.txt \
      --predicted    /mnt/storage/aditya/Evaluation/bl/mai_output_from_mt.txt \
      --output_dir   /mnt/storage/aditya/results_from_tts_metric/mms_results
"""

import argparse
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
 
from transformers import VitsModel, AutoTokenizer
from transformers import AutoModel
from sacrebleu.metrics import BLEU
 
# ─────────────────────────────────────────────────────────────────────────────
 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TTS_MODEL   = "facebook/mms-tts-mai"
ASR_MODEL   = "ai4bharat/indic-conformer-600m-multilingual"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────
 
def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — TTS: facebook/mms-tts-mai
# ─────────────────────────────────────────────────────────────────────────────
 
def run_tts(sentences: list[str], out_dir: Path):
    print(f"\n[TTS] Loading {TTS_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
    model     = VitsModel.from_pretrained(TTS_MODEL).to(DEVICE)
    model.eval()
 
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TTS] Synthesising {len(sentences)} sentences -> {out_dir}/")
 
    for idx, text in enumerate(tqdm(sentences, desc="MMS-TTS")):
        inputs   = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model(**inputs)
        waveform = output.waveform[0].cpu().numpy()
        sr       = model.config.sampling_rate
        sf.write(out_dir / f"{idx:05d}.wav", waveform, sr)
 
    print(f"[TTS] Done. {len(sentences)} audio files saved to {out_dir}/")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ASR: ai4bharat/indic-conformer-600m-multilingual
# ─────────────────────────────────────────────────────────────────────────────
 
def run_asr(audio_dir: Path, out_txt: Path, target_sr: int = 16_000) -> list[str]:
    print(f"\n[ASR] Loading {ASR_MODEL} ...")
    # Official inference pattern from ai4bharat/indic-conformer-600m-multilingual HuggingFace page:
    #   model = AutoModel.from_pretrained(..., trust_remote_code=True)
    #   wav, sr = torchaudio.load("audio.wav")
    #   transcription = model(wav, "mai", "ctc")   # language code, decoding mode
    # No AutoProcessor or AutoModelForCTC — the model handles everything internally.
    asr_model = AutoModel.from_pretrained(ASR_MODEL, trust_remote_code=True)
    asr_model.eval()
 
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}/")
 
    print(f"[ASR] Transcribing {len(wav_files)} files ...")
    transcriptions = []
 
    for wav_path in tqdm(wav_files, desc="ASR"):
        wav, sr = torchaudio.load(str(wav_path))
        # Convert to mono by averaging channels
        wav = torch.mean(wav, dim=0, keepdim=True)
        # Resample to 16kHz if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = resampler(wav)
        with torch.no_grad():
            transcript = asr_model(wav, "mai", "ctc")
        transcriptions.append(transcript.strip())
 
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(transcriptions) + "\n")
 
    print(f"[ASR] Transcriptions saved -> {out_txt}")
    return transcriptions
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BLEU
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    assert len(hypotheses) == len(references), (
        f"[BLEU] Length mismatch: {len(hypotheses)} hyps vs {len(references)} refs"
    )
    bleu   = BLEU(tokenize="none")
    result = bleu.corpus_score(hypotheses, [references])
 
    print("\n" + "=" * 60)
    print("  BLEU RESULT  —  facebook/mms-tts-mai")
    print("=" * 60)
    print(f"  Score  : {result.score:.4f}")
    print(f"  Detail : {result}")
    print("=" * 60)
    return result.score
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="facebook/mms-tts-mai -> ASR -> BLEU pipeline"
    )
    parser.add_argument("--ground_truth", required=True,
                        help="Ground-truth Maithili text file (one sentence per line)")
    parser.add_argument("--predicted",    required=True,
                        help="Predicted Maithili text file (one sentence per line)")
    parser.add_argument("--output_dir",   default="mms_results",
                        help="Directory for audio files and transcriptions (default: ./mms_results)")
    parser.add_argument("--skip_tts", action="store_true",
                        help="Skip TTS synthesis (reuse existing audio in output_dir/audio/)")
    parser.add_argument("--skip_asr", action="store_true",
                        help="Skip ASR (reuse existing output_dir/transcriptions.txt)")
    args = parser.parse_args()
 
    out_root    = Path(args.output_dir)
    audio_dir   = out_root / "audio"
    transcripts = out_root / "transcriptions.txt"
    out_root.mkdir(parents=True, exist_ok=True)
 
    print(f"Output directory : {out_root.resolve()}")
    print(f"  audio/               <- synthesised .wav files")
    print(f"  transcriptions.txt   <- ASR output")
 
    # Read input files
    ground_truth = read_lines(args.ground_truth)
    predicted    = read_lines(args.predicted)
    print(f"\nGround truth sentences : {len(ground_truth)}")
    print(f"Predicted   sentences  : {len(predicted)}")
    assert len(ground_truth) == len(predicted), \
        "ground_truth and predicted must have the same number of lines!"
 
    # Step 1 — TTS
    if not args.skip_tts:
        run_tts(predicted, audio_dir)
    else:
        print(f"\n[TTS] Skipping — reusing audio in {audio_dir}/")
 
    # Step 2 — ASR
    if not args.skip_asr:
        transcriptions = run_asr(audio_dir, transcripts)
    else:
        print(f"\n[ASR] Skipping — reading {transcripts}")
        transcriptions = read_lines(str(transcripts))
 
    # Step 3 — BLEU
    compute_bleu(transcriptions, ground_truth)
 
 
if __name__ == "__main__":
    main()