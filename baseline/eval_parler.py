"""
ai4bharat/indic-parler-tts  →  ASR  →  BLEU
=============================================
Steps:
  1. Read predicted text (one sentence per line)
  2. Synthesize audio with ai4bharat/indic-parler-tts
  3. Transcribe audio with ai4bharat/indic-conformer-600m-multilingual
  4. Compute BLEU score (tokenize="none") against ground truth

Usage:
  python eval_parler.py \
      --ground_truth /mnt/storage/aditya/Evaluation/Benchmarking_dataset/mai_gt_bl.txt \
      --predicted    /mnt/storage/aditya/Evaluation/bl/mai_output_from_mt.txt \
      --output_dir   /mnt/storage/aditya/results_from_tts_metric/parler_results
"""
import argparse
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from transformers import AutoModel
from sacrebleu.metrics import BLEU

# ─────────────────────────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TTS_MODEL   = "ai4bharat/indic-parler-tts"
ASR_MODEL   = "ai4bharat/indic-conformer-600m-multilingual"

PARLER_DESCRIPTION = (
    "A female speaker delivers a slightly expressive and animated speech "
    "with a moderate speed and pitch. The audio quality is very high."
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — TTS: ai4bharat/indic-parler-tts
# ─────────────────────────────────────────────────────────────────────────────

def run_tts(sentences: list[str], out_dir: Path, description: str):
    print(f"\n[TTS] Loading {TTS_MODEL} ...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL).to(DEVICE)
    model.eval()

    tokenizer             = AutoTokenizer.from_pretrained(TTS_MODEL)
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Check how many files are already done
    existing = {int(p.stem) for p in out_dir.glob("*.wav") if p.stat().st_size > 0}
    remaining = [(idx, text) for idx, text in enumerate(sentences) if idx not in existing]

    if existing:
        print(f"[TTS] Resuming — {len(existing)} files already done, "
              f"{len(remaining)} remaining.")
    else:
        print(f"[TTS] Starting fresh — {len(sentences)} sentences to synthesise.")

    if not remaining:
        print("[TTS] All files already exist, skipping synthesis.")
        return

    description_input_ids = description_tokenizer(description, return_tensors="pt").to(DEVICE)
    sr = model.config.sampling_rate

    print(f"[TTS] Description : \"{description}\"")

    for idx, text in tqdm(remaining, desc="Parler-TTS", total=len(remaining)):
        out_path = out_dir / f"{idx:05d}.wav"
        prompt_input_ids = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generation = model.generate(
                input_ids=description_input_ids.input_ids,
                attention_mask=description_input_ids.attention_mask,
                prompt_input_ids=prompt_input_ids.input_ids,
                prompt_attention_mask=prompt_input_ids.attention_mask,
            )
        # np.atleast_1d prevents 0-D array crash in sf.write
        # when generation has shape (1,) or (1, 1)
        waveform = np.atleast_1d(generation.cpu().numpy().squeeze())
        sf.write(out_path, waveform, sr)

    print(f"[TTS] Done. {len(sentences)} audio files in {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ASR: ai4bharat/indic-conformer-600m-multilingual
# ─────────────────────────────────────────────────────────────────────────────

def run_asr(audio_dir: Path, out_txt: Path, target_sr: int = 16_000) -> list[str]:
    print(f"\n[ASR] Loading {ASR_MODEL} ...")
    asr_model = AutoModel.from_pretrained(ASR_MODEL, trust_remote_code=True)
    asr_model.eval()

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}/")

    # Resume: load already-transcribed lines if the output file exists
    done: dict[int, str] = {}
    if out_txt.exists():
        existing_lines = read_lines(str(out_txt))
        # The file stores lines in order — map back to indices
        existing_wav = sorted(audio_dir.glob("*.wav"))
        for i, line in enumerate(existing_lines):
            if i < len(existing_wav):
                done[int(existing_wav[i].stem)] = line
        if done:
            print(f"[ASR] Resuming — {len(done)} transcriptions already done, "
                  f"{len(wav_files) - len(done)} remaining.")

    remaining_wavs = [p for p in wav_files if int(p.stem) not in done]

    print(f"[ASR] Transcribing {len(remaining_wavs)} files ...")
    for wav_path in tqdm(remaining_wavs, desc="ASR"):
        wav, sr = torchaudio.load(str(wav_path))
        wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = resampler(wav)
        with torch.no_grad():
            transcript = asr_model(wav, "mai", "ctc")
        done[int(wav_path.stem)] = transcript.strip()

    # Write all transcriptions in global order
    ordered = [done[int(p.stem)] for p in wav_files]
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(ordered) + "\n")

    print(f"[ASR] Transcriptions saved -> {out_txt}")
    return ordered


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
    print("  BLEU RESULT  —  ai4bharat/indic-parler-tts")
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
        description="ai4bharat/indic-parler-tts -> ASR -> BLEU pipeline"
    )
    parser.add_argument("--ground_truth", required=True,
                        help="Ground-truth Maithili text file (one sentence per line)")
    parser.add_argument("--predicted",    required=True,
                        help="Predicted Maithili text file (one sentence per line)")
    parser.add_argument("--output_dir",   default="parler_results",
                        help="Directory for audio files and transcriptions (default: ./parler_results)")
    parser.add_argument("--description",  default=PARLER_DESCRIPTION,
                        help="Voice style description for indic-parler-tts (must be in English)")
    parser.add_argument("--skip_tts", action="store_true",
                        help="Skip TTS synthesis entirely")
    parser.add_argument("--skip_asr", action="store_true",
                        help="Skip ASR entirely")
    args = parser.parse_args()

    out_root    = Path(args.output_dir)
    audio_dir   = out_root / "audio"
    transcripts = out_root / "transcriptions.txt"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output directory : {out_root.resolve()}")

    ground_truth = read_lines(args.ground_truth)
    predicted    = read_lines(args.predicted)
    print(f"Ground truth sentences : {len(ground_truth)}")
    print(f"Predicted   sentences  : {len(predicted)}")
    assert len(ground_truth) == len(predicted), \
        "ground_truth and predicted must have the same number of lines!"

    if not args.skip_tts:
        run_tts(predicted, audio_dir, args.description)
    else:
        print(f"\n[TTS] Skipping — reusing audio in {audio_dir}/")

    if not args.skip_asr:
        transcriptions = run_asr(audio_dir, transcripts)
    else:
        print(f"\n[ASR] Skipping — reading {transcripts}")
        transcriptions = read_lines(str(transcripts))

    compute_bleu(transcriptions, ground_truth)


if __name__ == "__main__":
    main()