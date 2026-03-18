import json
import os

# Hardcoded paths
HINDI_TEXT_FILE    = "/mnt/storage/aditya/Evaluation/Benchmarking_dataset/Hindi_test_set.txt"
MAITHILI_TEXT_FILE = "/mnt/storage/aditya/Evaluation/Benchmarking_dataset/mai_gt_bl.txt"
AUDIO_DIR          = "/mnt/storage/aditya/results_from_tts_metric/parler_results/audio"
OUTPUT_JSON        = "/mnt/storage/aditya/results_from_tts_metric/test_translations_parler.json"

# Load text files
with open(HINDI_TEXT_FILE, "r", encoding="utf-8") as f:
    hindi_lines = [line.strip() for line in f.readlines()]

with open(MAITHILI_TEXT_FILE, "r", encoding="utf-8") as f:
    maithili_lines = [line.strip() for line in f.readlines()]

# Get sorted list of wav files
wav_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])

# Build JSON
translations = {}
for wav_file in wav_files:
    # Extract line number from filename: utt_00001.wav -> index 0
    line_number = int(wav_file.replace("utt_", "").replace(".wav", "")) - 1

    translations[wav_file] = {
        "full_audio_path": os.path.join(AUDIO_DIR, wav_file),
        "hi_text"        : hindi_lines[line_number],
        "mai_text"       : maithili_lines[line_number],
    }

# Save
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print(f"Saved {len(translations)} entries to {OUTPUT_JSON}")