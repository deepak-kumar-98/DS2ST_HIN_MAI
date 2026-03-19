#!/usr/bin/env python3
"""
Prepare Maithili data from IISc SYSPIN format for LoRA training.
Creates a small subset for quick training and evaluation.
"""

import json
import random
from pathlib import Path


def prepare_maithili_subset(
    json_path: str,
    wav_dir: str,
    output_dir: str,
    num_samples: int = 500,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Convert IISc Maithili data to Chatterbox manifest format with a small subset.

    Args:
        json_path: Path to IISc transcripts JSON
        wav_dir: Directory containing WAV files
        output_dir: Output directory for manifests
        num_samples: Number of samples to use (for quick training)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Load transcripts
    print(f"Loading transcripts from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transcripts = data['Transcripts']
    print(f"Found {len(transcripts)} total transcripts")

    # Convert to manifest format
    wav_dir = Path(wav_dir)
    samples = []
    missing_files = 0

    for file_id, info in transcripts.items():
        wav_path = wav_dir / f"{file_id}.wav"
        if wav_path.exists():
            samples.append({
                "audio_path": str(wav_path),
                "text": info['Transcript'].strip(),
                "language_id": "mai",
                "domain": info.get('Domain', 'UNKNOWN'),
                "id": file_id
            })
        else:
            missing_files += 1

    print(f"Valid samples: {len(samples)}, Missing files: {missing_files}")

    # Shuffle and take subset
    random.shuffle(samples)
    samples = samples[:num_samples]
    print(f"Using {len(samples)} samples for quick training")

    # Split into train/val/test
    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    print(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save manifests as JSON arrays (not JSONL)
    for name, data_samples in [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples),
        ('all', samples)
    ]:
        output_path = output_dir / f"{name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {output_path} ({len(data_samples)} samples)")

    # Print domain distribution
    domains = {}
    for s in samples:
        d = s['domain']
        domains[d] = domains.get(d, 0) + 1

    print("\nDomain distribution:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")

    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    # Paths
    IISC_DATA_DIR = "/Users/ntiwari/IITP/chatterbox/data/IISc_SYSPIN_Data/IISc_SYSPINProject_Maithili_Female_Spk001_HC"
    JSON_PATH = f"{IISC_DATA_DIR}/IISc_SYSPINProject_Maithili_Female_Spk001_HC_Transcripts.json"
    WAV_DIR = f"{IISC_DATA_DIR}/wav"
    OUTPUT_DIR = "/Users/ntiwari/IITP/attempt3/chatterbox/data/maithili"

    # Prepare small subset for quick training (500 samples)
    prepare_maithili_subset(
        json_path=JSON_PATH,
        wav_dir=WAV_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=500,  # Small subset for 30-min training
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
