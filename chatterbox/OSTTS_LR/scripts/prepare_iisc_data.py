#!/usr/bin/env python3
"""
Prepare IISc SYSPIN Bhojpuri data for Chatterbox LoRA training.
"""

import json
import os
import sys
from pathlib import Path
import random

def convert_iisc_to_manifest(
    json_path: str,
    wav_dir: str,
    output_dir: str,
    language_id: str = "bho",
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42
):
    """
    Convert IISc SYSPIN data format to Chatterbox manifest format.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = Path(wav_dir)

    # Load transcripts
    print(f"Loading transcripts from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transcripts = data.get('Transcripts', {})
    print(f"Found {len(transcripts)} transcripts")

    # Create manifest entries
    samples = []
    missing_audio = 0

    for file_id, info in transcripts.items():
        audio_path = wav_dir / f"{file_id}.wav"

        if not audio_path.exists():
            missing_audio += 1
            continue

        sample = {
            'audio_path': str(audio_path),
            'text': info['Transcript'],
            'language_id': language_id,
            'domain': info.get('Domain', 'GENERAL'),
            'id': file_id
        }
        samples.append(sample)

    print(f"Created {len(samples)} samples ({missing_audio} missing audio files)")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    # Save manifests
    splits = [
        ('train.json', train_samples),
        ('val.json', val_samples),
        ('test.json', test_samples),
        ('all.json', samples)
    ]

    for filename, data in splits:
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} samples to {filepath}")

    # Print statistics
    print("\n" + "=" * 50)
    print("DATA STATISTICS")
    print("=" * 50)
    print(f"Total samples: {len(samples)}")
    print(f"Train: {len(train_samples)}")
    print(f"Val: {len(val_samples)}")
    print(f"Test: {len(test_samples)}")

    # Domain distribution
    domains = {}
    for s in samples:
        d = s.get('domain', 'UNKNOWN')
        domains[d] = domains.get(d, 0) + 1

    print("\nDomain distribution:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")

    return output_dir / 'train.json', output_dir / 'val.json', output_dir / 'test.json'


if __name__ == '__main__':
    # IISc SYSPIN Bhojpuri data paths
    json_path = "/Users/ntiwari/IITP/chatterbox/data/IISc_SYSPIN_Data/IISc_SYSPINProject_Bhojpuri_Male_Spk001_HC/IISc_SYSPINProject_Bhojpuri_Male_Spk001_HC_Transcripts.json"
    wav_dir = "/Users/ntiwari/IITP/chatterbox/data/IISc_SYSPIN_Data/IISc_SYSPINProject_Bhojpuri_Male_Spk001_HC/wav"
    output_dir = "/Users/ntiwari/IITP/attempt3/chatterbox/data/bhojpuri"

    convert_iisc_to_manifest(
        json_path=json_path,
        wav_dir=wav_dir,
        output_dir=output_dir,
        language_id="bho",
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05
    )
