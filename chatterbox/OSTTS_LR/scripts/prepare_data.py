#!/usr/bin/env python3
"""
Data preparation script for Chatterbox TTS fine-tuning.
Prepares audio and text data for Bhojpuri and Maithili languages.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import librosa
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.s3gen import S3GEN_SR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """Represents a single audio sample with metadata."""
    audio_path: str
    text: str
    language_id: str
    duration: float
    sample_rate: int
    speaker_id: Optional[str] = None


@dataclass
class ProcessedSample:
    """Represents a processed sample ready for training."""
    audio_path: str
    text: str
    language_id: str
    duration: float
    speech_tokens_path: str
    num_speech_tokens: int
    speaker_id: Optional[str] = None


def validate_audio_file(audio_path: str, min_duration: float = 0.5, max_duration: float = 30.0) -> Tuple[bool, str, float]:
    """
    Validate an audio file for training.

    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds

    Returns:
        (is_valid, message, duration)
    """
    try:
        # Load audio to check properties
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr

        if duration < min_duration:
            return False, f"Too short: {duration:.2f}s < {min_duration}s", duration

        if duration > max_duration:
            return False, f"Too long: {duration:.2f}s > {max_duration}s", duration

        # Check for silence
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.001:
            return False, "Audio appears to be silence", duration

        return True, "Valid", duration

    except Exception as e:
        return False, f"Error loading: {str(e)}", 0.0


def load_manifest(manifest_path: str) -> List[Dict]:
    """
    Load a data manifest file.
    Supports JSON and JSONL formats.

    Expected format:
    {"audio_path": "path/to/audio.wav", "text": "transcription", "language_id": "bho"}
    """
    manifest_path = Path(manifest_path)
    samples = []

    if manifest_path.suffix == '.jsonl':
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:  # .json
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            else:
                samples = data.get('samples', data.get('data', []))

    return samples


def create_manifest_from_directory(
    audio_dir: str,
    text_file: str,
    language_id: str,
    output_path: str
) -> str:
    """
    Create a manifest from a directory of audio files and a text file.

    Args:
        audio_dir: Directory containing audio files
        text_file: Text file with format: filename|transcription
        language_id: Language code (e.g., "bho", "mai")
        output_path: Path to save the manifest

    Returns:
        Path to created manifest
    """
    audio_dir = Path(audio_dir)
    samples = []

    # Read text file
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue

            parts = line.split('|')
            if len(parts) >= 2:
                filename = parts[0].strip()
                text = parts[1].strip()

                # Find matching audio file
                for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                    audio_path = audio_dir / f"{filename}{ext}"
                    if audio_path.exists():
                        samples.append({
                            'audio_path': str(audio_path),
                            'text': text,
                            'language_id': language_id
                        })
                        break

    # Save manifest
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info(f"Created manifest with {len(samples)} samples at {output_path}")
    return output_path


def process_single_audio(
    audio_path: str,
    output_dir: str,
    target_sr: int = S3_SR
) -> Tuple[str, int]:
    """
    Process a single audio file: resample and save.

    Args:
        audio_path: Path to input audio
        output_dir: Directory to save processed audio
        target_sr: Target sample rate

    Returns:
        (processed_audio_path, num_samples)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and resample
    audio, sr = librosa.load(audio_path, sr=target_sr)

    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # Save processed audio
    filename = Path(audio_path).stem + '.npy'
    output_path = output_dir / filename
    np.save(output_path, audio.astype(np.float32))

    return str(output_path), len(audio)


def tokenize_audio_batch(
    audio_paths: List[str],
    tokenizer,
    device: str = 'cuda'
) -> List[Tuple[str, torch.Tensor]]:
    """
    Tokenize a batch of audio files using S3 tokenizer.

    Args:
        audio_paths: List of audio file paths
        tokenizer: S3Tokenizer instance
        device: Device for processing

    Returns:
        List of (path, tokens) tuples
    """
    results = []

    for audio_path in audio_paths:
        try:
            # Load audio
            if audio_path.endswith('.npy'):
                audio = np.load(audio_path)
            else:
                audio, _ = librosa.load(audio_path, sr=S3_SR)

            # Tokenize
            tokens, _ = tokenizer.forward([audio], max_len=4096)
            results.append((audio_path, tokens.squeeze(0)))

        except Exception as e:
            logger.error(f"Error tokenizing {audio_path}: {e}")
            results.append((audio_path, None))

    return results


def prepare_dataset(
    manifest_path: str,
    output_dir: str,
    language_id: str,
    tokenizer=None,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    num_workers: int = 4,
    device: str = 'cuda'
) -> str:
    """
    Prepare a complete dataset for training.

    Args:
        manifest_path: Path to input manifest
        output_dir: Directory for processed data
        language_id: Language code
        tokenizer: S3Tokenizer instance (optional, will create if None)
        min_duration: Minimum audio duration
        max_duration: Maximum audio duration
        num_workers: Number of parallel workers
        device: Device for tokenization

    Returns:
        Path to processed manifest
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    audio_dir = output_dir / 'audio'
    tokens_dir = output_dir / 'tokens'
    audio_dir.mkdir(exist_ok=True)
    tokens_dir.mkdir(exist_ok=True)

    # Load manifest
    samples = load_manifest(manifest_path)
    logger.info(f"Loaded {len(samples)} samples from manifest")

    # Validate and filter samples
    valid_samples = []
    for sample in tqdm(samples, desc="Validating audio"):
        is_valid, msg, duration = validate_audio_file(
            sample['audio_path'],
            min_duration,
            max_duration
        )
        if is_valid:
            sample['duration'] = duration
            valid_samples.append(sample)
        else:
            logger.debug(f"Skipping {sample['audio_path']}: {msg}")

    logger.info(f"Valid samples: {len(valid_samples)} / {len(samples)}")

    # Process audio files
    processed_samples = []

    logger.info("Processing audio files...")
    for sample in tqdm(valid_samples, desc="Processing audio"):
        try:
            processed_path, num_samples = process_single_audio(
                sample['audio_path'],
                audio_dir
            )
            sample['processed_audio_path'] = processed_path
            processed_samples.append(sample)
        except Exception as e:
            logger.error(f"Error processing {sample['audio_path']}: {e}")

    # Tokenize audio if tokenizer provided
    if tokenizer is not None:
        logger.info("Tokenizing audio files...")

        for sample in tqdm(processed_samples, desc="Tokenizing"):
            try:
                audio_path = sample['processed_audio_path']

                # Load audio
                if audio_path.endswith('.npy'):
                    audio = np.load(audio_path)
                else:
                    audio, _ = librosa.load(audio_path, sr=S3_SR)

                # Tokenize
                with torch.no_grad():
                    tokens, _ = tokenizer.forward([audio], max_len=4096)
                    tokens = tokens.squeeze(0).cpu()

                # Save tokens
                tokens_filename = Path(audio_path).stem + '_tokens.pt'
                tokens_path = tokens_dir / tokens_filename
                torch.save(tokens, tokens_path)

                sample['speech_tokens_path'] = str(tokens_path)
                sample['num_speech_tokens'] = len(tokens)

            except Exception as e:
                logger.error(f"Error tokenizing {sample['audio_path']}: {e}")
                sample['speech_tokens_path'] = None
                sample['num_speech_tokens'] = 0

    # Create output manifest
    output_manifest = []
    for sample in processed_samples:
        output_sample = {
            'audio_path': sample.get('processed_audio_path', sample['audio_path']),
            'text': sample['text'],
            'language_id': language_id,
            'duration': sample['duration'],
            'speaker_id': sample.get('speaker_id'),
        }
        if 'speech_tokens_path' in sample:
            output_sample['speech_tokens_path'] = sample['speech_tokens_path']
            output_sample['num_speech_tokens'] = sample['num_speech_tokens']

        output_manifest.append(output_sample)

    # Save processed manifest
    output_manifest_path = output_dir / 'manifest.json'
    with open(output_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(output_manifest, f, ensure_ascii=False, indent=2)

    # Print statistics
    total_duration = sum(s['duration'] for s in output_manifest)
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total samples: {len(output_manifest)}")
    logger.info(f"  Total duration: {total_duration / 3600:.2f} hours")
    logger.info(f"  Average duration: {total_duration / len(output_manifest):.2f} seconds")
    logger.info(f"  Output manifest: {output_manifest_path}")

    return str(output_manifest_path)


def split_dataset(
    manifest_path: str,
    output_dir: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[str, str, str]:
    """
    Split dataset into train/val/test sets.

    Args:
        manifest_path: Path to manifest file
        output_dir: Directory to save split manifests
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        (train_path, val_path, test_path)
    """
    import random

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    samples = load_manifest(manifest_path)

    # Shuffle
    random.seed(seed)
    random.shuffle(samples)

    # Calculate split indices
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    # Save splits
    train_path = output_dir / 'train.json'
    val_path = output_dir / 'val.json'
    test_path = output_dir / 'test.json'

    for path, data in [(train_path, train_samples), (val_path, val_samples), (test_path, test_samples)]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    return str(train_path), str(val_path), str(test_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Chatterbox TTS fine-tuning")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Create manifest command
    create_parser = subparsers.add_parser('create-manifest', help='Create manifest from directory')
    create_parser.add_argument('--audio-dir', required=True, help='Directory with audio files')
    create_parser.add_argument('--text-file', required=True, help='Text file (filename|transcription)')
    create_parser.add_argument('--language-id', required=True, help='Language code (bho, mai)')
    create_parser.add_argument('--output', required=True, help='Output manifest path')

    # Prepare dataset command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset for training')
    prepare_parser.add_argument('--manifest', required=True, help='Input manifest path')
    prepare_parser.add_argument('--output-dir', required=True, help='Output directory')
    prepare_parser.add_argument('--language-id', required=True, help='Language code')
    prepare_parser.add_argument('--min-duration', type=float, default=0.5)
    prepare_parser.add_argument('--max-duration', type=float, default=30.0)
    prepare_parser.add_argument('--num-workers', type=int, default=4)
    prepare_parser.add_argument('--tokenize', action='store_true', help='Tokenize audio')
    prepare_parser.add_argument('--device', default='cuda')

    # Split dataset command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--manifest', required=True, help='Input manifest')
    split_parser.add_argument('--output-dir', required=True, help='Output directory')
    split_parser.add_argument('--train-ratio', type=float, default=0.9)
    split_parser.add_argument('--val-ratio', type=float, default=0.05)
    split_parser.add_argument('--test-ratio', type=float, default=0.05)
    split_parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.command == 'create-manifest':
        create_manifest_from_directory(
            args.audio_dir,
            args.text_file,
            args.language_id,
            args.output
        )

    elif args.command == 'prepare':
        tokenizer = None
        if args.tokenize:
            from chatterbox.models.s3tokenizer import S3Tokenizer
            tokenizer = S3Tokenizer()
            tokenizer.to(args.device)

        prepare_dataset(
            args.manifest,
            args.output_dir,
            args.language_id,
            tokenizer=tokenizer,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            num_workers=args.num_workers,
            device=args.device
        )

    elif args.command == 'split':
        split_dataset(
            args.manifest,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
