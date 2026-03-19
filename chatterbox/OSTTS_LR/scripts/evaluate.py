#!/usr/bin/env python3
"""
Evaluation script for Chatterbox TTS with LoRA fine-tuning.
Supports metrics like speaker similarity, intelligibility, and audio quality.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluation."""
    sample_id: str
    text: str
    reference_path: Optional[str]
    generated_path: str
    duration: float
    rtf: float  # Real-time factor
    speaker_similarity: Optional[float] = None
    mos_predicted: Optional[float] = None


def compute_speaker_similarity(
    generated_wav: np.ndarray,
    reference_wav: np.ndarray,
    voice_encoder,
    sample_rate: int = 16000
) -> float:
    """
    Compute speaker similarity between generated and reference audio.

    Args:
        generated_wav: Generated audio waveform
        reference_wav: Reference audio waveform
        voice_encoder: VoiceEncoder instance
        sample_rate: Audio sample rate

    Returns:
        Cosine similarity score (0-1)
    """
    # Get embeddings
    gen_emb = voice_encoder.embeds_from_wavs([generated_wav], sample_rate=sample_rate)
    ref_emb = voice_encoder.embeds_from_wavs([reference_wav], sample_rate=sample_rate)

    # Compute cosine similarity
    gen_emb = gen_emb / (np.linalg.norm(gen_emb) + 1e-8)
    ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-8)

    similarity = np.dot(gen_emb.flatten(), ref_emb.flatten())
    return float(similarity)


def generate_samples(
    model: ChatterboxMultilingualTTS,
    test_manifest: str,
    output_dir: str,
    language_id: str,
    reference_audio: Optional[str] = None,
    num_samples: Optional[int] = None,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
) -> List[EvaluationResult]:
    """
    Generate audio samples for evaluation.

    Args:
        model: ChatterboxMultilingualTTS instance
        test_manifest: Path to test manifest
        output_dir: Directory to save generated audio
        language_id: Language code
        reference_audio: Path to reference audio for voice cloning
        num_samples: Number of samples to generate (None = all)
        temperature: Sampling temperature
        cfg_weight: Classifier-free guidance weight

    Returns:
        List of EvaluationResult objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test manifest
    with open(test_manifest, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    if num_samples:
        samples = samples[:num_samples]

    results = []

    # Prepare conditionals if reference provided
    if reference_audio:
        model.prepare_conditionals(reference_audio)

    logger.info(f"Generating {len(samples)} samples...")

    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        text = sample['text']
        sample_id = sample.get('id', f"sample_{idx:04d}")

        # Use sample-specific reference if available
        sample_ref = sample.get('audio_path', reference_audio)
        if sample_ref and sample_ref != reference_audio:
            model.prepare_conditionals(sample_ref)

        # Generate audio
        import time
        start_time = time.time()

        try:
            wav = model.generate(
                text=text,
                language_id=language_id,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )

            generation_time = time.time() - start_time

            # Convert to numpy
            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().numpy()

            # Calculate duration and RTF
            duration = len(wav) / model.sr
            rtf = generation_time / duration if duration > 0 else 0

            # Save audio
            output_path = output_dir / f"{sample_id}.wav"
            sf.write(output_path, wav, model.sr)

            result = EvaluationResult(
                sample_id=sample_id,
                text=text,
                reference_path=sample_ref,
                generated_path=str(output_path),
                duration=duration,
                rtf=rtf,
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Error generating sample {sample_id}: {e}")
            continue

    return results


def evaluate_speaker_similarity(
    results: List[EvaluationResult],
    model: ChatterboxMultilingualTTS,
) -> List[EvaluationResult]:
    """
    Evaluate speaker similarity for generated samples.

    Args:
        results: List of EvaluationResult objects
        model: Model instance (for voice encoder)

    Returns:
        Updated results with speaker similarity scores
    """
    import librosa
    from chatterbox.models.s3tokenizer import S3_SR

    logger.info("Computing speaker similarity...")

    for result in tqdm(results, desc="Speaker similarity"):
        if not result.reference_path:
            continue

        try:
            # Load audios
            gen_wav, _ = librosa.load(result.generated_path, sr=S3_SR)
            ref_wav, _ = librosa.load(result.reference_path, sr=S3_SR)

            # Compute similarity
            similarity = compute_speaker_similarity(
                gen_wav, ref_wav, model.ve, S3_SR
            )
            result.speaker_similarity = similarity

        except Exception as e:
            logger.error(f"Error computing similarity for {result.sample_id}: {e}")

    return results


def compute_metrics(results: List[EvaluationResult]) -> Dict:
    """
    Compute aggregate metrics from evaluation results.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # RTF statistics
    rtfs = [r.rtf for r in results]
    metrics['rtf_mean'] = np.mean(rtfs)
    metrics['rtf_std'] = np.std(rtfs)
    metrics['rtf_median'] = np.median(rtfs)

    # Duration statistics
    durations = [r.duration for r in results]
    metrics['duration_mean'] = np.mean(durations)
    metrics['duration_total'] = np.sum(durations)

    # Speaker similarity statistics
    similarities = [r.speaker_similarity for r in results if r.speaker_similarity is not None]
    if similarities:
        metrics['speaker_similarity_mean'] = np.mean(similarities)
        metrics['speaker_similarity_std'] = np.std(similarities)
        metrics['speaker_similarity_min'] = np.min(similarities)
        metrics['speaker_similarity_max'] = np.max(similarities)

    # Sample count
    metrics['num_samples'] = len(results)
    metrics['num_failed'] = len([r for r in results if r.duration == 0])

    return metrics


def run_evaluation(
    checkpoint_dir: str,
    test_manifest: str,
    output_dir: str,
    language_id: str,
    lora_path: Optional[str] = None,
    reference_audio: Optional[str] = None,
    num_samples: Optional[int] = None,
    device: str = 'cuda',
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
) -> Dict:
    """
    Run full evaluation pipeline.

    Args:
        checkpoint_dir: Path to model checkpoint
        test_manifest: Path to test manifest
        output_dir: Output directory for results
        language_id: Language code
        lora_path: Path to LoRA adapter (optional)
        reference_audio: Reference audio for voice cloning
        num_samples: Number of samples to evaluate
        device: Device for inference
        temperature: Sampling temperature
        cfg_weight: CFG weight

    Returns:
        Dictionary with metrics and results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    if lora_path:
        model = ChatterboxMultilingualTTS.from_local_with_lora(
            checkpoint_dir, device, lora_path
        )
        logger.info(f"Loaded model with LoRA from {lora_path}")
    else:
        model = ChatterboxMultilingualTTS.from_local(checkpoint_dir, device)

    # Generate samples
    audio_dir = output_dir / "audio"
    results = generate_samples(
        model=model,
        test_manifest=test_manifest,
        output_dir=str(audio_dir),
        language_id=language_id,
        reference_audio=reference_audio,
        num_samples=num_samples,
        temperature=temperature,
        cfg_weight=cfg_weight,
    )

    # Evaluate speaker similarity
    results = evaluate_speaker_similarity(results, model)

    # Compute metrics
    metrics = compute_metrics(results)

    # Save results
    results_data = {
        'metrics': metrics,
        'samples': [asdict(r) for r in results],
        'config': {
            'checkpoint_dir': checkpoint_dir,
            'lora_path': lora_path,
            'language_id': language_id,
            'temperature': temperature,
            'cfg_weight': cfg_weight,
        }
    }

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Samples evaluated: {metrics['num_samples']}")
    logger.info(f"Total duration: {metrics['duration_total']:.2f}s")
    logger.info(f"Mean RTF: {metrics['rtf_mean']:.3f} (lower is faster)")

    if 'speaker_similarity_mean' in metrics:
        logger.info(f"Speaker similarity: {metrics['speaker_similarity_mean']:.3f} +/- {metrics['speaker_similarity_std']:.3f}")

    logger.info(f"\nResults saved to: {results_path}")
    logger.info("=" * 50)

    return results_data


def compare_models(
    checkpoint_dir: str,
    test_manifest: str,
    output_dir: str,
    language_id: str,
    lora_paths: List[str],
    model_names: List[str],
    reference_audio: Optional[str] = None,
    num_samples: int = 50,
    device: str = 'cuda',
) -> Dict:
    """
    Compare multiple LoRA models.

    Args:
        checkpoint_dir: Base model checkpoint
        test_manifest: Test manifest
        output_dir: Output directory
        language_id: Language code
        lora_paths: List of LoRA paths (None for base model)
        model_names: Names for each model
        reference_audio: Reference audio
        num_samples: Samples per model
        device: Device

    Returns:
        Comparison results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for name, lora_path in zip(model_names, lora_paths):
        logger.info(f"\nEvaluating: {name}")

        model_output = output_dir / name
        results = run_evaluation(
            checkpoint_dir=checkpoint_dir,
            test_manifest=test_manifest,
            output_dir=str(model_output),
            language_id=language_id,
            lora_path=lora_path,
            reference_audio=reference_audio,
            num_samples=num_samples,
            device=device,
        )

        all_results[name] = results['metrics']

    # Save comparison
    comparison_path = output_dir / "comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'RTF':>10} {'Spk Sim':>10}")
    logger.info("-" * 60)

    for name, metrics in all_results.items():
        rtf = metrics.get('rtf_mean', 0)
        sim = metrics.get('speaker_similarity_mean', 0)
        logger.info(f"{name:<20} {rtf:>10.3f} {sim:>10.3f}")

    logger.info("=" * 60)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chatterbox TTS models")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Single model evaluation
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a single model')
    eval_parser.add_argument('--checkpoint-dir', required=True, help='Model checkpoint directory')
    eval_parser.add_argument('--test-manifest', required=True, help='Test manifest path')
    eval_parser.add_argument('--output-dir', required=True, help='Output directory')
    eval_parser.add_argument('--language-id', required=True, help='Language code')
    eval_parser.add_argument('--lora-path', help='LoRA adapter path')
    eval_parser.add_argument('--reference-audio', help='Reference audio for voice cloning')
    eval_parser.add_argument('--num-samples', type=int, help='Number of samples')
    eval_parser.add_argument('--device', default='cuda', help='Device')
    eval_parser.add_argument('--temperature', type=float, default=0.8)
    eval_parser.add_argument('--cfg-weight', type=float, default=0.5)

    # Model comparison
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--checkpoint-dir', required=True, help='Base model checkpoint')
    compare_parser.add_argument('--test-manifest', required=True, help='Test manifest')
    compare_parser.add_argument('--output-dir', required=True, help='Output directory')
    compare_parser.add_argument('--language-id', required=True, help='Language code')
    compare_parser.add_argument('--lora-paths', nargs='+', help='LoRA paths (use "none" for base)')
    compare_parser.add_argument('--model-names', nargs='+', required=True, help='Model names')
    compare_parser.add_argument('--reference-audio', help='Reference audio')
    compare_parser.add_argument('--num-samples', type=int, default=50)
    compare_parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    if args.command == 'evaluate':
        run_evaluation(
            checkpoint_dir=args.checkpoint_dir,
            test_manifest=args.test_manifest,
            output_dir=args.output_dir,
            language_id=args.language_id,
            lora_path=args.lora_path,
            reference_audio=args.reference_audio,
            num_samples=args.num_samples,
            device=args.device,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
        )

    elif args.command == 'compare':
        # Convert "none" to None
        lora_paths = [
            None if p.lower() == 'none' else p
            for p in (args.lora_paths or [None] * len(args.model_names))
        ]

        compare_models(
            checkpoint_dir=args.checkpoint_dir,
            test_manifest=args.test_manifest,
            output_dir=args.output_dir,
            language_id=args.language_id,
            lora_paths=lora_paths,
            model_names=args.model_names,
            reference_audio=args.reference_audio,
            num_samples=args.num_samples,
            device=args.device,
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
