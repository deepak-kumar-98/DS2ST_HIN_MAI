#!/usr/bin/env python3
"""
Example: Bengali TTS with LoRA-finetuned Chatterbox

This script demonstrates how to use the LoRA-finetuned Chatterbox model
for Bengali text-to-speech synthesis.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


def save_audio(filename, wav, sample_rate):
    """Save audio using soundfile (more reliable than torchaudio.save on Mac)"""
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    if wav.ndim > 1:
        wav = wav.squeeze()
    sf.write(filename, wav, sample_rate)
    print(f"Saved: {filename} ({len(wav) / sample_rate:.2f}s)")


def main():
    # Configuration
    LORA_PATH = "outputs/bengali_lora/final_lora"  # Path to trained LoRA adapter
    AUDIO_PROMPT_PATH = "Dhruv_1.wav"  # Reference audio for voice cloning (optional)

    # Check if LoRA adapter exists
    lora_path = Path(LORA_PATH)
    if not lora_path.exists():
        print(f"Warning: LoRA adapter not found at {LORA_PATH}")
        print("Using base model without LoRA. Train the model first using:")
        print("  python scripts/train_lora.py --help")
        use_lora = False
    else:
        use_lora = True
        print(f"Loading LoRA adapter from: {LORA_PATH}")

    # Load model
    if use_lora:
        # Method 1: Load with LoRA adapter
        model = ChatterboxMultilingualTTS.from_pretrained_with_lora(
            device=device,
            lora_path=LORA_PATH
        )
        print("Model loaded with LoRA adapter")
    else:
        # Fallback: Load base model
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        print("Base model loaded (no LoRA)")

    # Bengali text samples
    texts = [
        # Greeting
        ("নমস্কার, আমার নাম চ্যাটারবক্স।", "bengali_greeting.wav"),
        # Weather
        ("আজ আবহাওয়া খুব ভালো।", "bengali_weather.wav"),
        # Question
        ("আপনি কেমন আছেন?", "bengali_howru.wav"),
        # Information
        ("এটি একটি বহুভাষিক টেক্সট-টু-স্পিচ মডেল।", "bengali_about.wav"),
        # Longer sentence
        ("বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ, যার রাজধানী ঢাকা।", "bengali_bangladesh.wav"),
        # Technology
        ("এই মডেল ২৩টি ভাষা সমর্থন করে।", "bengali_tech.wav"),
    ]

    # Check if reference audio exists
    audio_prompt = None
    if Path(AUDIO_PROMPT_PATH).exists():
        audio_prompt = AUDIO_PROMPT_PATH
        print(f"Using reference audio: {AUDIO_PROMPT_PATH}")
    else:
        print(f"No reference audio found at {AUDIO_PROMPT_PATH}, using default voice")

    # Generate speech for each text
    print("\nGenerating Bengali speech...")
    print("=" * 50)

    for text, output_file in texts:
        print(f"\nText: {text}")

        # Generate audio
        wav = model.generate(
            text=text,
            language_id="bn",
            audio_prompt_path=audio_prompt,
            exaggeration=0.5,   # Emotion intensity
            temperature=0.8,    # Sampling temperature
            cfg_weight=0.5,     # Classifier-free guidance
        )

        # Save output
        save_audio(output_file, wav, model.sr)

    print("\n" + "=" * 50)
    print("Done! Generated audio files:")
    for _, filename in texts:
        print(f"  - {filename}")


def compare_with_hindi():
    """
    Compare Bengali LoRA output with base Hindi model.
    Useful for evaluating the effect of LoRA fine-tuning.
    """
    LORA_PATH = "outputs/bengali_lora/best_lora"
    AUDIO_PROMPT_PATH = "Neha_1.wav"

    # Same text in Bengali
    text = "নমস্কার, আপনি কেমন আছেন? এটি একটি বহুভাষিক মডেল।"

    audio_prompt = AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None

    # Load base model first
    print("Loading base model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Generate with Hindi (closest language in base model for Indic languages)
    print("\nGenerating with base model (Hindi token)...")
    wav_hindi = model.generate(
        text=text,
        language_id="hi",
        audio_prompt_path=audio_prompt,
    )
    save_audio("compare_base_hindi_bn.wav", wav_hindi, model.sr)

    # Load LoRA adapter if available
    lora_path = Path(LORA_PATH)
    if lora_path.exists():
        print("\nLoading LoRA adapter...")
        model.load_lora_adapter(LORA_PATH)

        # Generate with Bengali LoRA
        print("Generating with LoRA (Bengali)...")
        wav_bn = model.generate(
            text=text,
            language_id="bn",
            audio_prompt_path=audio_prompt,
        )
        save_audio("compare_lora_bengali.wav", wav_bn, model.sr)

        print("\nComparison files generated:")
        print("  - compare_base_hindi_bn.wav (base model with Hindi)")
        print("  - compare_lora_bengali.wav (LoRA model with Bengali)")
    else:
        print(f"\nLoRA adapter not found at {LORA_PATH}")
        print("Only base model output generated.")


def example_with_different_emotions():
    """
    Example showing different emotion/exaggeration levels.
    """
    print("Loading model for emotion examples...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    text = "এটি একটি পরীক্ষা।"

    # Different exaggeration levels
    exaggerations = [
        (0.0, "bengali_neutral.wav"),      # Neutral
        (0.5, "bengali_moderate.wav"),     # Moderate emotion
        (1.0, "bengali_expressive.wav"),   # Very expressive
    ]

    print("\nGenerating speech with different emotion levels...")
    for exag, filename in exaggerations:
        print(f"Exaggeration: {exag} -> {filename}")
        wav = model.generate(
            text=text,
            language_id="bn",
            exaggeration=exag,
            temperature=0.8,
            cfg_weight=0.5,
        )
        save_audio(filename, wav, model.sr)

    print("Done! Compare the different emotion levels.")


if __name__ == "__main__":
    # Run main example
    main()

    # Uncomment to try other examples:
    # compare_with_hindi()
    # example_with_different_emotions()
