#!/usr/bin/env python3
"""
Example: Bhojpuri TTS with LoRA-finetuned Chatterbox

This script demonstrates how to use the LoRA-finetuned Chatterbox model
for Bhojpuri text-to-speech synthesis.
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
    LORA_PATH = "outputs/bhojpuri_lora/best_lora"  # Path to trained LoRA adapter
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

    # Bhojpuri text samples
    texts = [
        # Greeting
        ("नमस्ते, हमार नाम चैटरबॉक्स बा।", "bhojpuri_greeting.wav"),
        # Weather
        ("आज मौसम बहुत सुहाना बा।", "bhojpuri_weather.wav"),
        # Question
        ("रउआ का हाल चाल बा?", "bhojpuri_howru.wav"),
        # Longer sentence
        ("ई एगो बहुभाषी टेक्स्ट-टू-स्पीच मॉडल बा, जे 23 गो भाषा के समर्थन करे ला।", "bhojpuri_about.wav"),
    ]

    # Check if reference audio exists
    audio_prompt = None
    if Path(AUDIO_PROMPT_PATH).exists():
        audio_prompt = AUDIO_PROMPT_PATH
        print(f"Using reference audio: {AUDIO_PROMPT_PATH}")
    else:
        print(f"No reference audio found at {AUDIO_PROMPT_PATH}, using default voice")

    # Generate speech for each text
    print("\nGenerating Bhojpuri speech...")
    print("=" * 50)

    for text, output_file in texts:
        print(f"\nText: {text}")

        # Generate audio
        # Note: Using "hi" (Hindi) as language_id since Bhojpuri shares the same
        # Devanagari script and phonetics. The [bho] token is not in the base vocabulary.
        wav = model.generate(
            text=text,
            language_id="bho",  # Using Hindi - closest language in vocab
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
    Compare Bhojpuri LoRA output with base Hindi model.
    Useful for evaluating the effect of LoRA fine-tuning.
    """
    LORA_PATH = "outputs/bhojpuri_lora/best_lora"
    AUDIO_PROMPT_PATH = "Dhruv_1.wav"

    # Same text in Bhojpuri
    text = "नमस्ते, रउआ का हाल बा? ई एगो बहुभाषी मॉडल बा।"

    audio_prompt = AUDIO_PROMPT_PATH if Path(AUDIO_PROMPT_PATH).exists() else None

    # Load base model first
    print("Loading base model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Generate with Hindi (closest language in base model)
    print("\nGenerating with base model (Hindi)...")
    wav_hindi = model.generate(
        text=text,
        language_id="hi",
        audio_prompt_path=audio_prompt,
    )
    save_audio("compare_base_hindi.wav", wav_hindi, model.sr)

    # Load LoRA adapter if available
    lora_path = Path(LORA_PATH)
    if lora_path.exists():
        print("\nLoading LoRA adapter...")
        model.load_lora_adapter(LORA_PATH)

        # Generate with Bhojpuri LoRA
        # Using "hi" since [bho] is not in the base vocabulary
        print("Generating with LoRA (Bhojpuri)...")
        wav_bho = model.generate(
            text=text,
            language_id="bho",
            audio_prompt_path=audio_prompt,
        )
        save_audio("compare_lora_bhojpuri.wav", wav_bho, model.sr)

        print("\nComparison files generated:")
        print("  - compare_base_hindi.wav (base model with Hindi)")
        print("  - compare_lora_bhojpuri.wav (LoRA model with Bhojpuri)")
    else:
        print(f"\nLoRA adapter not found at {LORA_PATH}")
        print("Only base model output generated.")


if __name__ == "__main__":
    main()
    # Uncomment to run comparison:
    # compare_with_hindi()
