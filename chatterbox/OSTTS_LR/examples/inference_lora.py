#!/usr/bin/env python3
"""
Example: Inference with LoRA-finetuned Chatterbox for Bhojpuri/Maithili
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths (update these)
    lora_path = "outputs/bhojpuri_lora/best_lora"  # Path to your trained LoRA
    reference_audio = "path/to/reference.wav"  # Reference audio for voice cloning
    output_path = "output.wav"

    # Text to synthesize (Bhojpuri example)
    text = "नमस्ते, हमार नाम चैटरबॉक्स बा।"
    language_id = "bho"  # "bho" for Bhojpuri, "mai" for Maithili

    print(f"Loading model with LoRA adapter...")

    # Method 1: Load pretrained + LoRA adapter
    model = ChatterboxMultilingualTTS.from_pretrained_with_lora(
        device=device,
        lora_path=lora_path
    )

    # Method 2: Alternative - load separately
    # model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    # model.load_lora_adapter(lora_path)

    print(f"Generating speech for: {text}")

    # Generate audio
    wav = model.generate(
        text=text,
        language_id=language_id,
        audio_prompt_path=reference_audio,
        exaggeration=0.5,  # Emotion intensity
        temperature=0.8,   # Sampling temperature
        cfg_weight=0.5,    # Classifier-free guidance
    )

    # Save output
    wav_numpy = wav.squeeze().cpu().numpy()
    sf.write(output_path, wav_numpy, model.sr)

    print(f"Audio saved to: {output_path}")
    print(f"Duration: {len(wav_numpy) / model.sr:.2f} seconds")


def batch_inference():
    """Example: Generate multiple samples."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = ChatterboxMultilingualTTS.from_pretrained_with_lora(
        device=device,
        lora_path="outputs/bhojpuri_lora/best_lora"
    )

    # Prepare voice
    model.prepare_conditionals("reference.wav", exaggeration=0.5)

    # Texts to synthesize
    texts = [
        ("राम राम भाई!", "greeting.wav"),
        ("आज मौसम बहुत सुहाना बा।", "weather.wav"),
        ("का हाल चाल बा?", "howru.wav"),
    ]

    for text, filename in texts:
        wav = model.generate(
            text=text,
            language_id="bho",
            temperature=0.8,
        )

        wav_numpy = wav.squeeze().cpu().numpy()
        sf.write(filename, wav_numpy, model.sr)
        print(f"Saved: {filename}")


def compare_base_vs_lora():
    """Example: Compare base model vs LoRA-finetuned."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = "नमस्ते, हमार नाम चैटरबॉक्स बा।"
    reference = "reference.wav"

    # Base model (Hindi)
    print("Generating with base model (Hindi)...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    wav_base = model.generate(
        text=text,
        language_id="hi",  # Using Hindi as closest
        audio_prompt_path=reference,
    )
    sf.write("output_base.wav", wav_base.squeeze().cpu().numpy(), model.sr)

    # With LoRA (Bhojpuri)
    print("Generating with LoRA (Bhojpuri)...")
    model.load_lora_adapter("outputs/bhojpuri_lora/best_lora")
    wav_lora = model.generate(
        text=text,
        language_id="bho",
        audio_prompt_path=reference,
    )
    sf.write("output_lora.wav", wav_lora.squeeze().cpu().numpy(), model.sr)

    print("Compare output_base.wav vs output_lora.wav")


if __name__ == "__main__":
    main()
    # batch_inference()
    # compare_base_vs_lora()
