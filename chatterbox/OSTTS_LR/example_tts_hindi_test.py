#!/usr/bin/env python3
"""
Example: Hindi TTS Test with Scientific/Technical Texts

This script generates audio for three specific Hindi text snippets
containing mathematical, scientific, and technical content.
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
    # Load multilingual model
    print("Loading Chatterbox Multilingual Model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("Model loaded successfully!")

    # Test texts - scientific/technical Hindi content
    texts = [
        # (a) Mathematical text with logarithm
        (
            "परंतु पूर्णांश यदि ऋणात्मक दो है और अपूर्णांश .4123(दशमलव चार एक दो  तीन) है तब log m = ऋणात्मक दो + दशमलव चार एक दो  तीन होगा।",
            "hindi_test_math.wav"
        ),
        # (b) Chemistry/Biology text with chemical formulas and ions
        (
            "प्लाज्मा में अनेक खनिज आयन जैसे सोडियम प्लस, कैल्सियम प्लस प्लस , मैग्नीशियम प्लस प्लस , हाइड्रोजन कार्बोनेट, क्लोराइड इत्यादि भी पाए जाते हैं।",
            "hindi_test_chemistry.wav"
        ),
        # (c) Agriculture/Food science text with percentages
        (
            "चीकू की क्रिकेट बॉल किस्म को  2 प्रतिशत ऑक्सीजन, 10 प्रतिशत कार्बन डाईऑक्साइड  और 88 प्रतिशत नाइट्रोजन  में रखने से फलों की पकने की प्रक्रिया धीमी हुई और इसके निधानी जीवन में लगभग 4-5 गुना ज्यादा वृद्धि पायी गयी।",
            "hindi_test_agriculture.wav"
        ),
        # (d) Text with bracket
        (
            "मेजबान पौधों (होस्ट) की निरंतरता और छोटे जीवन चक्र के कारण, इनका नियंत्रण मुश्किल है और संक्रमित फल उत्पादक क्षेत्रों में समस्या से निपटने के लिए क्षेत्रीय प्रोटोकॉल विकसित किए गए हैं।",
            "hindi_test_bracket.wav"
        ),
    ]

    # Optional: Reference audio for voice cloning
    AUDIO_PROMPT_PATH = "Neha_1.wav"  # Change to your reference audio file
    audio_prompt = None
    if Path(AUDIO_PROMPT_PATH).exists():
        audio_prompt = AUDIO_PROMPT_PATH
        print(f"Using reference audio: {AUDIO_PROMPT_PATH}")
    else:
        print(f"No reference audio found at {AUDIO_PROMPT_PATH}, using default voice")

    # Create output directory
    output_dir = Path("outputs/hindi_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate speech for each text
    print("\nGenerating Hindi speech for test texts...")
    print("=" * 60)

    for i, (text, output_file) in enumerate(texts, 1):
        print(f"\n[{i}/3] Text: {text[:50]}...")
        print(f"       Full: {text}")

        # Generate audio
        wav = model.generate(
            text=text,
            language_id="hi",       # Hindi language code
            audio_prompt_path=audio_prompt,
            exaggeration=0.5,       # Emotion intensity (0.0 to 1.0)
            temperature=0.8,        # Sampling temperature (higher = more varied)
            cfg_weight=0.5,         # Classifier-free guidance (0.0 to 1.0)
        )

        # Save output
        output_path = output_dir / output_file
        save_audio(str(output_path), wav, model.sr)

    print("\n" + "=" * 60)
    print("Done! Generated audio files in outputs/hindi_test/:")
    for _, filename in texts:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
