#!/usr/bin/env python3
"""
Example: Bhojpuri TTS for News Article

This script generates Bhojpuri audio for a news article about
Karnataka's new IT City plan, with two statements per audio file.
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
    AUDIO_PROMPT_PATH = "Neha_1.wav"  # Reference audio for voice cloning (optional)

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
        model = ChatterboxMultilingualTTS.from_pretrained_with_lora(
            device=device,
            lora_path=LORA_PATH
        )
        print("Model loaded with LoRA adapter")
    else:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        print("Base model loaded (no LoRA)")

    # Article segments - two statements per audio
    article_segments = [
        # Segment 1: Title and Introduction
        (
            "कर्नाटक सरकार के नया प्लान: बेंगलुरु से बहरें बन सकत बा नया आईटी सिटी। "
            "कर्नाटक के उप-मुख्यमंत्री डीके शिवकुमार घोषणा कइले बाड़ें कि सरकार बिदादी में एगो नया आईटी सिटी बनावे के योजना बना रहल बा।",
            "article_01_intro.wav"
        ),
        # Segment 2: Investment Interest
        (
            "उनकर कहनाम बा कि दुनिया भर के कई गो बड़हन नेता लोग इहाँ अरबों डॉलर के निवेश करे के चाहत बा। "
            "एही से सरकार ई कदम उठावे जा रहल बा।",
            "article_02_investment.wav"
        ),
        # Segment 3: New IT City Details
        (
            "डीके शिवकुमार बतवलें कि बेंगलुरु टेक समिट 2025 में 60 गो देसन के प्रतिनिधि लोग आ रहल बा। "
            "एतने भारी निवेश के संभाले खातिर बिदादी में नया जगह बनावल जाई।",
            "article_03_tech_summit.wav"
        ),
        # Segment 4: Beyond Bengaluru Policy
        (
            "सरकार के नया आईटी पॉलिसी 2025-2030 के तहत, अगर कवनो स्टार्टअप बेंगलुरु के छोड़ के टीयर-2 शहरन में अपन ऑफिस खोली, त ओकरा के भारी छूट मिल सकत बा। "
            "जइसे कि मैसूर, मैंगलोर, हुबली, बेलगावी, कालाबुर्गी, आदि शहरन में।",
            "article_04_tier2_policy.wav"
        ),
        # Segment 5: Rent and Property Tax Incentives
        (
            "किराया पर 50% वापस मिल जाई, अधिकतम 2 करोड़ रउपा तक। "
            "प्रॉपर्टी टैक्स पर 3 साल खातिर 30% के छूट मिली।",
            "article_05_rent_tax.wav"
        ),
        # Segment 6: Electricity and Internet Incentives
        (
            "बिजली ड्यूटी 5 साल खातिर पूरा माफ़ रही। "
            "इंटरनेट आ फोन बिल पर 25% के छूट मिली, 12 लाख रउपा तक।",
            "article_06_electricity_internet.wav"
        ),
        # Segment 7: R&D and Budget
        (
            "रिसर्च के खर्चा पर 40% वापस मिली, 50 करोड़ रउपा तक। "
            "सरकार एह योजना खातिर 5 साल में करीब 960 करोड़ रउपा खर्च करी।",
            "article_07_research_budget.wav"
        ),
        # Segment 8: Application and Mekedatu
        (
            "एकर आवेदन प्रक्रिया दिसंबर के शुरुआत में शुरू होखे के उम्मीद बा। "
            "शिवकुमार सुप्रीम कोर्ट के फैसला के स्वागत कइलें, जवना में तमिलनाडु के अर्जी खारिज कर दिहल गइल रहे।",
            "article_08_application_mekedatu.wav"
        ),
        # Segment 9: Conclusion
        (
            "उहाँ के कहनी कि ई कर्नाटक के जीत ह। "
            "संक्षेप में, सरकार चाहत बा कि अब आईटी कंपनी सब खाली बेंगलुरु में भीड़ ना लगावे, बल्कि राज्य के बाकी शहरन में भी जास।",
            "article_09_conclusion.wav"
        ),
    ]

    # Check if reference audio exists
    audio_prompt = None
    if Path(AUDIO_PROMPT_PATH).exists():
        audio_prompt = AUDIO_PROMPT_PATH
        print(f"Using reference audio: {AUDIO_PROMPT_PATH}")
    else:
        print(f"No reference audio found at {AUDIO_PROMPT_PATH}, using default voice")

    # Generate speech for each segment
    print("\nGenerating Bhojpuri speech for article segments...")
    print("=" * 60)

    for i, (text, output_file) in enumerate(article_segments, 1):
        print(f"\nSegment {i}/{len(article_segments)}")
        print(f"Text: {text[:80]}...")

        # Generate audio
        wav = model.generate(
            text=text,
            language_id="bho",
            audio_prompt_path=audio_prompt,
            exaggeration=0.5,
            temperature=0.8,
            cfg_weight=0.5,
        )

        # Save output
        save_audio(output_file, wav, model.sr)

    print("\n" + "=" * 60)
    print("Done! Generated audio files:")
    for _, filename in article_segments:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
