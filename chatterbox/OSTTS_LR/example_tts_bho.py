import sys
from pathlib import Path
# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torchaudio as ta
import torch
import soundfile as sf
from chatterbox.tts import ChatterboxTTS
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
    # Convert tensor to numpy if needed
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    # Ensure it's 1D or 2D (mono or stereo)
    if wav.ndim > 1:
        wav = wav.squeeze()
    sf.write(filename, wav, sample_rate)

# model = ChatterboxTTS.from_pretrained(device=device)
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
'''

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
save_audio("test-1.wav", wav, model.sr)


text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav = multilingual_model.generate(text, language_id="fr")
save_audio("test-2.wav", wav, multilingual_model.sr)

'''
# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "Dhruv_1.wav"
text = "नमस्ते, रउआ का हाल बा? ई एगो बहुभाषी वॉच मॉडल बा, जे 23 गो भाषा के समर्थन करे ला।"
wav = multilingual_model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH, language_id="hi")
save_audio("test-1.wav", wav, multilingual_model.sr)

#text = "नमस्ते, रउआ का हाल बा? ई एगो बहुभाषी वॉच मॉडल बा, जे 23 गो भाषा के समर्थन करे ला।"
#AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
#wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
#wav = multilingual_model.generate(text, language_id="hi")
#save_audio("test-3-0.wav", wav, multilingual_model.sr)


#text = "नमस्कार, अहाँ के हाल की छै? ई एक बहुभाषी वॉच मॉडल छै, जे 23 भाषाक समर्थन करै छै।"
#wav = multilingual_model.generate(text, language_id="hi")
#save_audio("test-4-0.wav", wav, multilingual_model.sr)
