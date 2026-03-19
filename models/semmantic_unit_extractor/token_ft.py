import os
import torch
import torchaudio
import numpy as np
import joblib
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor''
from fastdtw import fastdtw

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "utter-project/mHuBERT-147"
KMEANS_PATH = "<path to mhubert147_semantic_k500_fixed.joblib>"
LAYER = 9

# -----------------------------
# LOAD MODEL + KMEANS
# -----------------------------
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
).to(DEVICE)

model.eval()
kmeans = joblib.load(KMEANS_PATH)

print("Model + KMeans loaded")

# -----------------------------
# AUDIO → TOKENS
# -----------------------------
@torch.no_grad()
def audio_to_semantic_tokens(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    inputs = feature_extractor(
        wav,
        sampling_rate=16000,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)
    feats = outputs.hidden_states[LAYER][0].cpu().numpy()

    # Speaker-invariant normalization (MUST match training)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

    ids = kmeans.predict(feats)

    # collapse repeats
    collapsed = [int(ids[0])]
    for t in ids[1:]:
        if int(t) != collapsed[-1]:
            collapsed.append(int(t))

    return collapsed

# -----------------------------
# METRICS
# -----------------------------
def edit_similarity(a, b):
    """Normalized edit similarity (0–1)"""
    if not a or not b:
        return 0.0

    dist, _ = fastdtw(
        a, b,
        dist=lambda x, y: 0 if x == y else 1
    )
    return 1.0 - dist / max(len(a), len(b))


def dtw_distance(a, b):
    dist, _ = fastdtw(
        a, b,
        dist=lambda x, y: 0 if x == y else 1
    )
    return dist

# -----------------------------
# EXECUTION
# -----------------------------
if __name__ == "__main__":

    aud_paths = [
        "/test_out1_s1.wav",
        "/test_out1_s2.wav"
    ]

    tokens = []
    for p in aud_paths:
        tks = audio_to_semantic_tokens(p)
        tokens.append(tks)
        print(f"{os.path.basename(p)} → {len(tks)} tokens")

    if len(tokens) == 2:
        sim = edit_similarity(tokens[0], tokens[1])
        dtw = dtw_distance(tokens[0], tokens[1])

        print("\n" + "=" * 50)
        print(f"Semantic Similarity (Edit) : {sim:.4f}")
        print(f"DTW Distance              : {dtw}")
        print(f"Token Lengths             : {len(tokens[0])} vs {len(tokens[1])}")
        print("=" * 50)

        if sim > 0.35:
            print("✅ GOOD semantic consistency across speakers")
        elif sim > 0.2:
            print("⚠ Partial semantic alignment")
        else:
            print("❌ Weak semantic consistency")