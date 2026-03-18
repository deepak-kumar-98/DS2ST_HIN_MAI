import os
import glob
import joblib
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.cluster import MiniBatchKMeans

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "utter-project/mHuBERT-147"

LAYER = 9
K = 500
MAX_TOTAL_FRAMES = 4_000_000
MAX_FRAMES_PER_UTT = 1500

print("Using device:", DEVICE)

# --------------------------------------------------
# LOAD mHuBERT (CORRECT WAY)
# --------------------------------------------------
print("Loading mHuBERT-147...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
).to(DEVICE)

model.eval()

print("Model loaded.")
print("Hidden size:", model.config.hidden_size)

# --------------------------------------------------
# AUDIO LOADING
# --------------------------------------------------
def load_audio(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------
@torch.no_grad()
def extract_normalized_feats(wav):
    inputs = feature_extractor(
        wav,
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)

    feats = outputs.hidden_states[LAYER][0]   # (T, D)
    feats = feats.cpu().numpy()

    # Speaker-invariant normalization
    mean = feats.mean(axis=0)
    std = feats.std(axis=0) + 1e-6
    feats = (feats - mean) / std

    return feats

# --------------------------------------------------
# DATA COLLECTION
# --------------------------------------------------
def collect_wavs(root):
    paths = []
    for sub in os.listdir(root):
        subdir = os.path.join(root, sub)
        if os.path.isdir(subdir):
            paths.extend(glob.glob(os.path.join(subdir, "*.wav")))
    return paths

paths = []
paths.extend(collect_wavs("/hindi_audio_data"))
paths.extend(collect_wavs("/maithili_audio_data"))

np.random.shuffle(paths)
print("Total audio files:", len(paths))

# --------------------------------------------------
# KMEANS TRAINING
# --------------------------------------------------
kmeans = MiniBatchKMeans(
    n_clusters=K,
    batch_size=20000,
    random_state=0
)

total_frames = 0
print("Training mHuBERT semantic KMeans...")

for wav_path in tqdm(paths):
    try:
        wav = load_audio(wav_path)
        feats = extract_normalized_feats(wav)

        if feats.shape[0] > MAX_FRAMES_PER_UTT:
            idx = np.random.choice(
                feats.shape[0],
                MAX_FRAMES_PER_UTT,
                replace=False
            )
            feats = feats[idx]

        kmeans.partial_fit(feats)
        total_frames += feats.shape[0]

        if total_frames >= MAX_TOTAL_FRAMES:
            break

    except Exception:
        continue

# --------------------------------------------------
# SAVE
# --------------------------------------------------
joblib.dump(kmeans, "mhubert147_semantic_k500.joblib")
print("Saved KMeans:", kmeans.cluster_centers_.shape)
