# PEFT/LoRA Implementation for Chatterbox TTS - Bhojpuri & Maithili

## Overview

This document details the implementation of Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) for the Chatterbox TTS system to support Bhojpuri and Maithili languages.

## Table of Contents

1. [Project Summary](#project-summary)
2. [Codebase Analysis](#codebase-analysis)
3. [Implementation Phases](#implementation-phases)
4. [Files Created/Modified](#files-createdmodified)
5. [Data Preparation](#data-preparation)
6. [Training Results](#training-results)
7. [Usage Instructions](#usage-instructions)
8. [Future Recommendations](#future-recommendations)

---

## Project Summary

### Objective
Implement PEFT/LoRA fine-tuning capability for Chatterbox's Llama-based T3 model to adapt it for Bhojpuri and Maithili text-to-speech synthesis.

### Key Achievements
- Created complete LoRA training infrastructure
- Added text preprocessing for Bhojpuri and Maithili (Devanagari script)
- Processed IISc SYSPIN Bhojpuri dataset (26,012 samples, ~42 hours)
- Demonstrated successful training with 47% loss reduction
- Implemented inference and evaluation pipelines

---

## Codebase Analysis

### Chatterbox Architecture

```
Text Input
    ↓
[Text Tokenization] → Text tokens (vocab: 2454 for multilingual)
    ↓
[T3 Model - Llama 520M backbone]
    ├── 30 transformer layers
    ├── 16 attention heads
    └── 1024 hidden dimension
    ↓
[Speech Tokens] (vocab: 6561)
    ↓
[S3Gen - Vocoder]
    ↓
Audio Output (24kHz)
```

### Key Components

| Component | Location | Parameters |
|-----------|----------|------------|
| T3 Model (Llama) | `src/chatterbox/models/t3/t3.py` | ~520M |
| Text Tokenizer | `src/chatterbox/models/tokenizers/tokenizer.py` | - |
| Voice Encoder | `src/chatterbox/models/voice_encoder/` | - |
| S3Gen Vocoder | `src/chatterbox/models/s3gen/` | - |

---

## Implementation Phases

### Phase 1: Data Preparation

#### Text Preprocessing for Bhojpuri/Maithili

**File**: `src/chatterbox/models/tokenizers/tokenizer.py`

Added two normalization functions:

```python
def bhojpuri_normalize(text: str) -> str:
    """
    Bhojpuri text normalization for TTS.
    - NFC Unicode normalization for Devanagari
    - Character mappings (Candra E/O to E/O)
    - Nukta normalization
    - Remove ZWNJ/ZWJ
    """

def maithili_normalize(text: str) -> str:
    """
    Maithili text normalization for TTS.
    - Similar to Bhojpuri with Maithili-specific handling
    """
```

#### Language Support Extension

**File**: `src/chatterbox/mtl_tts.py`

Added to `SUPPORTED_LANGUAGES`:
```python
"bho": "Bhojpuri",
"mai": "Maithili",
```

Updated `MTLTokenizer.encode()` to handle these languages.

---

### Phase 2: LoRA Configuration

#### LoRA Config Module

**File**: `src/chatterbox/models/t3/lora_config.py`

```python
@dataclass
class LoRAConfig:
    r: int = 16                    # LoRA rank
    lora_alpha: int = 32           # Scaling factor
    lora_dropout: float = 0.05     # Dropout
    target_modules: List[str]      # Attention + MLP layers
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"  # For LlamaModel
```

**Key Functions**:
- `get_lora_config()` - Get config for specific language
- `apply_lora_to_model()` - Apply LoRA adapters
- `save_lora_weights()` - Save adapter weights
- `load_lora_weights()` - Load adapter weights
- `merge_lora_weights()` - Merge for faster inference

#### T3 Model LoRA Methods

**File**: `src/chatterbox/models/t3/t3.py`

Added methods to T3 class:
- `apply_lora()` - Apply LoRA adapters
- `save_lora()` - Save LoRA weights
- `load_lora()` - Load LoRA weights
- `merge_lora()` - Merge into base model
- `set_lora_trainable()` - Freeze/unfreeze params
- `get_trainable_parameters()` - Get trainable params
- `print_trainable_parameters()` - Print summary

#### Loss Function Fix

Fixed cross-entropy loss tensor shape mismatch:
```python
# Before: F.cross_entropy(out.text_logits, masked_text)  # Shape error

# After: Reshape [B, S, V] -> [B*S, V]
text_logits_flat = out.text_logits.view(-1, V_text)
masked_text_flat = masked_text.view(-1)
loss_text = F.cross_entropy(text_logits_flat, masked_text_flat, ignore_index=IGNORE_ID)
```

---

### Phase 3: Training Infrastructure

#### Data Preparation Script

**File**: `scripts/prepare_data.py`

Commands:
- `create-manifest` - Create manifest from directory
- `prepare` - Process and validate dataset
- `split` - Split into train/val/test

#### IISc Data Converter

**File**: `scripts/prepare_iisc_data.py`

Converts IISc SYSPIN format to Chatterbox manifest format.

#### Training Script

**File**: `scripts/train_lora.py`

Features:
- `TTSDataset` class for loading audio + text
- `Trainer` class with training loop
- Mixed precision support
- Gradient clipping
- Checkpoint saving
- Validation evaluation

Key components:
```python
class TTSDataset(Dataset):
    # Loads audio, tokenizes text, extracts speaker embeddings

class Trainer:
    def train_epoch(self, epoch)  # Single epoch training
    def validate(self)            # Validation loop
    def save_checkpoint(self)     # Save LoRA + training state
    def train(self, num_epochs)   # Full training loop
```

---

### Phase 4: Configuration Files

**File**: `configs/lora_bho.yaml`
**File**: `configs/lora_mai.yaml`

YAML configuration for:
- LoRA hyperparameters
- Data paths
- Training settings
- Hardware configuration

---

### Phase 5: Inference Updates

**File**: `src/chatterbox/mtl_tts.py`

Added to `ChatterboxMultilingualTTS`:
- `load_lora_adapter()` - Load LoRA weights
- `unload_lora_adapter()` - Unload adapter
- `merge_lora_adapter()` - Merge weights
- `from_pretrained_with_lora()` - Load with LoRA
- `from_local_with_lora()` - Load local with LoRA

---

### Phase 6: Evaluation

**File**: `scripts/evaluate.py`

Commands:
- `evaluate` - Single model evaluation
- `compare` - Compare multiple models

Metrics:
- Speaker similarity
- Real-time factor (RTF)
- Duration statistics

---

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| `src/chatterbox/models/t3/lora_config.py` | LoRA configuration and utilities |
| `scripts/prepare_data.py` | General data preparation |
| `scripts/prepare_iisc_data.py` | IISc SYSPIN data converter |
| `scripts/train_lora.py` | LoRA training script |
| `scripts/evaluate.py` | Model evaluation |
| `configs/lora_bho.yaml` | Bhojpuri training config |
| `configs/lora_mai.yaml` | Maithili training config |
| `requirements-lora.txt` | LoRA dependencies |
| `docs/LORA_FINETUNING.md` | Documentation |
| `examples/inference_lora.py` | Inference example |
| `data/bhojpuri/train.json` | Training manifest |
| `data/bhojpuri/val.json` | Validation manifest |
| `data/bhojpuri/test.json` | Test manifest |
| `data/bhojpuri/all.json` | Full dataset manifest |
| `data/bhojpuri/train_small.json` | Small training set (500 samples) |
| `data/bhojpuri/val_small.json` | Small validation set (50 samples) |

### Modified Files

| File | Changes |
|------|---------|
| `src/chatterbox/__init__.py` | Added fallback version handling |
| `src/chatterbox/models/tokenizers/tokenizer.py` | Added Bhojpuri/Maithili normalization |
| `src/chatterbox/models/t3/t3.py` | Added LoRA methods, fixed loss function |
| `src/chatterbox/mtl_tts.py` | Added LoRA loading methods, language support |

---

## Data Preparation

### IISc SYSPIN Bhojpuri Dataset

**Source**: `/Users/ntiwari/IITP/chatterbox/data/IISc_SYSPIN_Data/IISc_SYSPINProject_Bhojpuri_Male_Spk001_HC`

#### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total samples | 26,012 |
| Total duration | ~41.7 hours |
| Average duration | 6.42 seconds |
| Sample rate | 48kHz |
| Speaker | Single male speaker |

#### Data Split

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 23,410 | 90% |
| Val | 1,300 | 5% |
| Test | 1,302 | 5% |

#### Domain Distribution

| Domain | Samples |
|--------|---------|
| SPORTS | 2,995 |
| TECHNOLOGY | 2,906 |
| AGRICULTURE | 2,859 |
| FINANCE | 2,854 |
| HEALTH | 2,821 |
| INDIA RELATED | 2,774 |
| FOOD | 2,569 |
| SOCIAL | 2,383 |
| LOCAL CONVERSATION | 1,803 |
| POLITICS | 1,749 |
| EVALUATION | 299 |

### Manifest Format

```json
[
  {
    "audio_path": "/path/to/audio.wav",
    "text": "Transcript in Devanagari",
    "language_id": "bho",
    "domain": "TECHNOLOGY",
    "id": "IISc_SYSPINProject_bho_m_TECH_01182"
  }
]
```

---

## Training Results

### Configuration

```bash
python scripts/train_lora.py \
    --train-manifest data/bhojpuri/train_small.json \
    --val-manifest data/bhojpuri/val_small.json \
    --language-id bho \
    --checkpoint-dir ~/.cache/huggingface/hub/models--ResembleAI--chatterbox/... \
    --output-dir outputs/bhojpuri_lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --epochs 2 \
    --batch-size 1 \
    --lr 1e-4 \
    --device mps
```

### Parameter Efficiency

| Metric | Value |
|--------|-------|
| Base model params | 514,692,096 |
| LoRA params | 11,304,960 |
| LoRA percentage | 2.20% |
| Total trainable | 43,907,072 |
| Trainable percentage | 8.02% |

### Loss Progression (95 samples)

| Step | Total Loss | Text Loss | Speech Loss |
|------|------------|-----------|-------------|
| 1 | 17.67 | 10.04 | 7.62 |
| 50 | 13.13 | 7.68 | 5.45 |
| 95 | 9.26 | 4.98 | 4.28 |

**Improvement**: 47.6% reduction in total loss

### Training Speed (MPS - Apple Silicon)

| Metric | Value |
|--------|-------|
| Initial speed | ~1s/sample |
| Average speed | ~1.5s/sample |
| Estimated full epoch | ~8-10 hours |

---

## Usage Instructions

### Installation

```bash
# Install base Chatterbox
pip install -e .

# Install LoRA dependencies
pip install -r requirements-lora.txt
```

### Data Preparation

```bash
# Convert IISc data
python scripts/prepare_iisc_data.py

# Or general preparation
python scripts/prepare_data.py create-manifest \
    --audio-dir /path/to/audio \
    --text-file transcripts.txt \
    --language-id bho \
    --output data/bhojpuri/manifest.json

python scripts/prepare_data.py split \
    --manifest data/bhojpuri/manifest.json \
    --output-dir data/bhojpuri
```

### Training

```bash
# Full training (GPU recommended)
python scripts/train_lora.py \
    --train-manifest data/bhojpuri/train.json \
    --val-manifest data/bhojpuri/val.json \
    --language-id bho \
    --checkpoint-dir /path/to/chatterbox/checkpoints \
    --output-dir outputs/bhojpuri_lora \
    --lora-rank 16 \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4 \
    --device cuda
```

### Inference

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load with LoRA
model = ChatterboxMultilingualTTS.from_pretrained_with_lora(
    device="cuda",
    lora_path="outputs/bhojpuri_lora/best_lora"
)

# Generate speech
wav = model.generate(
    text="नमस्ते, हमार नाम चैटरबॉक्स बा।",
    language_id="bho",
    audio_prompt_path="reference.wav"
)

# Save audio
import soundfile as sf
sf.write("output.wav", wav.squeeze().cpu().numpy(), model.sr)
```

### Evaluation

```bash
python scripts/evaluate.py evaluate \
    --checkpoint-dir /path/to/checkpoints \
    --test-manifest data/bhojpuri/test.json \
    --output-dir results/bhojpuri \
    --language-id bho \
    --lora-path outputs/bhojpuri_lora/best_lora
```

---

## Future Recommendations

### For Production Training

1. **Use GPU (CUDA)**: Training on MPS is slow (~1.5s/sample). CUDA would be 10-50x faster.

2. **Full Dataset**: Train on all 23,410 samples instead of the demo subset.

3. **More Epochs**: Train for 10-20 epochs for full convergence.

4. **Hyperparameter Tuning**:
   - LoRA rank: Try 8, 16, 32
   - Learning rate: 5e-5 to 2e-4
   - Batch size: 4-8 with gradient accumulation

5. **Multiple Speakers**: Add more speaker data for better generalization.

### For Maithili

1. Prepare Maithili dataset in same format
2. Run same pipeline with `--language-id mai`
3. Create separate LoRA adapter

### Multi-Language Support

```python
# Switch adapters at runtime
model.load_lora_adapter("outputs/bho_lora/best_lora")
wav_bho = model.generate(text, language_id="bho")

model.unload_lora_adapter()
model.load_lora_adapter("outputs/mai_lora/best_lora")
wav_mai = model.generate(text, language_id="mai")
```

### Optimization Ideas

1. **Gradient Checkpointing**: Reduce memory usage
2. **Mixed Precision**: Use bf16/fp16 for faster training
3. **Data Augmentation**: Speed perturbation, noise injection
4. **Curriculum Learning**: Train on shorter samples first

---

## Technical Notes

### Key Fixes Applied

1. **Cross-entropy loss shape**: The original loss function expected 2D tensors but received 3D. Fixed by reshaping logits from `[B,S,V]` to `[B*S,V]`.

2. **PEFT task type**: Changed from `CAUSAL_LM` to `FEATURE_EXTRACTION` because Chatterbox uses `LlamaModel` (encoder-only) not `LlamaForCausalLM`.

3. **Package version**: Added fallback for version detection when package not installed in development mode.

4. **MPS compatibility**: Tested and enabled training on Apple Silicon (MPS backend).

### Architecture Decisions

1. **Target Modules**: All attention (Q,K,V,O) and MLP (gate, up, down) projections for maximum adaptation.

2. **Frozen Components**: Voice Encoder, S3Gen, S3Tokenizer remain frozen - only T3 Llama backbone is adapted.

3. **Task Type**: Using `FEATURE_EXTRACTION` since we're training on embeddings, not causal language modeling.

---

## References

- [PEFT Library](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [IISc SYSPIN Project](https://syspin.iisc.ac.in/)

---

## Contact

For issues or questions about this implementation, please refer to the documentation in `docs/LORA_FINETUNING.md` or create an issue in the repository.

---

*Last updated: November 18, 2025*
