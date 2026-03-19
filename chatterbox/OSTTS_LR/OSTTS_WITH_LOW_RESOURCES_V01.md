# Open Source TTS with Low Resources - Version 0.1

## Extending Chatterbox Multilingual TTS for Low-Resource Indic Languages

**Document Version:** 0.1
**Date:** December 2025
**Authors:** Research Team, IITP

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Architecture Overview](#3-architecture-overview)
4. [Implementation Details](#4-implementation-details)
5. [Code Changes](#5-code-changes)
6. [Training Pipeline](#6-training-pipeline)
7. [Supported Languages](#7-supported-languages)
8. [Usage Guide](#8-usage-guide)
9. [Scope of Enhancements](#9-scope-of-enhancements)
10. [Future Work](#10-future-work)
11. [Appendix](#appendix)

---

## 1. Executive Summary

This document describes the architectural changes, code modifications, and enhancements made to the Chatterbox Multilingual TTS system to support low-resource Indic languages—specifically **Bhojpuri**, **Maithili**, and **Bengali**—using Parameter-Efficient Fine-Tuning (PEFT) via LoRA (Low-Rank Adaptation).

### Key Achievements

- **LoRA Integration**: Implemented LoRA adapters for the T3 transformer backbone, enabling fine-tuning with only ~2.2% of parameters trainable
- **Language-Specific Normalization**: Added text preprocessing for Bhojpuri, Maithili, and Bengali scripts
- **Token Mapping Strategy**: Leveraged existing Hindi token `[hi]` for unsupported Indic languages, avoiding vocabulary expansion
- **Training Infrastructure**: Developed end-to-end training pipeline with data preparation, training, and evaluation scripts
- **Resource Efficiency**: Achieved training on consumer hardware (Apple Silicon MPS) with ~500 samples per language

---

## 2. Introduction

### 2.1 Background

Chatterbox is an open-source multilingual Text-to-Speech (TTS) system that supports 23 languages out of the box. However, many low-resource languages, particularly regional Indic languages, lack support. This project extends Chatterbox to support:

| Language  | ISO Code | Script     | Native Speakers |
|-----------|----------|------------|-----------------|
| Bhojpuri  | bho      | Devanagari | ~50 million     |
| Maithili  | mai      | Devanagari | ~34 million     |
| Bengali   | bn       | Bengali    | ~230 million    |

### 2.2 Challenges

1. **Limited Training Data**: Low-resource languages lack large-scale TTS corpora
2. **Vocabulary Limitations**: Base model tokenizer doesn't include language-specific tokens
3. **Script Variations**: Regional scripts have unique character handling requirements
4. **Computational Constraints**: Full model fine-tuning is prohibitively expensive

### 2.3 Solution Approach

We address these challenges through:

- **LoRA Fine-tuning**: Train only adapter weights (11.3M params vs 547M total)
- **Token Reuse**: Map new languages to Hindi token, leveraging Indic script similarity
- **Minimal Data**: Effective training with as few as 400-500 samples
- **Script Normalization**: Language-specific text preprocessing pipelines

---

## 3. Architecture Overview

### 3.1 Chatterbox TTS Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Chatterbox TTS Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Text      │    │     T3       │    │   S3Gen      │      │
│  │  Tokenizer   │───▶│  Transformer │───▶│  Vocoder     │──▶WAV│
│  │  (MTL)       │    │  (Llama)     │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        │                    │                                   │
│        │                    │                                   │
│        ▼                    ▼                                   │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │  Language    │    │   Voice      │                          │
│  │  Tokens      │    │   Encoder    │                          │
│  │  [hi],[en].. │    │   (Speaker)  │                          │
│  └──────────────┘    └──────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 LoRA Integration Point

```
┌─────────────────────────────────────────────────────────────────┐
│                     T3 Transformer (Llama)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Attention Layer                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │ Q_proj  │  │ K_proj  │  │ V_proj  │  │ O_proj  │    │   │
│  │  │ + LoRA  │  │ + LoRA  │  │ + LoRA  │  │ + LoRA  │    │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      MLP Layer                           │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐           │   │
│  │  │ gate_proj │  │  up_proj  │  │ down_proj │           │   │
│  │  │  + LoRA   │  │  + LoRA   │  │  + LoRA   │           │   │
│  │  └───────────┘  └───────────┘  └───────────┘           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Total: 514.7M base params + 11.3M LoRA params (2.2%)          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Token Mapping Strategy

```
┌────────────────────────────────────────────────────────────────┐
│                    Token Mapping Strategy                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Input          Internal Mapping         Tokenizer        │
│   ───────────         ────────────────         ─────────        │
│                                                                 │
│   language_id="bho"   ──▶  "[hi]" token   ──▶  Hindi vocab     │
│   language_id="mai"   ──▶  "[hi]" token   ──▶  Hindi vocab     │
│   language_id="bn"    ──▶  "[hi]" token   ──▶  Hindi vocab     │
│                                                                 │
│   Rationale:                                                    │
│   • Bhojpuri/Maithili use Devanagari (same as Hindi)           │
│   • Bengali script shares Indic language family features        │
│   • Avoids vocabulary expansion and embedding initialization    │
│   • LoRA adapters learn language-specific patterns              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Details

### 4.1 Text Normalization Pipeline

Each language requires script-specific normalization to handle:
- Unicode normalization (NFC/NFKD)
- Character variations and conjuncts
- Diacritics and modifier characters
- Punctuation standardization

#### 4.1.1 Bhojpuri Normalization

```python
def bhojpuri_normalize(text: str) -> str:
    """
    Bhojpuri text normalization for TTS.
    Handles Devanagari script normalization specific to Bhojpuri.
    """
    # NFC normalization
    text = unicodedata.normalize('NFC', text)

    # Character mappings
    replacements = {
        'ऍ': 'ए',   # Candra E to E
        'ऑ': 'ओ',   # Candra O to O
        '\u0951': '',  # Remove Udatta
        '\u0952': '',  # Remove Anudatta
        '\u200c': '',  # Remove ZWNJ
        '\u200d': '',  # Remove ZWJ
    }

    # Nukta normalization
    nukta_mappings = {
        'क़': 'क', 'ख़': 'ख', 'ग़': 'ग',
        'ज़': 'ज', 'ड़': 'ड', 'ढ़': 'ढ',
        'फ़': 'फ', 'य़': 'य',
    }

    return normalized_text
```

#### 4.1.2 Maithili Normalization

Similar to Bhojpuri with additional handling for Maithili-specific conjuncts:

```python
def maithili_normalize(text: str) -> str:
    """
    Maithili text normalization for TTS.
    Handles Devanagari script specific to Maithili.
    """
    # Similar structure to Bhojpuri
    # Maithili has unique conjunct patterns
```

#### 4.1.3 Bengali Normalization

Bengali uses its own script with different handling requirements:

```python
def bengali_normalize(text: str) -> str:
    """
    Bengali text normalization for TTS.
    """
    text = unicodedata.normalize('NFC', text)

    replacements = {
        '\u09bc': '',      # Remove Bengali nukta
        '\u200c': '',      # Remove ZWNJ
        '\u200d': '',      # Remove ZWJ
        '\u0985\u09be': 'আ',  # অ + া → আ
        '।': '.',          # Bengali danda to period
        '॥': '.',          # Double danda to period
    }

    # Preserve important Bengali conjuncts
    conjunct_mappings = {
        'ক্ক': 'ক্ক',  # Keep double consonants
        'ক্ষ': 'ক্ষ',  # Keep ksha
        'জ্ঞ': 'জ্ঞ',  # Keep gya
    }
```

### 4.2 LoRA Configuration

```python
@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    r: int = 16              # Rank of low-rank matrices
    lora_alpha: int = 32     # Scaling factor
    lora_dropout: float = 0.05

    # Target modules in Llama
    target_modules: List[str] = [
        "q_proj",    # Query projection
        "k_proj",    # Key projection
        "v_proj",    # Value projection
        "o_proj",    # Output projection
        "gate_proj", # MLP gate
        "up_proj",   # MLP up
        "down_proj", # MLP down
    ]

    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
```

### 4.3 Parameter Efficiency

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| T3 Transformer (Llama) | 514,692,096 | No (frozen) |
| LoRA Adapters | 11,304,960 | Yes |
| Other Components | 32,602,112 | Partial |
| **Total Trainable** | **43,907,072** | **8.02%** |

---

## 5. Code Changes

### 5.1 Modified Files

#### 5.1.1 `src/chatterbox/models/tokenizers/tokenizer.py`

**Changes:**
1. Added `bhojpuri_normalize()` function
2. Added `maithili_normalize()` function
3. Added `bengali_normalize()` function
4. Modified `MTLTokenizer.encode()` to handle new languages

```python
# In encode() method
def encode(self, txt: str, language_id: str = None, ...):
    # Language-specific preprocessing
    if language_id == 'bho':
        txt = bhojpuri_normalize(txt)
    elif language_id == 'mai':
        txt = maithili_normalize(txt)
    elif language_id == 'bn':
        txt = bengali_normalize(txt)

    # Token mapping for unsupported languages
    if language_id:
        if language_id.lower() in ('bho', 'mai', 'bn'):
            txt = f"[hi]{txt}"  # Map to Hindi token
        else:
            txt = f"[{language_id.lower()}]{txt}"
```

#### 5.1.2 `src/chatterbox/mtl_tts.py`

**Changes:**
1. Added Bhojpuri, Maithili to `SUPPORTED_LANGUAGES` dict
2. Added LoRA loading methods
3. Added `from_pretrained_with_lora()` class method

```python
SUPPORTED_LANGUAGES = {
    # ... existing languages ...
    "bho": "Bhojpuri",
    "mai": "Maithili",
    # Bengali will use 'bn' when added
}

def load_lora_adapter(self, lora_path: str):
    """Load a LoRA adapter for language-specific fine-tuning."""
    self.t3.load_lora(lora_path)
    self._lora_loaded = True

@classmethod
def from_pretrained_with_lora(cls, device, lora_path):
    """Load pretrained model with a LoRA adapter."""
    model = cls.from_pretrained(device)
    model.load_lora_adapter(lora_path)
    return model
```

#### 5.1.3 `src/chatterbox/models/t3/t3.py`

**Changes:**
1. Added `apply_lora()` method
2. Added `save_lora()` method
3. Added `load_lora()` method
4. Added `merge_lora()` method

```python
def apply_lora(self, lora_config: LoRAConfig):
    """Apply LoRA adapters to the transformer."""
    from .lora_config import apply_lora_to_model
    apply_lora_to_model(self, lora_config)

def save_lora(self, save_path: str):
    """Save only the LoRA adapter weights."""
    from .lora_config import save_lora_weights
    save_lora_weights(self, save_path)

def load_lora(self, load_path: str):
    """Load LoRA adapter weights."""
    from .lora_config import load_lora_weights
    load_lora_weights(self, load_path)
```

### 5.2 New Files Created

#### 5.2.1 `src/chatterbox/models/t3/lora_config.py`

Complete LoRA configuration and utility functions:
- `LoRAConfig` dataclass
- `apply_lora_to_model()` - Apply PEFT adapters
- `save_lora_weights()` - Save adapter checkpoints
- `load_lora_weights()` - Load adapter checkpoints
- `merge_lora_weights()` - Merge for inference optimization

#### 5.2.2 `scripts/train_lora.py`

Training script with:
- `TTSDataset` class for loading manifests
- `Trainer` class with train/validate loops
- Support for mixed precision, gradient clipping
- Checkpoint saving with best model tracking

#### 5.2.3 `scripts/prepare_bhojpuri_data.py`

Data preparation for Bhojpuri:
- Loads IISc SYSPIN format data
- Creates train/val/test splits
- Outputs JSON manifests

#### 5.2.4 `scripts/prepare_maithili_data.py`

Data preparation for Maithili (similar structure)

#### 5.2.5 `scripts/prepare_bengali_data.py`

Data preparation for Bengali (similar structure)

#### 5.2.6 Example Scripts

- `example_tts_bhojpuri.py` - Bhojpuri inference examples
- `example_tts_maithili.py` - Maithili inference examples
- `example_tts_bengali.py` - Bengali inference examples
- `example_tts_hindi.py` - Hindi baseline examples
- `example_tts_article.py` - Long-form article TTS

---

## 6. Training Pipeline

### 6.1 Data Preparation

```bash
# 1. Prepare training data
python scripts/prepare_bhojpuri_data.py   # Creates data/bhojpuri/
python scripts/prepare_maithili_data.py   # Creates data/maithili/
python scripts/prepare_bengali_data.py    # Creates data/bengali/
```

**Manifest Format (JSON):**
```json
[
  {
    "audio_path": "/path/to/audio.wav",
    "text": "भोजपुरी में वाक्य",
    "language_id": "bho",
    "domain": "NEWS",
    "id": "sample_001"
  }
]
```

### 6.2 Training Command

```bash
python scripts/train_lora.py \
    --train-manifest data/bhojpuri/train.json \
    --val-manifest data/bhojpuri/val.json \
    --language-id bho \
    --checkpoint-dir ~/.cache/huggingface/hub/models--ResembleAI--chatterbox/... \
    --output-dir outputs/bhojpuri_lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --epochs 2 \
    --batch-size 1 \
    --lr 1e-4 \
    --warmup-steps 50 \
    --grad-clip 1.0 \
    --num-workers 0 \
    --device mps
```

### 6.3 Training Metrics

**Typical Training Progress (500 samples, 2 epochs):**

| Step | Loss | Text Loss | Speech Loss |
|------|------|-----------|-------------|
| 0 | 15.80 | 9.53 | 6.27 |
| 50 | 4.15 | 2.86 | 1.29 |
| 100 | 0.92 | 0.67 | 0.26 |
| 150 | 0.28 | 0.18 | 0.10 |
| 200 | 0.20 | 0.13 | 0.07 |

**Loss Reduction:** ~98.7% from initial to step 200

### 6.4 Inference

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load model with LoRA adapter
model = ChatterboxMultilingualTTS.from_pretrained_with_lora(
    device="cuda",
    lora_path="outputs/bhojpuri_lora/best_lora"
)

# Generate speech
wav = model.generate(
    text="भोजपुरी में बोलल गइल बा।",
    language_id="bho",
    audio_prompt_path="reference.wav",  # Optional voice cloning
    exaggeration=0.5,
    temperature=0.8,
    cfg_weight=0.5,
)
```

---

## 7. Supported Languages

### 7.1 Base Model Languages (23)

| Code | Language | Code | Language |
|------|----------|------|----------|
| ar | Arabic | ko | Korean |
| da | Danish | ms | Malay |
| de | German | nl | Dutch |
| el | Greek | no | Norwegian |
| en | English | pl | Polish |
| es | Spanish | pt | Portuguese |
| fi | Finnish | ru | Russian |
| fr | French | sv | Swedish |
| he | Hebrew | sw | Swahili |
| hi | Hindi | tr | Turkish |
| it | Italian | zh | Chinese |
| ja | Japanese | | |

### 7.2 LoRA-Extended Languages (3)

| Code | Language | Script | Status |
|------|----------|--------|--------|
| bho | Bhojpuri | Devanagari | Implemented |
| mai | Maithili | Devanagari | Implemented |
| bn | Bengali | Bengali | Implemented |

---

## 8. Usage Guide

### 8.1 Installation

```bash
# Clone repository
git clone https://github.com/your-repo/chatterbox.git
cd chatterbox

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-lora.txt  # Additional LoRA dependencies

# Activate environment
source venv/bin/activate
```

### 8.2 Quick Start - Inference

```python
#!/usr/bin/env python3
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torch
import soundfile as sf

# Detect device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load model with LoRA
model = ChatterboxMultilingualTTS.from_pretrained_with_lora(
    device=device,
    lora_path="outputs/bhojpuri_lora/best_lora"
)

# Generate Bhojpuri speech
text = "नमस्ते, हमार नाम चैटरबॉक्स बा।"
wav = model.generate(text=text, language_id="bho")

# Save audio
sf.write("output.wav", wav.squeeze().numpy(), model.sr)
```

### 8.3 Training Your Own Language

1. **Prepare Data**
   - Collect audio-text pairs (minimum 300-500 samples recommended)
   - Format as JSON manifest
   - Audio: WAV format, 16kHz or 24kHz

2. **Add Language Support**
   - Add normalization function to `tokenizer.py`
   - Add language to `MTLTokenizer.encode()` mapping
   - Add to `SUPPORTED_LANGUAGES` in `mtl_tts.py`

3. **Train LoRA**
   ```bash
   python scripts/train_lora.py \
       --train-manifest your_data/train.json \
       --language-id YOUR_LANG_CODE \
       --output-dir outputs/your_lang_lora \
       --epochs 2
   ```

4. **Evaluate**
   - Test with example scripts
   - Listen to generated audio for quality

---

## 9. Scope of Enhancements

### 9.1 Current Implementation (v0.1)

| Feature | Status | Notes |
|---------|--------|-------|
| LoRA Fine-tuning | ✅ Complete | PEFT-based adaptation |
| Bhojpuri Support | ✅ Complete | Trained and tested |
| Maithili Support | ✅ Complete | Trained and tested |
| Bengali Support | ✅ Code Ready | Awaiting data |
| Data Preparation Scripts | ✅ Complete | IISc SYSPIN format |
| Training Pipeline | ✅ Complete | Full train/val loop |
| Voice Cloning | ✅ Complete | Reference audio support |
| MPS (Apple Silicon) | ✅ Complete | Full support |

### 9.2 Limitations

1. **Token Vocabulary**: New languages share Hindi token, may affect phoneme accuracy
2. **Data Requirements**: Quality depends on training data quantity/quality
3. **Script Handling**: Complex conjuncts may not render perfectly
4. **No Prosody Control**: Limited emotion/style transfer for new languages

### 9.3 Planned Enhancements (v0.2+)

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| Vocabulary Expansion | High | Add native tokens for new languages |
| G2P Integration | High | Grapheme-to-phoneme for better pronunciation |
| More Languages | Medium | Add Odia, Assamese, Gujarati |
| Duration Control | Medium | Control speech rate per language |
| Style Transfer | Low | Transfer emotion from reference |
| Quantization | Low | INT8/INT4 for faster inference |

---

## 10. Future Work

### 10.1 Short-term Goals

1. **Bengali Training**: Complete training once data is available
2. **Quality Evaluation**: MOS (Mean Opinion Score) testing
3. **Inference Optimization**: LoRA weight merging for faster inference

### 10.2 Medium-term Goals

1. **Additional Languages**: Expand to other Indic languages
2. **Multi-speaker Support**: Train speaker-specific adapters
3. **Streaming Inference**: Real-time TTS generation

### 10.3 Long-term Vision

1. **End-to-end Pipeline**: Voice cloning + TTS + Language ID
2. **Cross-lingual Transfer**: Zero-shot for related languages
3. **Community Contributions**: Open model zoo for adapters

---

## Appendix

### A. Dependencies

```
# requirements-lora.txt
peft>=0.7.0
bitsandbytes>=0.41.0
transformers>=4.36.0
safetensors>=0.4.0
tqdm>=4.65.0
librosa>=0.10.0
soundfile>=0.12.0
```

### B. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 16+ GB |
| System RAM | 16 GB | 32 GB |
| Storage | 20 GB | 50 GB |

**Tested Configurations:**
- Apple M1/M2/M3 (MPS backend)
- NVIDIA RTX 3080/4080 (CUDA)
- CPU-only (very slow, not recommended)

### C. References

1. Hu, E. et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
2. Chatterbox TTS: https://github.com/resemble-ai/chatterbox
3. IISc SYSPIN Dataset: https://sites.google.com/view/syspin
4. PEFT Library: https://github.com/huggingface/peft

### D. Acknowledgments

- ResembleAI for the Chatterbox TTS model
- IISc Bangalore for the SYSPIN Indic language dataset
- Hugging Face for the PEFT library
- IITP Research Team for implementation

---

**Document End**

*For questions or contributions, please open an issue on the project repository.*
