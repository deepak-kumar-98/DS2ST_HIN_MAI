# Chatterbox TTS: Low-Resource Language Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Details: Bhojpuri & Maithili](#implementation-details-bhojpuri--maithili)
4. [Workflow](#workflow)
5. [Code Walkthrough](#code-walkthrough)
6. [Adding New Languages: Bengali, Odia, Marathi](#adding-new-languages-bengali-odia-marathi)
7. [Assessment: Current LoRA Implementation Sufficiency](#assessment-current-lora-implementation-sufficiency)
8. [Recommendations](#recommendations)

---

## Overview

Chatterbox TTS is a multilingual text-to-speech system that uses a Token-To-Token (T3) architecture with a Llama-based transformer backbone. This document explains how low-resource Indian languages (Bhojpuri and Maithili) have been incorporated using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA (Low-Rank Adaptation)**.

### Why LoRA for Low-Resource Languages?

- **Efficiency**: Only 0.5-2% of parameters are trained, reducing memory and compute requirements
- **Preservation**: Base model knowledge (Hindi, English, etc.) is preserved
- **Portability**: Small adapter files (~50-100MB) instead of full model checkpoints (~4GB+)
- **Modularity**: Easy to switch between language adapters

---

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Chatterbox TTS Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Text       │    │   T3 Model   │    │      S3Gen               │  │
│  │   Input      │───▶│   (Llama +   │───▶│   (Speech                │  │
│  │              │    │    LoRA)     │    │    Synthesis)            │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│         │                   │                       │                   │
│         ▼                   ▼                       ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │ MTLTokenizer │    │   Speech     │    │      Audio               │  │
│  │ (Text→Tokens)│    │   Tokens     │    │      Output              │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### T3 Model with LoRA Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         T3 Model Architecture                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                    Llama Transformer (Frozen)                   │   │
│   │  ┌──────────────────────────────────────────────────────────┐  │   │
│   │  │  Self-Attention Layer                                     │  │   │
│   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │  │   │
│   │  │  │ Q_proj  │  │ K_proj  │  │ V_proj  │  │ O_proj  │      │  │   │
│   │  │  │ +LoRA_A │  │ +LoRA_A │  │ +LoRA_A │  │ +LoRA_A │      │  │   │
│   │  │  │ +LoRA_B │  │ +LoRA_B │  │ +LoRA_B │  │ +LoRA_B │      │  │   │
│   │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │  │   │
│   │  └──────────────────────────────────────────────────────────┘  │   │
│   │  ┌──────────────────────────────────────────────────────────┐  │   │
│   │  │  MLP Layer                                                │  │   │
│   │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                │  │   │
│   │  │  │ gate_proj│  │ up_proj  │  │down_proj │                │  │   │
│   │  │  │ +LoRA_A  │  │ +LoRA_A  │  │ +LoRA_A  │                │  │   │
│   │  │  │ +LoRA_B  │  │ +LoRA_B  │  │ +LoRA_B  │                │  │   │
│   │  │  └──────────┘  └──────────┘  └──────────┘                │  │   │
│   │  └──────────────────────────────────────────────────────────┘  │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │
│   │  Text Embedding │  │ Speech Embedding│  │  Conditioning Enc   │   │
│   │    (Frozen)     │  │    (Frozen)     │  │     (Frozen)        │   │
│   └─────────────────┘  └─────────────────┘  └─────────────────────┘   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### LoRA Mathematics

The LoRA adaptation works by decomposing weight updates:

```
W' = W + ΔW = W + BA

Where:
- W: Original frozen weights (d × k)
- B: Low-rank matrix (d × r)
- A: Low-rank matrix (r × k)
- r: Rank (typically 8-32, much smaller than d and k)
```

**Current Configuration:**
- Rank (r): 16
- Alpha (scaling): 32
- Dropout: 0.05

---

## Implementation Details: Bhojpuri & Maithili

### 1. Text Normalization (`src/chatterbox/models/tokenizers/tokenizer.py`)

Both languages use Devanagari script and share text normalization logic:

```python
def bhojpuri_normalize(text: str) -> str:
    """Bhojpuri text normalization for TTS."""
    # NFC normalization for Devanagari
    text = unicodedata.normalize('NFC', text)

    # Replace non-standard characters with standard forms
    replacements = {
        'ऍ': 'ए',  # Candra E to E
        'ऑ': 'ओ',  # Candra O to O
        '\u0951': '',  # Remove Udatta
        '\u0952': '',  # Remove Anudatta
        '\u200c': '',  # Remove ZWNJ
        '\u200d': '',  # Remove ZWJ
    }

    # Normalize nukta combinations
    nukta_mappings = {
        'क़': 'क', 'ख़': 'ख', 'ग़': 'ग', 'ज़': 'ज',
        'ड़': 'ड', 'ढ़': 'ढ', 'फ़': 'फ', 'य़': 'य',
    }
    # ... apply replacements
    return text.strip()

def maithili_normalize(text: str) -> str:
    """Maithili text normalization - similar to Bhojpuri."""
    # Same approach as Bhojpuri since both use Devanagari
    # ...
```

### 2. Language Token Strategy

Since Bhojpuri and Maithili don't have dedicated tokens in the base vocabulary, they leverage Hindi's grapheme tokens:

```python
# In MTLTokenizer.encode() at tokenizer.py:394-400
if language_id.lower() in ('bho', 'mai'):
    txt = f"[hi]{txt}"  # Use Hindi language token
else:
    txt = f"[{language_id.lower()}]{txt}"
```

This approach works because:
- All three languages share Devanagari script
- The base model has learned Hindi phoneme patterns
- LoRA adapts these patterns for regional pronunciation differences

### 3. LoRA Configuration (`src/chatterbox/models/t3/lora_config.py`)

```python
@dataclass
class LoRAConfig:
    r: int = 16                    # Rank
    lora_alpha: int = 32           # Scaling factor (usually 2x rank)
    lora_dropout: float = 0.05    # Regularization

    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",      # Query projection in attention
        "k_proj",      # Key projection
        "v_proj",      # Value projection
        "o_proj",      # Output projection
        "gate_proj",   # MLP gating
        "up_proj",     # MLP up-projection
        "down_proj",   # MLP down-projection
    ])

    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"  # Not CAUSAL_LM

    @classmethod
    def for_bhojpuri(cls, r: int = 16):
        return cls(r=r, lora_alpha=r * 2, language_id="bho")

    @classmethod
    def for_maithili(cls, r: int = 16):
        return cls(r=r, lora_alpha=r * 2, language_id="mai")
```

### 4. T3 Model LoRA Integration (`src/chatterbox/models/t3/t3.py`)

```python
class T3(nn.Module):
    def apply_lora(self, lora_config, language=None, rank=16):
        """Apply LoRA to Llama backbone."""
        apply_lora_to_model(self, lora_config)
        self._lora_applied = True

        # Required for attention output during inference
        self.tfmr.config._attn_implementation = 'eager'
        return self

    def load_lora(self, load_path: str):
        """Load pre-trained LoRA weights."""
        load_lora_weights(self, load_path)
        self._lora_applied = True
        return self

    def save_lora(self, save_path: str):
        """Save only LoRA adapter weights."""
        save_lora_weights(self, save_path)

    def merge_lora(self):
        """Merge LoRA into base weights (faster inference)."""
        merge_lora_weights(self)
        self._lora_applied = False
        return self
```

---

## Workflow

### Training Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LoRA Training Pipeline                           │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: Data Preparation
─────────────────────────
   Raw Audio + Text        scripts/prepare_data.py         Manifest JSON
        │                          │                            │
        │    ┌─────────────────────┴──────────────────────┐    │
        ▼    ▼                                            ▼    ▼
   ┌─────────────┐    ┌─────────────────────┐    ┌─────────────────┐
   │  Validate   │───▶│  Resample to 16kHz  │───▶│  Create JSON    │
   │   Audio     │    │  Normalize Volume   │    │  Manifest       │
   └─────────────┘    └─────────────────────┘    └─────────────────┘

Step 2: Model Setup
───────────────────
   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │  Load Pretrained │───▶│  Apply LoRA      │───▶│  Freeze Base     │
   │  T3 Model        │    │  Adapters        │    │  Parameters      │
   └──────────────────┘    └──────────────────┘    └──────────────────┘

Step 3: Training Loop (scripts/train_lora.py)
─────────────────────────────────────────────
   ┌─────────────────────────────────────────────────────────────────┐
   │  For each batch:                                                 │
   │    1. Encode text with MTLTokenizer (language_id="bho"/"mai")   │
   │    2. Extract speech tokens with S3Tokenizer                     │
   │    3. Compute speaker embedding with VoiceEncoder                │
   │    4. Forward pass through T3 (LoRA layers active)               │
   │    5. Compute loss: loss_text + loss_speech                      │
   │    6. Backprop (only LoRA parameters updated)                    │
   │    7. Save checkpoint with LoRA weights only                     │
   └─────────────────────────────────────────────────────────────────┘

Step 4: Export
──────────────
   ┌──────────────────┐
   │  Save LoRA       │───▶  outputs/bhojpuri_lora/best_lora/
   │  Adapter (~50MB) │      ├── adapter_config.json
   └──────────────────┘      └── adapter_model.safetensors
```

### Inference Workflow

```python
# examples/inference_lora.py

# 1. Load base model with LoRA adapter
model = ChatterboxMultilingualTTS.from_pretrained_with_lora(
    device=device,
    lora_path="outputs/bhojpuri_lora/best_lora"
)

# 2. Prepare voice conditioning from reference audio
model.prepare_conditionals("reference_speaker.wav")

# 3. Generate speech
wav = model.generate(
    text="राम राम भाई!",
    language_id="bho",  # Triggers bhojpuri_normalize + [hi] token
    temperature=0.8,
    cfg_weight=0.5,
)

# 4. Save output
sf.write("output.wav", wav.squeeze().numpy(), model.sr)
```

---

## Code Walkthrough

### Key Files and Their Roles

| File | Purpose |
|------|---------|
| `src/chatterbox/models/t3/lora_config.py` | LoRA configuration and PEFT integration |
| `src/chatterbox/models/t3/t3.py` | T3 model with LoRA apply/load/save methods |
| `src/chatterbox/models/tokenizers/tokenizer.py` | Text normalization for Bhojpuri/Maithili |
| `src/chatterbox/mtl_tts.py` | High-level TTS interface with LoRA support |
| `scripts/train_lora.py` | Training script for LoRA fine-tuning |
| `scripts/prepare_data.py` | Data preparation utilities |
| `configs/lora_bho.yaml` | Bhojpuri training configuration |
| `configs/lora_mai.yaml` | Maithili training configuration |

### Data Format

**Manifest JSON Structure:**
```json
[
  {
    "audio_path": "/path/to/audio.wav",
    "text": "राम राम भाई!",
    "language_id": "bho",
    "duration": 2.5,
    "speaker_id": "speaker001"
  }
]
```

### Training Command

```bash
python scripts/train_lora.py \
    --train-manifest data/bhojpuri/train.json \
    --val-manifest data/bhojpuri/val.json \
    --language-id bho \
    --checkpoint-dir ~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/... \
    --output-dir outputs/bhojpuri_lora \
    --lora-rank 16 \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4 \
    --mixed-precision
```

---

## Adding New Languages: Bengali, Odia, Marathi

### Step-by-Step Guide

#### 1. Add Text Normalizer

Create normalization functions in `src/chatterbox/models/tokenizers/tokenizer.py`:

```python
def bengali_normalize(text: str) -> str:
    """Bengali text normalization for TTS."""
    import unicodedata

    # NFC normalization for Bengali script
    text = unicodedata.normalize('NFC', text)

    # Bengali-specific character mappings
    replacements = {
        '\u09BC': '',  # Remove Nukta if needed
        '\u200c': '',  # Remove ZWNJ
        '\u200d': '',  # Remove ZWJ
        'ৎ': 'ত্',    # Khanda Ta normalization (optional)
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return ' '.join(text.split()).strip()


def odia_normalize(text: str) -> str:
    """Odia (Oriya) text normalization for TTS."""
    import unicodedata

    text = unicodedata.normalize('NFC', text)

    replacements = {
        '\u200c': '',  # Remove ZWNJ
        '\u200d': '',  # Remove ZWJ
        '\u0B3C': '',  # Remove Odia Nukta
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return ' '.join(text.split()).strip()


def marathi_normalize(text: str) -> str:
    """Marathi text normalization for TTS."""
    import unicodedata

    # Marathi uses Devanagari, similar to Hindi
    text = unicodedata.normalize('NFC', text)

    # Marathi-specific handling
    replacements = {
        'ळ': 'ळ',     # Ensure standard form of Marathi-specific ळ
        '\u0951': '', # Remove Udatta
        '\u0952': '', # Remove Anudatta
        '\u200c': '', # Remove ZWNJ
        '\u200d': '', # Remove ZWJ
    }

    # Nukta handling
    nukta_mappings = {
        'क़': 'क', 'ख़': 'ख', 'ग़': 'ग', 'ज़': 'ज',
        'ड़': 'ड', 'ढ़': 'ढ', 'फ़': 'फ',
    }

    for old, new in {**replacements, **nukta_mappings}.items():
        text = text.replace(old, new)

    return ' '.join(text.split()).strip()
```

#### 2. Register in MTLTokenizer

Update the `encode` method in `tokenizer.py`:

```python
def encode(self, txt: str, language_id: str = None, ...):
    # ... existing code ...

    # Add new language cases
    elif language_id == 'bn':
        txt = bengali_normalize(txt)
    elif language_id == 'or':
        txt = odia_normalize(txt)
    elif language_id == 'mr':
        txt = marathi_normalize(txt)

    # Language token strategy
    if language_id:
        # Languages that can share existing tokens:
        # - Bengali (bn): Could use [hi] or needs [bn] if in vocab
        # - Odia (or): Could use [hi] or needs dedicated token
        # - Marathi (mr): Uses Devanagari, can use [hi]
        if language_id.lower() in ('bho', 'mai', 'mr'):
            txt = f"[hi]{txt}"  # Devanagari languages
        elif language_id.lower() in ('bn', 'or'):
            # Check if dedicated token exists
            if f"[{language_id.lower()}]" in self.tokenizer.get_vocab():
                txt = f"[{language_id.lower()}]{txt}"
            else:
                txt = f"[hi]{txt}"  # Fallback to Hindi
        else:
            txt = f"[{language_id.lower()}]{txt}"
```

#### 3. Add to Supported Languages

Update `src/chatterbox/mtl_tts.py`:

```python
SUPPORTED_LANGUAGES = {
    # ... existing languages ...
    "bn": "Bengali",
    "mr": "Marathi",
    "or": "Odia",
}
```

#### 4. Create LoRA Config Methods

Add to `src/chatterbox/models/t3/lora_config.py`:

```python
@classmethod
def for_bengali(cls, r: int = 16):
    """LoRA config for Bengali."""
    return cls(r=r, lora_alpha=r * 2, lora_dropout=0.05, language_id="bn")

@classmethod
def for_marathi(cls, r: int = 16):
    """LoRA config for Marathi."""
    return cls(r=r, lora_alpha=r * 2, lora_dropout=0.05, language_id="mr")

@classmethod
def for_odia(cls, r: int = 16):
    """LoRA config for Odia."""
    return cls(r=r, lora_alpha=r * 2, lora_dropout=0.05, language_id="or")


def get_lora_config(language: str = None, rank: int = 16) -> LoRAConfig:
    configs = {
        "bho": LoRAConfig.for_bhojpuri,
        "mai": LoRAConfig.for_maithili,
        "bn": LoRAConfig.for_bengali,
        "mr": LoRAConfig.for_marathi,
        "or": LoRAConfig.for_odia,
    }
    if language in configs:
        return configs[language](r=rank)
    return LoRAConfig(r=rank, lora_alpha=rank * 2, language_id=language)
```

#### 5. Create Configuration Files

`configs/lora_bn.yaml`:
```yaml
language_id: "bn"
language_name: "Bengali"

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

data:
  train_manifest: "data/bengali/train.json"
  val_manifest: "data/bengali/val.json"
  max_text_len: 512
  max_speech_len: 2048

training:
  epochs: 10
  batch_size: 4
  learning_rate: 1.0e-4
  warmup_steps: 500
  mixed_precision: true
```

---

## Assessment: Current LoRA Implementation Sufficiency

### What Works Well

| Aspect | Status | Notes |
|--------|--------|-------|
| **LoRA Infrastructure** | ✅ Complete | Full apply/load/save/merge support |
| **Training Pipeline** | ✅ Complete | TTSDataset, Trainer, checkpointing |
| **Text Normalization** | ✅ Complete | Extensible design for new languages |
| **Inference** | ✅ Complete | Seamless LoRA loading |
| **Configuration** | ✅ Complete | YAML configs, language-specific presets |

### Potential Gaps for New Languages

| Aspect | Status | Recommendation |
|--------|--------|----------------|
| **Script Support** | ⚠️ Partial | Bengali/Odia need dedicated tokenizer support |
| **Language Tokens** | ⚠️ Depends | Check if [bn], [or] exist in base vocab |
| **Phoneme Coverage** | ⚠️ Varies | May need additional grapheme mappings |
| **Evaluation Metrics** | ⚠️ Missing | Add MOS, intelligibility scoring |

### Required Additional Steps for Bengali, Odia, Marathi

#### Definitely Required:

1. **Text Normalizers**: Create `bengali_normalize()`, `odia_normalize()`, `marathi_normalize()` functions
2. **Register Languages**: Add to `SUPPORTED_LANGUAGES` dict
3. **Config Files**: Create `lora_bn.yaml`, `lora_or.yaml`, `lora_mr.yaml`
4. **Training Data**: Prepare manifests with audio-text pairs

#### Potentially Required:

1. **Tokenizer Vocabulary Check**:
   ```python
   # Check if language tokens exist
   tokenizer = MTLTokenizer("grapheme_mtl_merged_expanded_v1.json")
   vocab = tokenizer.tokenizer.get_vocab()
   print("[bn]" in vocab)  # Bengali
   print("[or]" in vocab)  # Odia
   print("[mr]" in vocab)  # Marathi
   ```

2. **Script-Specific Handling**:
   - Bengali and Odia use different scripts than Devanagari
   - May need grapheme-to-phoneme conversion if base tokenizer lacks coverage
   - Consider adding script-specific special characters

3. **Higher LoRA Rank for Distant Languages**:
   ```python
   # For languages with different scripts, consider higher rank
   LoRAConfig(
       r=32,  # Higher rank for more adaptation capacity
       lora_alpha=64,
       target_modules=[...],  # Consider adding embedding layers
   )
   ```

#### Recommended Improvements:

1. **Add Embedding Layer LoRA** (for script adaptation):
   ```python
   target_modules: List[str] = field(default_factory=lambda: [
       "q_proj", "k_proj", "v_proj", "o_proj",
       "gate_proj", "up_proj", "down_proj",
       "text_emb",  # Add text embedding adaptation
   ])
   ```

2. **Multi-Language Adapter**:
   ```python
   # Train single adapter for related languages
   class MultiLanguageLoRAConfig(LoRAConfig):
       languages: List[str] = field(default_factory=lambda: ["bn", "or", "as"])
   ```

3. **Evaluation Pipeline**:
   ```python
   # Add evaluation script
   def evaluate_tts(model, test_manifest, language_id):
       # Compute Character Error Rate
       # Compute speaker similarity
       # Compute MOS (if human eval available)
   ```

---

## Recommendations

### For Marathi (Devanagari Script)
- **Difficulty**: Easy
- **Approach**: Same as Bhojpuri/Maithili - use `[hi]` token
- **LoRA Rank**: 16 (standard)
- **Special Handling**: Preserve Marathi-specific ळ (retroflex lateral)

### For Bengali (Bengali Script)
- **Difficulty**: Medium
- **Approach**: Check if `[bn]` exists; if not, may need vocabulary extension or use Hindi with script conversion
- **LoRA Rank**: 24-32 (more capacity for script differences)
- **Special Handling**: Khanda Ta (ৎ), chandrabindu variations

### For Odia (Odia Script)
- **Difficulty**: Medium-Hard
- **Approach**: Similar to Bengali; Odia script is distinct
- **LoRA Rank**: 24-32
- **Special Handling**: Odia-specific vowel signs, wa-variations

### General Best Practices

1. **Start with a small dataset** (500-1000 samples) for initial experiments
2. **Use mixed-precision training** to fit larger batches
3. **Monitor both text and speech losses** - they should decrease together
4. **Save checkpoints frequently** - LoRA files are small
5. **Compare against base model** - ensure improvement, not regression
6. **Test with diverse prompts** - short, long, questions, exclamations

---

## Quick Reference

### Training Command Template
```bash
python scripts/train_lora.py \
    --train-manifest data/${LANG}/train.json \
    --val-manifest data/${LANG}/val.json \
    --language-id ${LANG_CODE} \
    --checkpoint-dir ${CHATTERBOX_CKPT} \
    --output-dir outputs/${LANG}_lora \
    --lora-rank 16 \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4 \
    --mixed-precision
```

### Inference Code Template
```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained_with_lora(
    device="cuda",
    lora_path=f"outputs/{lang}_lora/best_lora"
)

wav = model.generate(
    text="Your text here",
    language_id=lang_code,
    audio_prompt_path="reference.wav"
)
```

---

## Conclusion

The current LoRA implementation for Bhojpuri and Maithili provides a solid foundation that can be extended to other Indic languages. For Devanagari-based languages (Marathi, Sanskrit, Konkani), the existing infrastructure is **sufficient with minimal changes**. For languages with different scripts (Bengali, Odia, Tamil, Telugu), **additional tokenizer work may be required**, but the core LoRA training and inference pipeline remains applicable.

The key strength of this implementation is its **modularity** - language-specific adapters can be developed, tested, and deployed independently without modifying the base model.
