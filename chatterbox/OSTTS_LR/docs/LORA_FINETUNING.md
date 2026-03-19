# LoRA Fine-tuning for Chatterbox TTS

This guide explains how to fine-tune Chatterbox TTS for Bhojpuri and Maithili languages using LoRA (Low-Rank Adaptation).

## Overview

LoRA allows efficient fine-tuning of the Llama backbone in Chatterbox by training only a small number of additional parameters (~3-8% of the model). This enables:

- **Fast training**: 10x faster than full fine-tuning
- **Low memory**: Train on consumer GPUs (16GB+)
- **Easy deployment**: Small adapter files (~40-80MB)
- **Multi-language**: Switch adapters at runtime

## Installation

```bash
# Install base Chatterbox
pip install -e .

# Install LoRA training dependencies
pip install -r requirements-lora.txt
```

## Quick Start

### 1. Prepare Your Data

Create a manifest file with your training data:

```json
[
  {
    "audio_path": "/path/to/audio1.wav",
    "text": "यह एक उदाहरण वाक्य है।",
    "language_id": "bho"
  },
  {
    "audio_path": "/path/to/audio2.wav",
    "text": "दूसरा वाक्य यहाँ है।",
    "language_id": "bho"
  }
]
```

### 2. Prepare Dataset

```bash
# Create manifest from directory
python scripts/prepare_data.py create-manifest \
    --audio-dir /path/to/audio \
    --text-file /path/to/transcripts.txt \
    --language-id bho \
    --output data/bhojpuri/manifest.json

# Process and split dataset
python scripts/prepare_data.py prepare \
    --manifest data/bhojpuri/manifest.json \
    --output-dir data/bhojpuri/processed \
    --language-id bho \
    --tokenize

python scripts/prepare_data.py split \
    --manifest data/bhojpuri/processed/manifest.json \
    --output-dir data/bhojpuri \
    --train-ratio 0.9 \
    --val-ratio 0.05 \
    --test-ratio 0.05
```

### 3. Train LoRA Adapter

```bash
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
    --mixed-precision
```

### 4. Inference with LoRA

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load model with LoRA adapter
model = ChatterboxMultilingualTTS.from_pretrained_with_lora(
    device="cuda",
    lora_path="outputs/bhojpuri_lora/best_lora"
)

# Generate speech
wav = model.generate(
    text="नमस्ते, कैसे हैं आप?",
    language_id="bho",
    audio_prompt_path="reference.wav"
)
```

### 5. Evaluate

```bash
python scripts/evaluate.py evaluate \
    --checkpoint-dir /path/to/checkpoints \
    --test-manifest data/bhojpuri/test.json \
    --output-dir results/bhojpuri \
    --language-id bho \
    --lora-path outputs/bhojpuri_lora/best_lora \
    --reference-audio reference.wav
```

## Data Requirements

### Minimum Data
- **10 hours** of clean speech for acceptable quality
- **50+ hours** recommended for production quality

### Audio Quality Guidelines
- Sample rate: 24kHz (will be resampled)
- Format: WAV, MP3, or FLAC
- Clean recordings with minimal background noise
- Single speaker per sample
- Duration: 0.5 - 30 seconds per sample

### Text Format
The transcript file should have format: `filename|transcription`

```
audio001|यह एक परीक्षण वाक्य है।
audio002|दूसरा वाक्य यहाँ लिखा है।
```

## Configuration

### LoRA Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 16 | LoRA rank (8, 16, 32) |
| `alpha` | 32 | Scaling factor (usually 2x rank) |
| `dropout` | 0.05 | LoRA dropout |
| `target_modules` | all | Which layers to adapt |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 10 | Training epochs |
| `batch_size` | 4 | Batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `warmup_steps` | 500 | LR warmup steps |
| `grad_clip` | 1.0 | Gradient clipping |

### Using Config Files

```bash
# Use YAML config
python scripts/train_lora.py --config configs/lora_bho.yaml
```

## Advanced Usage

### Custom LoRA Configuration

```python
from chatterbox.models.t3.lora_config import LoRAConfig

# Minimal config (fewer parameters)
config = LoRAConfig.minimal()

# Full config (maximum capacity)
config = LoRAConfig.full()

# Custom config
config = LoRAConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
)

# Apply to model
model.t3.apply_lora(config)
```

### Multi-Language Training

Train separate adapters for each language:

```bash
# Bhojpuri
python scripts/train_lora.py \
    --language-id bho \
    --output-dir outputs/bho_lora \
    ...

# Maithili
python scripts/train_lora.py \
    --language-id mai \
    --output-dir outputs/mai_lora \
    ...
```

### Switching Adapters at Runtime

```python
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Load Bhojpuri adapter
model.load_lora_adapter("outputs/bho_lora/best_lora")
wav_bho = model.generate(text="...", language_id="bho")

# Unload and load Maithili adapter
model.unload_lora_adapter()
model.load_lora_adapter("outputs/mai_lora/best_lora")
wav_mai = model.generate(text="...", language_id="mai")
```

### Merge Adapters for Faster Inference

```python
# Merge LoRA into base model (permanent, ~5% faster inference)
model.merge_lora_adapter()
```

### Resume Training

```python
# Resume from checkpoint
python scripts/train_lora.py \
    --resume outputs/bhojpuri_lora/checkpoint_epoch_5.pt \
    ...
```

## Model Architecture

### What Gets Trained

LoRA adapts these layers in the Llama backbone:

- **Attention**: Q, K, V, O projections
- **MLP**: gate_proj, up_proj, down_proj

### What Stays Frozen

- Voice Encoder (speaker embeddings)
- S3 Tokenizer (audio tokenization)
- S3Gen (waveform synthesis)
- Text embeddings (vocabulary)

### Parameter Efficiency

| LoRA Rank | Additional Params | % of Base |
|-----------|-------------------|-----------|
| 8 | ~20M | 3.8% |
| 16 | ~40M | 7.7% |
| 32 | ~80M | 15.4% |

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
--batch-size 2

# Use gradient checkpointing
--gradient-checkpointing

# Use smaller LoRA rank
--lora-rank 8
```

### Poor Quality Output

1. **Check data quality**: Clean audio is crucial
2. **Increase training**: More epochs or data
3. **Adjust rank**: Try r=32 for more capacity
4. **Lower learning rate**: Try 5e-5

### Training Not Converging

1. Check text preprocessing is correct
2. Verify audio and text are aligned
3. Reduce learning rate
4. Increase warmup steps

## Best Practices

1. **Data Quality > Quantity**: Clean recordings are more valuable than noisy data
2. **Start Small**: Begin with r=16, adjust based on results
3. **Monitor Validation Loss**: Stop if overfitting occurs
4. **Use Reference Audio**: Helps with speaker consistency
5. **Evaluate Regularly**: Run inference during training to check quality

## File Structure

```
chatterbox/
├── configs/
│   ├── lora_bho.yaml          # Bhojpuri config
│   └── lora_mai.yaml          # Maithili config
├── scripts/
│   ├── prepare_data.py        # Data preparation
│   ├── train_lora.py          # Training script
│   └── evaluate.py            # Evaluation script
├── src/chatterbox/
│   ├── models/t3/
│   │   ├── lora_config.py     # LoRA configuration
│   │   └── t3.py              # T3 model with LoRA
│   ├── models/tokenizers/
│   │   └── tokenizer.py       # Language preprocessing
│   └── mtl_tts.py             # TTS with LoRA loading
├── requirements-lora.txt       # LoRA dependencies
└── docs/
    └── LORA_FINETUNING.md     # This guide
```

## Citation

If you use this for research, please cite:

```bibtex
@software{chatterbox_lora,
  title = {Chatterbox TTS with LoRA Fine-tuning},
  year = {2025},
  url = {https://github.com/resemble-ai/chatterbox}
}
```

## Support

For issues and questions:
- GitHub Issues: [chatterbox/issues](https://github.com/resemble-ai/chatterbox/issues)
- Discussions: [chatterbox/discussions](https://github.com/resemble-ai/chatterbox/discussions)
