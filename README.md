# Direct Speech-to-Speech Translation: Hindi → Maithili

This repository contains the full implementation of a direct speech-to-speech translation (S2ST) system for Hindi to Maithili, along with a cascade baseline, dataset creation pipeline, evaluation scripts, and inference code.

---

## Architecture

<!-- Insert architecture diagram here -->

---

## Repository Structure

```
.
├── models/          # Proposed S2ST model (main approach)
├── baseline/        # Cascade baseline system
├── chatterbox/      # TTS fine-tuning and inference for dataset creation
├── IndicTrans2/     # MT model fine-tuned for baseline
├── evaluation/      # Evaluation scripts
└── inference/       # Inference scripts for the proposed model
```

---

## Proposed Approach (`models/`)

The core of this repository. A direct S2ST system that translates Hindi speech into Maithili speech without an intermediate text representation. The system is composed of three modules:

### 1. Semantic Unit Extractor (`models/semmantic_unit_extractor/`)
Extracts language-agnostic semantic units from source speech using a fine-tuned mHuBERT model with k-means clustering (k=500). These discrete units form the bridge between source and target speech.

### 2. Semantic Decoder (`models/semantic_decoder/`)
A Qwen2.5-3B-based sequence-to-sequence model with a lightweight speech adapter. Trained in two stages:
- **Stage 1** — Adapter alignment on Hindi ASR
- **Stage 2** — Full S2ST fine-tuning with multi-task learning (S2ST + ASR + MT) to prevent catastrophic forgetting

### 3. Acoustic Generation Module (`models/acoustic_generation_module/`)
Converts predicted Maithili semantic units back into waveforms using a two-stage acoustic model:
- **AR model** (`ar/`) — Autoregressive model that generates the first codebook (prosody/speaker identity), conditioned on semantic units and a reference audio clip
- **NAR model** (`nar/`) — Non-autoregressive transformer that predicts the remaining 7 EnCodec codebooks in parallel, given the first codebook

---

## Baseline (`baseline/`)

A cascade S2ST baseline built from:
1. **ASR** — Hindi speech → Hindi text (IndicWav2Vec / Indic Conformer)
2. **MT** — Hindi text → Maithili text (fine-tuned IndicTrans2)
3. **TTS** — Maithili text → Maithili speech (fine-tuned Chatterbox)

The `baseline/` folder contains ASR evaluation scripts, WER/CER comparisons, and inference scripts for the MT and TTS components.

---

## Dataset Creation (`chatterbox/`)

Hindi→Maithili parallel MT data was used to synthesise a paired speech dataset for training the proposed model. Chatterbox was fine-tuned on Maithili (and Hindi) audio and used to generate the training speech.

> For details on Chatterbox and its fine-tuning procedure, refer to the [Chatterbox repository](https://github.com/resemble-ai/chatterbox).

**Key scripts:**
| File | Description |
|---|---|
| `inference_hi.py` | Batch Hindi TTS inference from a JSON dataset |
| `inference_mai.py` | Batch Maithili TTS inference from a JSON dataset |
| `inf_from_hi_txt.py` | Hindi TTS inference from a plain text file |
| `inf_from_mai_txt.py` | Maithili TTS inference from a plain text file |

---

## MT Baseline (`IndicTrans2/`)

IndicTrans2 was fine-tuned on Hindi→Maithili parallel data and used as the translation component in the cascade baseline.

> For details on IndicTrans2, refer to the [IndicTrans2 repository](https://github.com/AI4Bharat/IndicTrans2).

---

## Inference (`inference/`)

End-to-end inference scripts for the proposed direct S2ST model.

| File | Description |
|---|---|
| `infer_from_audio_to_sem.py` | Runs the full pipeline: Hindi audio → semantic units → Maithili semantic units |
| `batch_infer_nar.py` | Batch inference through the NAR acoustic generation module |

**Usage:**
1. Set the required paths (checkpoint, input audio, output directory) in the scripts.
2. Run `infer_from_audio_to_sem.py` to obtain predicted Maithili semantic units.
3. Run `batch_infer_nar.py` to synthesise the final Maithili waveform.

---

## Evaluation (`evaluation/`)

| File | Description |
|---|---|
| `eval_metrics.py` | Computes evaluation metrics (e.g. ASR-BLEU, chrF) on generated speech |
| `gen_test_translation_json.py` | Generates a JSON of test translations for evaluation |

---

## Setup

Ensure you configure the path placeholders in each script (marked as `<path to ...>`) before running.

Core dependencies:
- PyTorch
- Transformers (HuggingFace)
- torchaudio
- EnCodec
- bitsandbytes (for 8-bit optimizer)
- peft (for LoRA)

---

## Citation

If you use this work, please cite accordingly.
