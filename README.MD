# Structured JSON Extraction: Parameter-Efficient Fine-Tuning Comparison

An empirical evaluation comparing Parameter-Efficient Fine-Tuning (PEFT) methods—**LoRA**, **QLoRA**, and **DoRA**—for structured named-entity extraction using `Qwen2.5-1.5B-Instruct` on Apple Silicon via `mlx-lm`.

---

## Overview

Small instruction-tuned language models frequently fail to produce structured outputs that conform strictly to target schemas:
- Generating invalid JSON syntax or nested dictionaries when flat lists are required.
- Omitting required keys or inserting hallucinated fields.
- Producing high false-positive rates in open categories (e.g., miscellaneous phrases).

This repository evaluates three fine-tuning methods on identical data, schema definitions, and hardware to measure task quality, schema validity, memory footprint, and inference throughput:

1. **LoRA (Low-Rank Adaptation):** Decomposes weight updates into low-rank matrices ($\Delta W = B \cdot A$) applied to attention and feed-forward projections of the 16-bit base model.
2. **QLoRA (Quantized LoRA):** Applies low-rank adapters over a 4-bit quantized base model.
3. **DoRA (Weight-Decomposed Low-Rank Adaptation):** Decomposes pre-trained weights into magnitude vectors and directional updates ($W = m \frac{V + \Delta V}{\|V + \Delta V\|}$) to align adapter updates more closely with full fine-tuning dynamics.

---

## Task Definition and Target Schema

Given an input text passage, the model must extract all named entities into a JSON object matching the following schema:

```json
{
  "persons": [],
  "organizations": [],
  "locations": [],
  "misc": []
}
```

- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Dataset:** CoNLL-2003 NER benchmark reformatted from sequential BIO tags into entity-grouped JSON pairs.
- **Data Splits:**
  - Training set: 14,041 samples
  - Validation set: 3,250 samples
  - Held-out test set: 3,453 samples (used strictly for evaluation)

---

## Empirical Results

All evaluations were conducted on the complete held-out test split of 3,453 samples using greedy decoding (temperature = 0.0).

### Overall Benchmark Metrics

| Metric | Zero-Shot Baseline | LoRA (16-bit) | QLoRA (4-bit) | DoRA (16-bit) |
| :--- | :---: | :---: | :---: | :---: |
| **JSON Schema Validity Rate** | 59.17% (2,043/3,453) | **99.83%** (3,447/3,453) | **99.91%** (3,450/3,453) | 99.16% (3,424/3,453) |
| **Micro Precision** | 0.3352 | 0.8229 | **0.8233** | 0.8062 |
| **Micro Recall** | 0.4070 | **0.7838** | 0.7820 | 0.7799 |
| **Micro F1** | 0.3676 | **0.8029** | 0.8021 | 0.7928 |
| **Macro F1** | 0.3857 | **0.7780** | 0.7776 | 0.7669 |
| **Base Model Precision** | 16-bit (bfloat16) | 16-bit (bfloat16) | 4-bit (affine) | 16-bit (bfloat16) |
| **Base Weights Size on Disk** | 3.10 GB | 3.10 GB | **0.87 GB** | 3.10 GB |
| **Adapter Checkpoint Size** | — | **42.2 MB** | **42.2 MB** | **42.2 MB** |
| **Inference Throughput** | 0.05 samples/s | 0.91 samples/s | **1.87 samples/s** | 0.11 samples/s |

---

### Category-Level F1 Scores

| Entity Category | Zero-Shot Baseline | LoRA F1 | QLoRA F1 | DoRA F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Persons (`PER`)** | 0.7279 | 0.8700 | **0.8897** | 0.8601 |
| **Organizations (`ORG`)** | 0.2938 | **0.7956** | 0.7684 | 0.7720 |
| **Locations (`LOC`)** | 0.4837 | 0.8217 | 0.8274 | **0.8431** |
| **Miscellaneous (`MISC`)** | 0.0372 | **0.6249** | 0.6248 | 0.5925 |

---

### Detailed Category Precision and Recall

#### LoRA (16-bit Base)
- **Persons:** Precision = 0.8953, Recall = 0.8460, F1 = 0.8700
- **Organizations:** Precision = 0.8328, Recall = 0.7616, F1 = 0.7956
- **Locations:** Precision = 0.8313, Recall = 0.8124, F1 = 0.8217
- **Miscellaneous:** Precision = 0.6245, Recall = 0.6254, F1 = 0.6249

#### QLoRA (4-bit Base)
- **Persons:** Precision = 0.9123, Recall = 0.8683, F1 = 0.8897
- **Organizations:** Precision = 0.8747, Recall = 0.6851, F1 = 0.7684
- **Locations:** Precision = 0.8092, Recall = 0.8465, F1 = 0.8274
- **Miscellaneous:** Precision = 0.5936, Recall = 0.6595, F1 = 0.6248

#### DoRA (16-bit Base)
- **Persons:** Precision = 0.8688, Recall = 0.8516, F1 = 0.8601
- **Organizations:** Precision = 0.7939, Recall = 0.7514, F1 = 0.7720
- **Locations:** Precision = 0.8966, Recall = 0.7956, F1 = 0.8431
- **Miscellaneous:** Precision = 0.5478, Recall = 0.6453, F1 = 0.5925

---

## Key Findings

1. **Schema Compliance:** Fine-tuning on structured chat pairs resolved structural failures, improving JSON schema compliance from 59.17% (zero-shot) to over 99.8% across all PEFT methods.
2. **QLoRA Efficiency and Parity:** 4-bit quantization reduced the base model disk and memory footprint by approximately 72% (from 3.1 GB to 868 MB) and doubled evaluation throughput (1.87 vs. 0.91 samples/s) with negligible impact on overall extraction accuracy (0.8021 vs. 0.8029 Micro F1).
3. **Category Trade-offs:**
   - QLoRA achieved the highest individual F1 on `PER` (0.8897) with 0.9123 precision.
   - DoRA achieved the highest precision and F1 on `LOC` (Precision = 0.8966, F1 = 0.8431).
   - Standard LoRA maintained the most balanced precision-recall trade-off across `ORG` and `MISC`.

---

## Project Structure

```
structured-json-finetuning/
├── configs/
│   ├── lora.yaml              # LoRA hyperparameters (16-bit base)
│   ├── qlora.yaml             # QLoRA hyperparameters (4-bit base)
│   └── dora.yaml              # DoRA hyperparameters (weight decomposition)
├── data/
│   ├── raw/                   # Raw CoNLL-2003 text splits (.gitignore)
│   └── processed/             # Preprocessed JSONL files (train, valid, test)
├── results/
│   ├── baseline_results.json  # Zero-shot evaluation metrics
│   ├── lora_results.json      # LoRA evaluation metrics
│   ├── qlora_results.json     # QLoRA evaluation metrics
│   └── dora_results.json      # DoRA evaluation metrics
├── src/
│   ├── dataset_prep.py        # Dataset downloader and BIO-to-JSON formatter
│   ├── inspect_model.py       # Model architecture and memory footprint utility
│   ├── train.py               # Training execution and telemetry logging
│   ├── evaluate.py            # Schema validation and entity-level F1 evaluation
│   └── run_experiment.py      # End-to-end training and evaluation pipeline
├── problem.md                 # Detailed project specification and hypotheses
└── README.md
```

---

## Reproduction Guide

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm datasets pyyaml huggingface_hub
```

### 2. Dataset Preprocessing

Download and convert the CoNLL-2003 dataset into structured JSONL format:

```bash
python src/dataset_prep.py
```

### 3. Base Model Quantization (for QLoRA)

```bash
python -m mlx_lm convert \
  --hf-path Qwen/Qwen2.5-1.5B-Instruct \
  -q --q-bits 4 \
  --mlx-path models/Qwen2.5-1.5B-Instruct-4bit
```

### 4. Training

```bash
# LoRA (16-bit base)
python -m mlx_lm lora --config configs/lora.yaml

# QLoRA (4-bit base)
python -m mlx_lm lora --config configs/qlora.yaml

# DoRA (16-bit base)
python -m mlx_lm lora --config configs/dora.yaml
```

### 5. Evaluation

```bash
# Evaluate Zero-Shot Baseline
python src/evaluate.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-file results/baseline_results.json

# Evaluate LoRA
python src/evaluate.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/lora \
  --output-file results/lora_results.json

# Evaluate QLoRA
python src/evaluate.py \
  --model models/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path checkpoints/qlora \
  --output-file results/qlora_results.json

# Evaluate DoRA
python src/evaluate.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/dora \
  --output-file results/dora_results.json
```

---

## References

- Tjong Kim Sang, E. F., & De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition.*
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.*
- Liu, S. Y., et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation.*
- MLX Team. (2023). *MLX: An array framework for Apple silicon.*