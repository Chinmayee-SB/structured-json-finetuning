# Structured JSON Extraction — PEFT Method Comparison

A head-to-head empirical benchmark comparing modern Parameter-Efficient Fine-Tuning (**LoRA**, **QLoRA**, and **DoRA**) methods on Apple Silicon using `mlx-lm` and `Qwen2.5-1.5B-Instruct` for structured Named Entity Extraction.

---

## 📌 Project Overview

Small instruction-tuned language models frequently struggle with strict JSON schema adherence:
- Producing malformed JSON or nested structures when flat arrays are expected.
- Dropping required keys or hallucinating extra properties.
- Drastic false positive rates in open categories (e.g. miscellaneous phrases).

This project benchmarks three parameter-efficient fine-tuning (PEFT) methods on the exact same dataset (CoNLL-2003 reformatted to JSON), evaluation harness, and hardware:
1. **LoRA (Low-Rank Adaptation):** Low-rank updates on 16-bit base weights ($\Delta W = B \cdot A$).
2. **QLoRA (Quantized LoRA):** Low-rank updates on a 4-bit quantized base model.
3. **DoRA (Weight-Decomposed LoRA):** Decomposing weights into directional updates and learnable magnitudes ($W = m \frac{V + \Delta V}{\|V + \Delta V\|}$).

---

## 🎯 Target Schema

Given any input text passage, the model must output valid JSON strictly conforming to:

```json
{
  "persons": [],
  "organizations": [],
  "locations": [],
  "misc": []
}
```

---

## 📊 Current Benchmark Results (Held-Out Test Set: 3,453 Samples)

| Method | JSON Schema Validity | Micro F1 | Macro F1 | Peak Metal Memory | Adapter Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot Baseline** (`Qwen2.5-1.5B`) | `59.17%` | `0.3676` | `0.3857` | — | — |
| **LoRA** (Rank 16, 16-bit Base) | **`99.83%`** | **`0.8029`** | **`0.7780`** | `9.10 GB` | `42.2 MB` |
| **QLoRA** (Rank 16, 4-bit Base) | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* |
| **DoRA** (Rank 16, 16-bit Base) | *Planned* | *Planned* | *Planned* | *Planned* | *Planned* |

### Entity Category F1 Breakdown

| Entity Category | Zero-Shot F1 | LoRA F1 (Δ Gain) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Persons (`PER`)** | `0.7279` | **`0.8700`** *(+14.2%)* | `0.8953` | `0.8460` |
| **Organizations (`ORG`)** | `0.2938` | **`0.7956`** *(+50.2%)* | `0.8328` | `0.7616` |
| **Locations (`LOC`)** | `0.4837` | **`0.8217`** *(+33.8%)* | `0.8313` | `0.8124` |
| **Miscellaneous (`MISC`)** | `0.0372` | **`0.6249`** *(+58.8%)* | `0.6245` | `0.6254` |

---

## 📂 Project Structure

```
├── configs/
│   ├── lora.yaml              # 16-bit LoRA training configuration
│   ├── qlora.yaml             # 4-bit QLoRA training configuration
│   └── dora.yaml              # DoRA magnitude/direction decomposed configuration
├── data/
│   ├── raw/                   # Downloaded raw CoNLL-2003 text splits (.gitignore)
│   └── processed/             # Formatted chat JSONL splits (train, valid, test)
├── results/
│   ├── baseline_results.json  # Zero-shot evaluation metrics & summary
│   ├── lora_results.json      # LoRA evaluation metrics & summary
│   └── lora_train_metrics.json# LoRA training loss & telemetry
├── src/
│   ├── dataset_prep.py        # Dataset downloader and BIO -> JSON reformatter
│   ├── inspect_model.py       # Model architecture and memory footprint inspector
│   ├── train.py               # Telemetry-instrumented training runner
│   ├── evaluate.py            # Strict schema validator & NER F1 evaluator
│   └── run_experiment.py      # End-to-end training + inference orchestrator
├── problem.md                 # Full project specification & hypotheses
└── README.MD
```

---

## 🛠️ Getting Started

### 1. Environment Setup

Ensure you have Python 3.10+ on Apple Silicon:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm datasets pyyaml huggingface_hub
```

### 2. Prepare Data

Downloads CoNLL-2003 and generates `train.jsonl` (14,041), `valid.jsonl` (3,250), and `test.jsonl` (3,453):

```bash
python src/dataset_prep.py
```

### 3. Run Fine-Tuning & Evaluation

Run end-to-end training and held-out evaluation with a single command:

```bash
# Run LoRA
python src/run_experiment.py --config configs/lora.yaml --name lora

# Run QLoRA (auto-quantizes base model to 4-bit)
python src/run_experiment.py --config configs/qlora.yaml --name qlora --quantize-base

# Run DoRA
python src/run_experiment.py --config configs/dora.yaml --name dora
```

### 4. Evaluate Checkpoints Standalone

```bash
python src/evaluate.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/lora \
  --data-file data/processed/test.jsonl \
  --output-file results/lora_results.json
```

---

## 📖 License & Details

See [`problem.md`](problem.md) for full project specifications, hypotheses, and roadmap.