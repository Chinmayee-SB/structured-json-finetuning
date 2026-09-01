# Structured JSON Extraction — Parameter-Efficient Fine-Tuning (PEFT) Comparison

## Problem Statement

Instruction-tuned LLMs are commonly used for structured information extraction (text → JSON matching a fixed schema), but small base models are often inconsistent: they may return malformed JSON, drop required fields, hallucinate values not present in the source text, or use inconsistent key naming.

This project fine-tunes a small language model to reliably perform named-entity extraction into a fixed JSON schema, and compares three modern parameter-efficient fine-tuning (PEFT) strategies — **LoRA**, **QLoRA**, and **DoRA (Weight-Decomposed Low-Rank Adaptation)** — head-to-head on the same task, same data, and same hardware (Apple Silicon via `mlx-lm`).

The goal is twofold:
1. **Learn the mechanics** of each PEFT approach end-to-end, locally, from data prep through training to evaluation.
2. **Produce a portfolio-quality comparison** with real measured numbers on task quality (entity F1, schema validity), memory footprint, training time, and adapter checkpoint size.

## Task Definition

- **Input:** A sentence or short passage of text.
- **Output:** A JSON object with a fixed schema:
```json
{
  "persons": [],
  "organizations": [],
  "locations": [],
  "misc": []
}
```
- **Base model:** Qwen2.5-1.5B-Instruct
- **Model family reused across all three runs** — only the fine-tuning adapter mechanism and base-model precision change.

## Dataset

- **Source:** CoNLL-2003 named entity recognition dataset, a standard NER benchmark with four entity types: PER, ORG, LOC, MISC.
- **Reformatting:** Original token-level BIO-tagged sequences are converted into (text, target JSON) pairs — each sentence's tagged entities are grouped by type into the schema above.
- **Splits:** Standard CoNLL-2003 train/validation/test splits are preserved; the test split (3,453 samples) is held out untouched until evaluation.
- **Format for `mlx-lm`:** `train.jsonl` / `valid.jsonl` in chat format (`messages` key), with the extraction instruction and schema as the prompt and the target JSON as the completion. Loss is masked to the completion only (`--mask-prompt`).

## Hypothesis & Method Comparison

| Method | Base Precision | Adapter Mechanism | Expected Task Quality | Expected Peak Memory | Expected Training Time | Checkpoint Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA** | 16-bit (bfloat16) | Standard low-rank update: $\Delta W = B \cdot A$ | High (strong baseline) | Moderate (~6–9 GB) | Fast baseline (~0.7–1.0 it/s) | Tiny (~42 MB) |
| **QLoRA** | 4-bit (Quantized) | Low-rank update on 4-bit base weights | Close to LoRA | Lowest (~3–5 GB) | Similar to LoRA (slight dequant overhead) | Tiny (~42 MB) |
| **DoRA** | 16-bit (bfloat16) | Decomposes weights into magnitude and direction: $W = m \frac{V + \Delta V}{\|V + \Delta V\|}$ | Highest PEFT quality (closest to Full SFT) | Moderate (~7–10 GB) | Slightly slower per-step (norm computations) | Tiny (~42 MB) |

### Key Empirical Questions:
- Does **DoRA's** directional weight decomposition provide a measurable F1 boost over standard **LoRA** on structured extraction?
- How much memory does **QLoRA** save over LoRA at the 1.5B model scale on Apple Silicon Unified Memory?
- Does 4-bit quantization in QLoRA cause any noticeable degradation in strict JSON schema compliance or entity recall?

## Evaluation Metrics

Computed identically across all fine-tuned models plus the untrained base model (zero-shot baseline) on the held-out CoNLL-2003 test split (3,453 samples):
- **JSON schema-validity rate** — does the output parse as valid JSON and strictly match the expected keys and string array types?
- **Entity-level F1, Precision, and Recall** — per entity type (`persons`, `organizations`, `locations`, `misc`) and overall micro/macro metrics.
- **Peak memory usage** during training (Metal memory).
- **Wall-clock training time** and iteration throughput.
- **Checkpoint / adapter size on disk**.

## Roadmap

1. **Data preparation** ✅
   - Reformat CoNLL-2003 BIO tags → grouped-entity JSON.
   - Build `train.jsonl` (14,041 samples), `valid.jsonl` (3,250 samples), `test.jsonl` (3,453 samples).
2. **Baseline (Zero-Shot)** ✅
   - Evaluated untrained `Qwen2.5-1.5B-Instruct` on held-out test set (`59.17%` validity, `0.3676` Micro F1).
3. **Run 1 — LoRA** ✅
   - Fine-tuned rank 16 adapters on 16-bit base (`99.83%` validity, `0.8029` Micro F1, 42.2 MB adapter).
4. **Run 2 — QLoRA** ⏳
   - Quantize base model to 4-bit (`mlx_lm.convert -q`) and fine-tune LoRA adapters.
5. **Run 3 — DoRA** ⏳
   - `mlx_lm.lora --fine-tune-type dora` on 16-bit base with rank 16 magnitude/direction decomposition.
6. **Comparative Evaluation & Benchmark** ⏳
   - Collect and synthesize head-to-head metrics table across Baseline, LoRA, QLoRA, and DoRA.
7. **Write-up** ⏳
   - Final report with loss curves, trade-off analysis, and key takeaways for local PEFT deployment.
