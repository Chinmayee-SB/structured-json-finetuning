# Structured JSON Extraction — Fine-Tuning Method Comparison

## Problem Statement

Instruction-tuned LLMs are commonly used for structured information extraction (text → JSON matching a fixed schema), but small base models are often inconsistent: they may return malformed JSON, drop required fields, hallucinate values not present in the source text, or use inconsistent key naming.

This project fine-tunes a small language model to reliably perform named-entity extraction into a fixed JSON schema, and compares three fine-tuning strategies — **full supervised fine-tuning (SFT)**, **LoRA**, and **QLoRA** — head-to-head on the same task, same data, and same hardware (Apple M4 Air, via `mlx-lm`).

The goal is twofold:
1. **Learn the mechanics** of each fine-tuning approach end-to-end, locally, from data prep through training to evaluation.
2. **Produce a portfolio-quality comparison** with real measured numbers (not just repeated conventional wisdom) on task quality, memory footprint, training time, and checkpoint size.

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
- **Model family reused across all three runs** — only the fine-tuning method and base-model precision change.

## Dataset

- **Source:** CoNLL-2003 named entity recognition dataset (via Hugging Face `datasets`), a well-established, small, public NER benchmark with four entity types: PER, ORG, LOC, MISC.
- **Reformatting:** Original token-level BIO-tagged sequences are converted into (text, target JSON) pairs — each sentence's tagged entities are grouped by type into the schema above.
- **Splits:** Standard CoNLL-2003 train/validation/test splits are preserved; the test split is held out and untouched until final evaluation across all three trained models.
- **Format for `mlx-lm`:** `train.jsonl` / `valid.jsonl` in chat format (`messages` key), with the extraction instruction and schema as the prompt and the target JSON as the completion. Loss is masked to the completion only (`--mask-prompt`).

## Hypothesis

Before running the experiments, the expected ranking:

| | Task quality (F1 / validity) | Peak memory | Training time | Checkpoint size | Forgetting risk |
|---|---|---|---|---|---|
| **Full SFT** | Highest | Highest | Slowest | Largest (full model) | Highest |
| **LoRA** | Close to full SFT | Low | Fast | Tiny (adapter only) | Low |
| **QLoRA** | Close to LoRA | Lowest | Similar to LoRA, possibly slightly slower per-step (dequant overhead) | Tiny (adapter only) | Low |

The interesting empirical questions this project should answer on real hardware:
- How much task-quality gap (if any) actually exists between full SFT and LoRA/QLoRA on this task size?
- How much memory does QLoRA actually save over LoRA at this model scale, in practice?
- Is there a measurable per-step slowdown from QLoRA's 4-bit dequantization on the same hardware?

## Evaluation Metrics

Computed identically across all three fine-tuned models plus the untrained base model (as a zero-shot baseline), on the held-out CoNLL-2003 test split:
- **JSON schema-validity rate** — does the output parse and match the expected schema/types?
- **Entity-level F1** — per entity type (PER, ORG, LOC, MISC) and overall, compared against gold labels.
- **Peak memory usage** during training.
- **Wall-clock training time**.
- **Checkpoint / adapter size on disk**.

## Roadmap

1. **Data preparation**
   - Load CoNLL-2003 via `datasets`.
   - Write reformatting script: BIO tags → grouped-entity JSON.
   - Build chat-format `train.jsonl` / `valid.jsonl` with masked prompts.
   - Hold out test split untouched.

2. **Baseline (zero-shot)**
   - Run the untrained base model on the test set with the extraction prompt.
   - Record JSON validity rate + entity F1 as the "before" reference point.

3. **Run 1 — Full SFT**
   - `mlx_lm.lora --fine-tune-type full`
   - Log memory, time, loss curve.

4. **Run 2 — LoRA**
   - `mlx_lm.lora --fine-tune-type lora` on bf16 base.
   - Same rank/layers config to be reused in Run 3.

5. **Run 3 — QLoRA**
   - Quantize base model to 4-bit (`mlx_lm.convert -q`).
   - `mlx_lm.lora --fine-tune-type lora` on the quantized base, same rank/layers as Run 2.

6. **Evaluation**
   - Run shared eval script across baseline + all three fine-tuned models on the held-out test set.
   - Collect the metrics table above.

7. **Write-up**
   - Compare hypothesis vs. actual results.
   - Document any surprises (e.g., unexpected forgetting, unexpected memory numbers, unexpected quality parity or gap).
   - Package as a portfolio piece with the comparison table, loss curves, and example before/after outputs.

## Out of Scope (for this project)

- Extending to other extraction schemas or domains.
- Deploying the fine-tuned model as a served endpoint.
- Any connection to other ongoing projects — this is a standalone fine-tuning learning exercise.
