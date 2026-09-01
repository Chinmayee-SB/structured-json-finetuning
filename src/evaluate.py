"""Evaluation harness for Structured JSON Named Entity Extraction.

Evaluates base, SFT, LoRA, and QLoRA models on held-out JSONL test splits.
Calculates JSON schema validity rate, category-level and overall entity Precision/Recall/F1,
and records generation throughput.
"""

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlx_lm import generate, load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

EXPECTED_SCHEMA_KEYS = ["persons", "organizations", "locations", "misc"]


def normalize_entity(ent: str) -> str:
    """Normalizes entity text for evaluation (lowercased, stripped, collapsed whitespace)."""
    if not isinstance(ent, str):
        ent = str(ent)
    return re.sub(r"\s+", " ", ent).strip().lower()


def parse_and_validate_json(raw_text: str) -> Tuple[bool, Dict[str, List[str]], str]:
    """Attempts to parse model output into the expected JSON schema.

    Returns:
        (is_valid_schema, parsed_entities_dict, status_description)
    """
    cleaned = raw_text.strip()

    # Strip markdown code blocks if present
    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if markdown_match:
        cleaned = markdown_match.group(1).strip()
    else:
        # Try to find the outermost JSON object in the text
        brace_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except Exception as e:
        return False, {k: [] for k in EXPECTED_SCHEMA_KEYS}, f"JSONDecodeError: {str(e)}"

    if not isinstance(data, dict):
        return False, {k: [] for k in EXPECTED_SCHEMA_KEYS}, "Parsed JSON is not an object"

    # Strict schema check:
    # 1. Must contain all 4 keys
    # 2. Each value must be a list
    # 3. Each element in the list must be a string (not nested dicts/objects)
    valid_schema = True
    validated_entities: Dict[str, List[str]] = {}

    for k in EXPECTED_SCHEMA_KEYS:
        if k not in data:
            valid_schema = False
            validated_entities[k] = []
        elif not isinstance(data[k], list):
            valid_schema = False
            validated_entities[k] = []
        else:
            extracted_items = []
            for item in data[k]:
                if isinstance(item, str):
                    extracted_items.append(item)
                elif isinstance(item, dict):
                    # Malformed schema (e.g. {"name": "..."})
                    valid_schema = False
                    val = item.get("name") or item.get("text") or item.get("entity") or str(item)
                    extracted_items.append(str(val))
                else:
                    valid_schema = False
                    extracted_items.append(str(item))
            validated_entities[k] = extracted_items

    # Check for unexpected extra top-level keys
    extra_keys = set(data.keys()) - set(EXPECTED_SCHEMA_KEYS)
    if extra_keys:
        valid_schema = False

    status = "Valid Schema" if valid_schema else "Invalid Schema Structure"
    return valid_schema, validated_entities, status


def compute_entity_metrics(
    gold_entities_list: List[Dict[str, List[str]]],
    pred_entities_list: List[Dict[str, List[str]]]
) -> Dict[str, Any]:
    """Computes Precision, Recall, and F1 per category and overall micro/macro metrics."""
    category_counts = {
        cat: {"tp": 0, "fp": 0, "fn": 0} for cat in EXPECTED_SCHEMA_KEYS
    }

    for gold, pred in zip(gold_entities_list, pred_entities_list):
        for cat in EXPECTED_SCHEMA_KEYS:
            g_list = [normalize_entity(x) for x in gold.get(cat, [])]
            p_list = [normalize_entity(x) for x in pred.get(cat, [])]

            g_counter = Counter(g_list)
            p_counter = Counter(p_list)

            # True Positives: multiset intersection
            tp = sum((g_counter & p_counter).values())
            fp = sum(p_counter.values()) - tp
            fn = sum(g_counter.values()) - tp

            category_counts[cat]["tp"] += tp
            category_counts[cat]["fp"] += fp
            category_counts[cat]["fn"] += fn

    metrics_by_category = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    precision_sum, recall_sum, f1_sum = 0.0, 0.0, 0.0

    for cat in EXPECTED_SCHEMA_KEYS:
        tp = category_counts[cat]["tp"]
        fp = category_counts[cat]["fp"]
        fn = category_counts[cat]["fn"]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if (tp + fn) == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if (tp + fp) == 0 else 0.0)
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics_by_category[cat] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_count": tp + fn,
            "pred_count": tp + fp,
        }

        precision_sum += prec
        recall_sum += rec
        f1_sum += f1

    # Overall Micro Metrics
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # Macro Metrics
    num_cats = len(EXPECTED_SCHEMA_KEYS)
    macro_prec = precision_sum / num_cats
    macro_rec = recall_sum / num_cats
    macro_f1 = f1_sum / num_cats

    return {
        "per_category": metrics_by_category,
        "micro": {
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4),
            "f1": round(micro_f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "macro": {
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "f1": round(macro_f1, 4),
        }
    }


def load_test_dataset(test_file: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads test JSONL dataset formatted with 'messages' list."""
    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            samples.append(item)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def evaluate_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    test_file: Path = Path("data/processed/test.jsonl"),
    output_results_file: Optional[Path] = None,
    predictions_file: Optional[Path] = None,
    max_samples: Optional[int] = None,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """Runs evaluation on the specified model and returns evaluation metrics."""
    logger.info("Loading model: %s (adapter: %s)...", model_path, adapter_path)
    load_start = time.time()

    if adapter_path:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_path)

    logger.info("Model loaded in %.2f seconds.", time.time() - load_start)

    test_samples = load_test_dataset(test_file, max_samples=max_samples)
    logger.info("Loaded %d test samples from %s.", len(test_samples), test_file)

    gold_entities_list: List[Dict[str, List[str]]] = []
    pred_entities_list: List[Dict[str, List[str]]] = []
    prediction_records: List[Dict[str, Any]] = []

    valid_json_count = 0
    total_samples = len(test_samples)
    start_eval_time = time.time()

    for idx, sample in enumerate(test_samples, 1):
        messages = sample["messages"]
        prompt_messages = [m for m in messages if m["role"] in ("system", "user")]

        gold_assistant_content = next(
            m["content"] for m in messages if m["role"] == "assistant"
        )
        try:
            gold_entities = json.loads(gold_assistant_content)
        except Exception:
            gold_entities = {k: [] for k in EXPECTED_SCHEMA_KEYS}

        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        gen_start = time.time()
        output_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        gen_duration = time.time() - gen_start

        is_valid_json, pred_entities, parse_status = parse_and_validate_json(output_text)
        if is_valid_json:
            valid_json_count += 1

        gold_entities_list.append(gold_entities)
        pred_entities_list.append(pred_entities)

        prediction_records.append({
            "index": idx,
            "input_text": next((m["content"] for m in prompt_messages if m["role"] == "user"), ""),
            "raw_output": output_text,
            "is_valid_json": is_valid_json,
            "parse_status": parse_status,
            "gold_entities": gold_entities,
            "pred_entities": pred_entities,
            "gen_time_sec": round(gen_duration, 4),
        })

        if idx % 100 == 0 or idx == total_samples:
            elapsed = time.time() - start_eval_time
            rate = idx / elapsed
            logger.info(
                "Progress: %d/%d (%.1f%%) | Valid JSON: %d (%.1f%%) | %.2f samples/sec",
                idx,
                total_samples,
                100.0 * idx / total_samples,
                valid_json_count,
                100.0 * valid_json_count / idx,
                rate,
            )

    total_eval_time = time.time() - start_eval_time
    validity_rate = valid_json_count / total_samples if total_samples > 0 else 0.0

    entity_metrics = compute_entity_metrics(gold_entities_list, pred_entities_list)

    summary_results = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "test_samples_evaluated": total_samples,
        "json_schema_validity_rate": round(validity_rate, 4),
        "valid_json_count": valid_json_count,
        "invalid_json_count": total_samples - valid_json_count,
        "total_eval_time_sec": round(total_eval_time, 2),
        "samples_per_second": round(total_samples / total_eval_time, 2) if total_eval_time > 0 else 0.0,
        "micro_metrics": entity_metrics["micro"],
        "macro_metrics": entity_metrics["macro"],
        "per_category_metrics": entity_metrics["per_category"],
    }

    if output_results_file:
        output_results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_results_file, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)
        logger.info("Saved evaluation summary to %s", output_results_file)

    if predictions_file:
        predictions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_file, "w", encoding="utf-8") as f:
            for rec in prediction_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Saved detailed prediction records to %s", predictions_file)

    return summary_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER structured JSON extraction.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model ID or path."
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights (if applicable)."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to test JSONL file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/baseline_results.json",
        help="Path to save evaluation summary metrics JSON."
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default="results/baseline_predictions.jsonl",
        help="Path to save detailed predictions JSONL."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of test samples to evaluate."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens for generation."
    )
    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        adapter_path=args.adapter_path,
        test_file=Path(args.data_file),
        output_results_file=Path(args.output_file) if args.output_file else None,
        predictions_file=Path(args.save_predictions) if args.save_predictions else None,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
    )

    print("\n" + "=" * 60)
    print("                EVALUATION SUMMARY RESULTS")
    print("=" * 60)
    print(f"Model                  : {results['model_path']}")
    print(f"Adapter                : {results['adapter_path']}")
    print(f"Samples Evaluated      : {results['test_samples_evaluated']}")
    print(f"JSON Validity Rate     : {results['json_schema_validity_rate'] * 100:.2f}% ({results['valid_json_count']}/{results['test_samples_evaluated']})")
    print(f"Overall Micro F1       : {results['micro_metrics']['f1']:.4f} (Prec: {results['micro_metrics']['precision']:.4f}, Rec: {results['micro_metrics']['recall']:.4f})")
    print(f"Overall Macro F1       : {results['macro_metrics']['f1']:.4f}")
    print("-" * 60)
    print("Category Breakdown:")
    for cat, cat_m in results["per_category_metrics"].items():
        print(f"  - {cat:14s}: F1={cat_m['f1']:.4f} | Prec={cat_m['precision']:.4f} | Rec={cat_m['recall']:.4f} (Gold: {cat_m['gold_count']}, Pred: {cat_m['pred_count']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
