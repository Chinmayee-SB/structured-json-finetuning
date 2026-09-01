"""End-to-end runner for training and evaluating fine-tuning experiments (LoRA, QLoRA, DoRA)."""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_cmd(cmd: list):
    """Executes a command and streams its output."""
    logger.info("Executing: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        print(line, end="")
    proc.stdout.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning and evaluation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (e.g., configs/lora.yaml)."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Experiment name identifier (e.g., lora, qlora, dora)."
    )
    parser.add_argument(
        "--quantize-base",
        action="store_true",
        help="Quantize base model before training if needed (for QLoRA)."
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Max evaluation samples on test set (default: full test set)."
    )
    args = parser.parse_args()

    exp_name = args.name.lower()
    train_metrics_file = f"results/{exp_name}_train_metrics.json"
    eval_results_file = f"results/{exp_name}_results.json"
    predictions_file = f"results/{exp_name}_predictions.jsonl"

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_path = cfg.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path = cfg.get("adapter_path", f"checkpoints/{exp_name}")

    print("=" * 70)
    print(f"   STARTING EXPERIMENT: {exp_name.upper()} (TRAINING + INFERENCE)")
    print("=" * 70)

    # 1. Training Phase
    train_cmd = [
        sys.executable,
        "src/train.py",
        "--config",
        args.config,
        "--output-metrics",
        train_metrics_file,
    ]
    if args.quantize_base:
        train_cmd.append("--quantize-base")

    train_start = time.time()
    run_cmd(train_cmd)
    train_duration = time.time() - train_start
    print(f"\n Training Phase Complete in {train_duration / 60.0:.2f} minutes.")

    # 2. Evaluation Phase
    print("\n" + "=" * 70)
    print(f"   STARTING EVALUATION: {exp_name.upper()} ON HELD-OUT TEST SET")
    print("=" * 70)

    eval_cmd = [
        sys.executable,
        "src/evaluate.py",
        "--model",
        model_path,
        "--adapter-path",
        adapter_path,
        "--data-file",
        "data/processed/test.jsonl",
        "--output-file",
        eval_results_file,
        "--save-predictions",
        predictions_file,
    ]
    if args.max_eval_samples:
        eval_cmd.extend(["--max-eval-samples", str(args.max_eval_samples)])

    eval_start = time.time()
    run_cmd(eval_cmd)
    eval_duration = time.time() - eval_start
    print(f"\n Evaluation Phase Complete in {eval_duration / 60.0:.2f} minutes.")

    print("\n" + "=" * 70)
    print(f"   EXPERIMENT {exp_name.upper()} FULLY COMPLETED")
    print(f"   Training Metrics  : {train_metrics_file}")
    print(f"   Evaluation Results: {eval_results_file}")
    print(f"   Predictions       : {predictions_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
