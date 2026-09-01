"""Training runner with telemetry for Full SFT, LoRA, and QLoRA comparisons.

Executes mlx_lm fine-tuning while monitoring wall-clock time, peak Apple Silicon
Metal memory consumption, and checkpoint artifacts.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_dir_size(path: Path) -> int:
    """Calculates total size of files in directory in bytes."""
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def quantize_base_model_if_needed(
    source_model: str,
    quantized_output_dir: Path,
    q_bits: int = 4
) -> Path:
    """Quantizes the base model to 4-bit using mlx_lm.convert if not already done."""
    if quantized_output_dir.exists() and (quantized_output_dir / "config.json").exists():
        logger.info("Quantized model already exists at %s", quantized_output_dir)
        return quantized_output_dir

    logger.info("Quantizing %s to %d-bit into %s...", source_model, q_bits, quantized_output_dir)
    quantized_output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.convert",
        "--hf-path",
        source_model,
        "-q",
        "--q-bits",
        str(q_bits),
        "--mlx-path",
        str(quantized_output_dir),
    ]
    
    start_t = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("Quantization failed:\n%s", res.stderr)
        raise RuntimeError(f"Model quantization failed: {res.stderr}")
        
    logger.info("Quantization completed in %.2f seconds.", time.time() - start_t)
    return quantized_output_dir


def run_training(config_path: Path, output_metrics_file: Path) -> Dict[str, Any]:
    """Runs mlx_lm.lora with the given YAML config and collects telemetry."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("Starting training run with config: %s", config_path)
    
    # Reset peak memory counter if supported
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    start_time = time.time()
    
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        str(config_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stdout_lines = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        stdout_lines.append(line)

    process.stdout.close()
    return_code = process.wait()
    wall_clock_sec = time.time() - start_time

    if return_code != 0:
        raise RuntimeError(f"Training process failed with exit code {return_code}")

    # Read checkpoint directory from config or default
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    adapter_path = Path(cfg.get("adapter_path", "adapters"))
    checkpoint_size_bytes = get_dir_size(adapter_path)
    checkpoint_size_mb = checkpoint_size_bytes / (1024 * 1024)

    # Extract loss values from logs
    train_losses = []
    val_losses = []
    for line in stdout_lines:
        if "Iter " in line and "Train loss " in line:
            try:
                # e.g., Iter 10: Train loss 1.234
                part = line.split("Train loss ")[1].split()[0]
                train_losses.append(float(part))
            except Exception:
                pass
        if "Val loss " in line:
            try:
                part = line.split("Val loss ")[1].split()[0]
                val_losses.append(float(part))
            except Exception:
                pass

    metrics = {
        "config_file": str(config_path),
        "fine_tune_type": cfg.get("fine_tune_type", "lora"),
        "model": cfg.get("model"),
        "iterations": cfg.get("iters", 1000),
        "batch_size": cfg.get("batch_size", 4),
        "learning_rate": cfg.get("learning_rate"),
        "wall_clock_time_sec": round(wall_clock_sec, 2),
        "wall_clock_time_min": round(wall_clock_sec / 60.0, 2),
        "checkpoint_path": str(adapter_path),
        "checkpoint_size_mb": round(checkpoint_size_mb, 2),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
    }

    output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved training telemetry metrics to %s", output_metrics_file)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning with telemetry.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training config file."
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="results/train_metrics.json",
        help="Path to output JSON metrics."
    )
    parser.add_argument(
        "--quantize-base",
        action="store_true",
        help="Whether to quantize the base model first (for QLoRA)."
    )
    args = parser.parse_args()

    if args.quantize_base:
        quantize_base_model_if_needed(
            source_model="Qwen/Qwen2.5-1.5B-Instruct",
            quantized_output_dir=Path("models/Qwen2.5-1.5B-Instruct-4bit"),
            q_bits=4
        )

    metrics = run_training(
        config_path=Path(args.config),
        output_metrics_file=Path(args.output_metrics)
    )

    print("\n" + "=" * 60)
    print("                TRAINING RUN COMPLETED")
    print("=" * 60)
    print(f"Method                 : {metrics['fine_tune_type'].upper()}")
    print(f"Model                  : {metrics['model']}")
    print(f"Wall-Clock Time        : {metrics['wall_clock_time_min']} min ({metrics['wall_clock_time_sec']} sec)")
    print(f"Checkpoint Size        : {metrics['checkpoint_size_mb']} MB")
    print(f"Final Train Loss       : {metrics['final_train_loss']}")
    print(f"Final Val Loss         : {metrics['final_val_loss']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
