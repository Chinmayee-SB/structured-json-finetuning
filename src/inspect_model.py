import sys
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.utils import load_config

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def print_separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    print_separator("1. MLX Environment & Device Info")
    print(f"MLX Version       : {mx.__version__}")
    print(f"Default Device   : {mx.default_device()}")
    
    start_time = time.time()
    print_separator(f"2. Loading Base Model: {MODEL_ID}")
    print(f"Fetching weights and tokenizer from Hugging Face hub...")
    
    # Load model and tokenizer via mlx_lm
    model, tokenizer = load(MODEL_ID)
    load_duration = time.time() - start_time
    print(f"Successfully loaded model in {load_duration:.2f} seconds.")
    
    # Inspect model configuration & parameters
    print_separator("3. Model Architecture & Parameter Inspection")
    args = model.args
    print(f"Model Type        : {getattr(args, 'model_type', 'qwen2')}")
    print(f"Hidden Size       : {getattr(args, 'hidden_size', 'N/A')}")
    print(f"Num Hidden Layers : {getattr(args, 'num_hidden_layers', 'N/A')}")
    print(f"Num Attention Heads: {getattr(args, 'num_attention_heads', 'N/A')}")
    print(f"Num KV Heads      : {getattr(args, 'num_key_value_heads', 'N/A')}")
    print(f"Vocab Size        : {getattr(args, 'vocab_size', 'N/A')}")
    print(f"Intermediate Size : {getattr(args, 'intermediate_size', 'N/A')}")
    
    # Count parameters
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    params_in_billions = total_params / 1e9
    print(f"\nTotal Parameters  : {total_params:,} ({params_in_billions:.3f} Billion)")
    
    # Measure memory footprint
    peak_mem_bytes = mx.get_peak_memory() if hasattr(mx, 'get_peak_memory') else 0
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)
    print(f"Peak Metal Memory : {peak_mem_mb:.2f} MB")

    # Test generation (Zero-Shot extraction test)
    print_separator("4. Initial Test Generation (Sanity Check)")
    system_prompt = (
        "You are an expert named entity extraction assistant. "
        "Extract entities from text into JSON matching schema: "
        "{\"persons\": [], \"organizations\": [], \"locations\": [], \"misc\": []}."
    )
    user_text = "Apple CEO Tim Cook announced a new facility in Austin, Texas."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Constructed Prompt:\n{prompt}")
    
    print("\nGenerating response from model...")
    gen_start = time.time()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=150, verbose=False)
    gen_duration = time.time() - gen_start
    
    print("\nGenerated Output:")
    print("-" * 40)
    print(response)
    print("-" * 40)
    print(f"Generation finished in {gen_duration:.2f} seconds.")

def tree_flatten(params):
    """Recursively flattens MLX nested parameter dictionaries/lists."""
    flat = []
    if isinstance(params, dict):
        for k, v in params.items():
            flat.extend(tree_flatten(v))
    elif isinstance(params, list):
        for item in params:
            flat.extend(tree_flatten(item))
    elif isinstance(params, mx.array):
        flat.append(("", params))
    return flat

if __name__ == "__main__":
    main()
