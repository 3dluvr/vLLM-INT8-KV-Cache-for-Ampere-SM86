#!/usr/bin/env python3
"""
Calibrate per-layer K/V scales for INT8 KV cache.

Usage:
1. Start server in recording mode:
   VLLM_KV_SCALES_RECORD=1 vllm serve Qwen/Qwen3.5-27B-GPTQ-Int4 \
       --tensor-parallel-size 2 \
       --kv-cache-dtype int8 \
       --calculate-kv-scales \
       --max-model-len 8192

2. Run this script:
   python scripts/calibrate_per_layer_scales.py

3. Stop server (Ctrl+C) - scales auto-save to /tmp/vllm_kv_scales.json

4. Restart server with pre-computed scales:
   VLLM_KV_SCALES_FILE=/tmp/vllm_kv_scales.json vllm serve ...
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict, timeout: int = 180) -> dict:
    """Post JSON to URL and return response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_calibration_corpus(path: Path) -> str:
    """Load calibration text from file."""
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def generate_calibration_prompts(corpus: str) -> list:
    """Generate diverse prompts for scale calibration."""
    prompts = []

    # Short prompts (various domains)
    prompts.extend([
        "What is 2+2? Answer briefly.",
        "Write a haiku about programming.",
        "List 5 programming languages.",
        "Explain TCP/IP in one sentence.",
        "Define machine learning.",
    ])

    # Medium prompts (technical)
    prompts.extend([
        "Explain the difference between TCP and UDP in networking. Include examples of when each is used.",
        "Write a Python function to check if a number is prime, with detailed comments explaining each step.",
        "Describe the key principles of object-oriented programming and give examples in Python.",
        "Explain how attention mechanisms work in transformer neural networks.",
        "What are the advantages and disadvantages of different quantization methods for LLMs?",
    ])

    # Long prompts from corpus (if available)
    if corpus:
        chunk_sizes = [2000, 4000, 8000, 16000]
        for size in chunk_sizes:
            for start in range(0, min(len(corpus), size * 3), size):
                chunk = corpus[start : start + size].strip()
                if chunk:
                    prompts.append(
                        f"Summarize the following text in 3 bullet points:\n\n{chunk}"
                    )

    # Code generation
    prompts.extend([
        "Write a complete Python implementation of a binary search tree with insert, delete, and search operations.",
        "Implement a simple HTTP server in Python that handles GET and POST requests.",
        "Write a recursive function to compute Fibonacci numbers with memoization.",
    ])

    # Multilingual (tests different activation patterns)
    prompts.extend([
        "Traduisez en français: The quick brown fox jumps over the lazy dog.",
        "Übersetzen Sie ins Deutsche: Machine learning is transforming many industries.",
        "Vertaal naar het Nederlands: Artificial intelligence is changing the world.",
    ])

    return prompts


def run_calibration(
    base_url: str,
    model: str,
    prompts: list,
    max_tokens: int = 64,
    verbose: bool = True,
) -> dict:
    """Run calibration prompts and collect statistics."""
    url = f"{base_url}/v1/completions"
    stats = {
        "prompts_processed": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "errors": 0,
    }

    if verbose:
        print(f"Running {len(prompts)} calibration prompts...")
        print("-" * 60)

    for i, prompt in enumerate(prompts, 1):
        try:
            start = time.time()
            resp = post_json(
                url,
                {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,  # Some randomness for diversity
                },
            )
            elapsed = time.time() - start

            usage = resp.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            stats["prompts_processed"] += 1
            stats["total_prompt_tokens"] += prompt_tokens
            stats["total_completion_tokens"] += completion_tokens

            if verbose:
                tok_per_s = completion_tokens / elapsed if elapsed > 0 else 0
                print(f"[{i:3}/{len(prompts)}] {prompt_tokens:5} prompt, {completion_tokens:3} gen, {tok_per_s:.1f} tok/s")

        except Exception as e:
            stats["errors"] += 1
            if verbose:
                print(f"[{i:3}/{len(prompts)}] ERROR: {e}")

        # Small delay to prevent overwhelming
        time.sleep(0.1)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://localhost:8081", help="vLLM server URL")
    parser.add_argument("--model", default="gemma-4-31b", help="Model name")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "dutch_parliament_text.txt",
        help="Calibration corpus file",
    )
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per response")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("Per-Layer K/V Scale Calibration")
    print("=" * 60)
    print(f"Server: {args.url}")
    print(f"Model: {args.model}")
    print()

    # Check server is running
    try:
        resp = urllib.request.urlopen(f"{args.url}/v1/models", timeout=5)
        print("Server is running")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        print("\nMake sure to start server with:")
        print("  VLLM_KV_SCALES_RECORD=1 vllm serve ... --calculate-kv-scales")
        return 1

    # Load corpus
    corpus = load_calibration_corpus(args.corpus)
    print(f"Loaded corpus: {len(corpus)} characters")

    # Generate prompts
    prompts = generate_calibration_prompts(corpus)
    print(f"Generated {len(prompts)} calibration prompts")
    print()

    # Run calibration
    start_time = time.time()
    stats = run_calibration(
        args.url, args.model, prompts, args.max_tokens, verbose=not args.quiet
    )
    elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"Prompts processed: {stats['prompts_processed']}/{len(prompts)}")
    print(f"Total tokens: {stats['total_prompt_tokens']} prompt + {stats['total_completion_tokens']} completion")
    print(f"Errors: {stats['errors']}")
    print(f"Time: {elapsed:.1f}s")
    print()
    print("NEXT STEPS:")
    print("1. Stop the server (Ctrl+C)")
    print("2. Scales will be saved to /tmp/vllm_kv_scales_tp*.json")
    print("3. Restart server with:")
    print("   VLLM_KV_SCALES_FILE=/tmp/vllm_kv_scales.json vllm serve ...")
    print()
    print("To view scales before stopping:")
    print("   cat /tmp/vllm_kv_scales_*.json")

    return 0


if __name__ == "__main__":
    exit(main())
