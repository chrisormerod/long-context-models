#!/usr/bin/env python3
"""
Benchmark script: compare inference speed of sequence-classification models
for input lengths from MIN_LEN..MAX_LEN (step STEP), averaging REPEATS runs.

Usage:
    python bench_sequence_classification.py \
        --modern_model YOUR_MODERNBERT_SEQCLASS_ID \
        --mamba_model state-spaces/mamba-130m-hf \
        --min_len 64 --max_len 8192 --step 64 --repeats 10 --device cuda

Notes:
 - The script uses AutoTokenizer + AutoModelForSequenceClassification where possible.
 - If you have a custom wrapper class (e.g., MambaForSequenceClassification), pass its HF repo id
   to --mamba_model or adjust the loading section below.
 - For correctness we generate text then call tokenizer(..., max_length=L, truncation=True, padding='max_length')
   to ensure tokenized input length is exactly L.
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from modeling.mamba_modeling import MambaForSequenceClassification

def load_mamba_and_tokenizer(model_id: str, device: torch.device) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """
    Load a Mamba sequence-classification model and its tokenizer.

    Args:
        model_id: Hugging Face model identifier or local path.
        device: PyTorch device (cpu or cuda).

    Returns:
        Tuple of (model, tokenizer) both loaded and ready for inference.

    Notes:
        If the HF repo contains custom code (and requires trust_remote_code), you can
        add `trust_remote_code=True` to from_pretrained calls below.
    """
    # Use AutoTokenizer (fast) and MambaForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = MambaForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, tokenizer

def load_model_and_tokenizer(model_id: str, device: torch.device) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """
    Load a standard sequence-classification model and its tokenizer.

    Args:
        model_id: Hugging Face model identifier or local path.
        device: PyTorch device (cpu or cuda).

    Returns:
        Tuple of (model, tokenizer) both loaded and ready for inference.

    Notes:
        If the HF repo contains custom code (and requires trust_remote_code), you can
        add `trust_remote_code=True` to from_pretrained calls below.
    """
    # Use AutoTokenizer (fast) and AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, tokenizer


def make_text_for_tokens(n_tokens: int) -> str:
    """
    Produce a simple string that tokenizes to approximately n_tokens tokens.
    
    Uses repeated short words which tokenizers typically split into roughly one token each.
    Adds a buffer to ensure sufficient tokens even with overhead from special tokens.

    Args:
        n_tokens: Target number of tokens.

    Returns:
        str: A simple repetitive text string.
    """
    # Use a short repetitive phrase. Multiply a bit to be safe.
    return ("word " * ((n_tokens // 1) + 50)).strip()


def single_forward_time_ms(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    text: str,
    target_len: int,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 10,
) -> float:
    """
    Measure average forward pass latency (ms) for a single example padded/truncated to target_len tokens.

    Args:
        model: Model in eval() mode already and on device.
        tokenizer: Fast tokenizer for encoding.
        text: Source string (will be force padded/truncated to max_length=target_len).
        target_len: Number of tokens to produce after tokenization.
        device: cpu/cuda device.
        warmup: Number of warm-up forwards to run (not timed).
        repeats: Number of timed forwards to run (results averaged).

    Returns:
        float: Average forward pass latency in milliseconds.
    """
    # Tokenize to exact length target_len using truncation/padding
    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=target_len,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
    )
    # Ensure tensors on device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    # Some models require token_type_ids
    # token_type_ids = enc["token_type_ids"].to(device) if "token_type_ids" in enc else None
    token_type_ids = None
    
    # Warmup runs: allow GPU/cache to stabilize
    with torch.inference_mode():
        for _ in range(warmup):
            if token_type_ids is not None:
                _ = model(input_ids=input_ids)
            else:
                _ = model(input_ids=input_ids)

            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed runs: measure actual latency
    times = []
    with torch.inference_mode():
        for _ in range(repeats):
            t0 = time.perf_counter()
            if token_type_ids is not None:
                _ = model(input_ids=input_ids)
            else:
                _ = model(input_ids=input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    return float(np.mean(times))


def benchmark_models(
    modern_model_id: str,
    mamba_model_id: str,
    mamba2_model_id: str,
    min_len: int = 64,
    max_len: int = 8192,
    step: int = 64,
    repeats: int = 10,
    device_str: str = "cuda",
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Run full benchmark for all three models across a range of sequence lengths.
    
    Benchmarks three models (ModernBERT, Mamba, Mamba2) over a range of input lengths,
    measuring inference latency for each.

    Args:
        modern_model_id: Hugging Face ID for ModernBERT model.
        mamba_model_id: Hugging Face ID for Mamba model.
        mamba2_model_id: Hugging Face ID for Mamba2 model.
        min_len: Minimum sequence length to benchmark.
        max_len: Maximum sequence length to benchmark.
        step: Step size between sequence lengths.
        repeats: Number of forward passes per sequence length (results averaged).
        device_str: Device string ("cuda" or "cpu").

    Returns:
        dict: Results keyed by model name ("modern", "mamba", "mamba2").
              Each maps to a list of (sequence_length, latency_ms) tuples.
    """
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Using device: {device}")

    print("Loading Modern model/tokenizer...")
    modern_model, modern_tokenizer = load_model_and_tokenizer(modern_model_id, device)

    print("Loading Mamba model/tokenizer...")
    mamba_model, mamba_tokenizer = load_mamba_and_tokenizer(mamba_model_id, device)
    
    print("Loading Mamba2 model/tokenizer...")
    mamba2_model, mamba2_tokenizer = load_mamba_and_tokenizer(mamba2_model_id, device)

    lengths = list(range(min_len, max_len + 1, step))
    results = {"modern": [], "mamba": [], "mamba2":[]}

    for L in tqdm(lengths, desc="Lengths"):
        text = make_text_for_tokens(L)

        # Modern model
        t_modern = single_forward_time_ms(
            model=modern_model,
            tokenizer=modern_tokenizer,
            text=text,
            target_len=L,
            device=device,
            warmup=2,
            repeats=repeats,
        )

        # Mamba model
        t_mamba = single_forward_time_ms(
            model=mamba_model,
            tokenizer=mamba_tokenizer,
            text=text,
            target_len=L,
            device=device,
            warmup=2,
            repeats=repeats,
        )

        # Mamba2 model
        t_mamba2 = single_forward_time_ms(
            model=mamba2_model,
            tokenizer=mamba2_tokenizer,
            text=text,
            target_len=L,
            device=device,
            warmup=2,
            repeats=repeats,
        )

        results["modern"].append((L, t_modern))
        results["mamba"].append((L, t_mamba))
        results["mamba2"].append((L, t_mamba2))

        # Print progress
        print(f" L={L:5d} tokens -> modern {t_modern:.2f} ms, mamba {t_mamba:.2f} ms, mamba2 {t_mamba2:.2f} ms")

    return results


def save_results_csv(results: Dict[str, List[Tuple[int, float]]], out_path: str = "bench_results.csv"):
    """
    Save benchmark results to a CSV file.
    
    Writes results in tabular format with columns for sequence length and latency per model.

    Args:
        results: Dict mapping model names to lists of (length, latency_ms) tuples.
        out_path: Output CSV file path.
    """
    import csv

    lengths = [l for l, _ in results["modern"]]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["length", "modern_ms", "mamba_ms", "mamba2_ms"])
        for i, L in enumerate(lengths):
            writer.writerow([L, results["modern"][i][1], results["mamba"][i][1], results["mamba2"][i][1]])
    print(f"Saved results to {out_path}")


def parse_args():
    """
    Parse and return command-line arguments.
    
    Returns:
        Namespace: Parsed arguments with the following attributes:
            - modern_model: HF model ID for ModernBERT
            - mamba_model: HF model ID for Mamba
            - mamba2_model: HF model ID for Mamba2
            - min_len: Minimum sequence length
            - max_len: Maximum sequence length
            - step: Step size between lengths
            - repeats: Number of forward passes per length
            - device: Device to use ("cuda" or "cpu")
            - out: Output CSV file path
    """
    p = argparse.ArgumentParser(description="Benchmark ModernBERT vs Mamba vs Mamba2 sequence classification inference speed")
    p.add_argument("--modern_model", type=str, default="answerdotai/ModernBERT-base", help="HF id of ModernBERT seq-class model (or substitute).")
    p.add_argument("--mamba_model", type=str, default="state-spaces/mamba-130m-hf", help="HF id/path for Mamba seq-class model.")
    p.add_argument("--mamba2_model", type=str, default="AntonV/mamba2-130m-hf", help="HF id/path for Mamba2 seq-class model.")
    p.add_argument("--min_len", type=int, default=64)
    p.add_argument("--max_len", type=int, default=8192)
    p.add_argument("--step", type=int, default=64)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out", type=str, default="bench_results.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Warn: user may want to replace default modern_model with an actual ModernBERT classifier repo id.
    if args.modern_model == "answerdotai/Modern":
        print("Note: default modern_model is 'ModernBERT'. If you have a ModernBERT classifier model_id, pass it with --modern_model.")

    results = benchmark_models(
        modern_model_id=args.modern_model,
        mamba_model_id=args.mamba_model,
        mamba2_model_id=args.mamba2_model,
        min_len=args.min_len,
        max_len=args.max_len,
        step=args.step,
        repeats=args.repeats,
        device_str=args.device,
    )

    save_results_csv(results, out_path=args.out)
    print("Done.")
