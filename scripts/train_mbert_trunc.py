#!/usr/bin/env python3
"""
Train a ModernBERT sequence classification model on ASAP 2 dataset with truncated sequences.

This script trains the ModernBERT-base model on essay scoring tasks with a configurable
maximum sequence length. Useful for studying how sequence length affects model performance.

Usage:
    python -m scripts.train_mbert_trunc --run-id 0 --output-csv output.csv --max-length 512
"""

import argparse
import gc
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data.asap_data import get_asap
from config import DATA_DIR, TMP_DIR, EPOCHS
from modeling.train_model import train


MODEL_ID = "answerdotai/ModernBERT-base"


@torch.no_grad()
def score_dataset(dataset, model, tokenizer, max_length):
    """
    Generate predictions for a dataset using a trained model.
    
    Applies the model to each example in the dataset and returns predicted
    class labels along with raw logits for each class.

    Args:
        dataset: Hugging Face Dataset with 'full_text' column.
        model: Sequence classification model in eval mode.
        tokenizer: Tokenizer for encoding text.
        max_length: Maximum sequence length for tokenization.

    Returns:
        Dataset: Original dataset with added columns:
            - 'pred_score': Predicted class label (int)
            - 'pred_logit_i': Logit for class i (float) for each class
    """
    device = model.device

    def score(example):
        """Score a single example and return prediction and logits."""
        model_input = tokenizer(
            example["full_text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        model_output = model(**model_input)
        logits = model_output.logits[0]

        output = {
            "pred_score": int(logits.argmax(-1).item())
        }
        output.update({
            f"pred_logit_{i}": float(logits[i].item())
            for i in range(logits.shape[0])
        })
        return output

    return dataset.map(score)


def cleanup():
    """
    Cleanup GPU memory and perform garbage collection.
    
    Explicitly releases GPU memory and empties cache to prevent
    out-of-memory errors between consecutive training runs.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def main(run_id: int, output_csv: str, max_length : int):
    """
    Main training and evaluation pipeline.
    
    Loads data, trains the ModernBERT model on the ASAP 2 dataset with the specified
    max sequence length, and saves predictions to a CSV file.

    Args:
        run_id: Unique identifier for this training run (used for checkpoint dir).
        output_csv: Path to save prediction CSV file.
        max_length: Maximum sequence length for tokenization.
    """
    data = get_asap(DATA_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=7,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    trainer, train_result, eval_metrics = train(
        model,
        tokenizer,
        data["train"].rename_columns({"full_text": "text", "score": "label"}),
        data["test"].rename_columns({"full_text": "text", "score": "label"}),
        output_dir=os.path.join(TMP_DIR, f"run_{run_id}"),
        warmup_ratio=0.0,
        num_train_epochs=EPOCHS,
        max_length=max_length,
        per_device_train_batch_size=8,
    )

    # Ensure eval/inference uses the trained model currently inside trainer
    trained_model = trainer.model
    trained_model.eval()

    scored_test = score_dataset(data["test"], trained_model, tokenizer, max_length=max_length)
    scored_test.to_csv(output_csv, index=False)

    # Explicit cleanup before process exit
    del trainer
    del train_result
    del eval_metrics
    del trained_model
    del model
    del tokenizer
    del data
    del scored_test
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ModernBERT on ASAP 2 with truncated sequences")
    parser.add_argument("--run-id", type=int, required=True, help="Unique run identifier")
    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV path for predictions")
    parser.add_argument("--max-length", type=int, required=True, help="Maximum sequence length")
    args = parser.parse_args()

    main(args.run_id, args.output_csv, args.max_length)
