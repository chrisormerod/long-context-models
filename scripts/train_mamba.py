#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:02:47 2026

@author: cormerod
"""

import argparse
import gc
import os

import torch
from transformers import AutoTokenizer
from modeling.mamba_modeling import MambaForSequenceClassification
from data.asap_data import get_asap
from config import DATA_DIR, TMP_DIR, EPOCHS
from modeling.train_model import train


MODEL_ID = "state-spaces/mamba-130m-hf"
MAX_LENGTH = 8196

@torch.no_grad()
def score_dataset(dataset, model, tokenizer, max_length=MAX_LENGTH):
    device = model.device

    def score(example):
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def main(run_id: int, output_csv: str):
    data = get_asap(DATA_DIR)

    model = MambaForSequenceClassification.from_pretrained(
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
        max_length=MAX_LENGTH,
        per_device_train_batch_size=8,
    )

    # Ensure eval/inference uses the trained model currently inside trainer
    trained_model = trainer.model
    trained_model.eval()

    scored_test = score_dataset(data["test"], trained_model, tokenizer, max_length=MAX_LENGTH)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()

    main(args.run_id, args.output_csv)