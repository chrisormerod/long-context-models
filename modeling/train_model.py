from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
    ProgressCallback
)
from sklearn.metrics import cohen_kappa_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    
    accuracy = (preds == labels).mean().item()

    # Simple binary precision/recall/F1 if applicable
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "QWK":qwk,
        }

    return {"accuracy": accuracy, "QWK":qwk}


def train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    text_column: str = "text",
    label_column: str = "label",
    max_length: int = 4096,
    seed: int = 42,
    num_train_epochs: float = 4.0,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.1,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    logging_steps: int = 50,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 2,
    metric_for_best_model: str = "QWK",
    greater_is_better: bool = True,
    load_best_model_at_end: bool = True,
    report_to: str = "none",
    dataloader_num_workers: int = 2,
    fp16: Optional[bool] = None,
):
    """
    Train a Hugging Face sequence classification model.

    Args:
        model:
            A Hugging Face model, e.g. MambaForSequenceClassification.
        tokenizer:
            A Hugging Face tokenizer.
        train_dataset:
            Hugging Face Dataset for training.
        eval_dataset:
            Hugging Face Dataset for evaluation.
        output_dir:
            Directory to save checkpoints and final model.
        text_column:
            Name of the text column in the datasets.
        label_column:
            Name of the label column in the datasets.
        max_length:
            Maximum tokenized sequence length.
        seed:
            Random seed.
        num_train_epochs:
            Number of training epochs.
        learning_rate:
            Optimizer learning rate.
        weight_decay:
            Weight decay.
        warmup_ratio:
            Fraction of total steps used for LR warmup.
        per_device_train_batch_size:
            Train batch size per device.
        per_device_eval_batch_size:
            Eval batch size per device.
        gradient_accumulation_steps:
            Gradient accumulation steps.
        logging_steps:
            Log every N steps.
        eval_strategy:
            Evaluation strategy for Trainer.
        save_strategy:
            Checkpoint save strategy.
        save_total_limit:
            Max checkpoints to keep.
        metric_for_best_model:
            Metric used to select best model.
        greater_is_better:
            Whether larger metric is better.
        load_best_model_at_end:
            Whether to reload best checkpoint at the end.
        report_to:
            Reporting backend, e.g. "none", "wandb".
        dataloader_num_workers:
            Number of dataloader workers.
        fp16:
            Whether to use fp16. Defaults to True if CUDA is available.

    Returns:
        trainer: The trained Trainer instance.
        train_result: Output of trainer.train()
        eval_metrics: Final evaluation metrics dict.
    """
    set_seed(seed)

    if fp16 is None:
        fp16 = torch.cuda.is_available()

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def preprocess_function(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = examples[label_column]
        return tokenized

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    tokenized_eval = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset",
    )

    tokenized_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    tokenized_eval.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    # Resize embeddings if tokenizer vocab grew
    if hasattr(model, "mamba") and hasattr(model.mamba, "get_input_embeddings"):
        input_embeddings = model.mamba.get_input_embeddings()
        if input_embeddings is not None and len(tokenizer) > input_embeddings.num_embeddings:
            model.mamba.resize_token_embeddings(len(tokenizer))

    # Ensure pad token id is set on model config
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_strategy="steps",
        logging_steps=logging_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=save_total_limit,
        report_to=report_to,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[ProgressCallback()]
    )

    train_result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_metrics = trainer.evaluate()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    return trainer, train_result, eval_metrics