# long-context-models

A small experimental codebase for comparing long-context sequence-classification models on the ASAP 2 essay-scoring dataset. The repository trains and evaluates six configurations:

- **ModernBERT (long context)**
- **ModernBERT (truncated)**
- **Mamba (long context)**
- **Mamba (truncated)**
- **Mamba2 (long context)**
- **Mamba2 (truncated)**

The current setup uses:

- `answerdotai/ModernBERT-base`
- `state-spaces/mamba-130m-hf`
- `AntonV/mamba2-130m-hf`

Long-context runs use `MAX_LENGTH = 8196`, while truncated baselines use `MAX_LENGTH = 512`.

## What this repository does

The project is organized around a simple experiment loop:

1. Load ASAP 2 training and test CSV files from disk.
2. Fine-tune a sequence-classification model for essay scoring.
3. Score the test split and write predictions to CSV.
4. Repeat the process across model families and context lengths.
5. Optionally run speed and bias analysis scripts on the generated predictions.

The training pipeline uses Hugging Face `Trainer`, evaluates with **quadratic weighted kappa (QWK)** and accuracy, and saves the best checkpoint at the end of training.

## Repository layout

```text
long-context-models/
├── config/
│   └── __init__.py            # Environment-specific paths and global constants
├── data/
│   └── asap_data.py           # ASAP 2 dataset loader
├── modeling/
│   ├── mamba_modeling.py      # Sequence-classification wrapper for Mamba/Mamba2
│   └── train_model.py         # Shared training loop and metrics
├── scripts/
│   ├── train_mbert.py
│   ├── train_mbert_trunc.py
│   ├── train_mamba.py
│   ├── train_mamba_trunc.py
│   ├── train_mamba2.py
│   └── train_mamba2_trunc.py
├── testing/
│   ├── bias.py                # Post-hoc agreement / subgroup analysis script
│   └── speed.py               # Inference-speed benchmark across sequence lengths
├── utils/
│   └── bias.py                # Helper functions for bias analysis
└── main.py                    # Runs all training scripts repeatedly
```

## Data layout

The loader expects the ASAP 2 files to exist under a local `asap2/` directory inside the configured data root.

```text
DATA_DIR/
└── asap2/
    ├── ASAP_2_Final_github_train.csv
    └── ASAP_2_Final_github_test.csv
```

The dataset loader returns a Hugging Face `DatasetDict` with `train` and `test` splits.

## Environment configuration

Configuration lives in `config/__init__.py` and depends on the `LOCATION` environment variable.

- When `LOCATION=LAPTOP`
  - `DATA_DIR=/mnt/c/data`
  - `MODEL_DIR=/mnt/c/models`
  - `TMP_DIR=/mnt/c/tmp`
- Otherwise
  - `DATA_DIR=/home/ubuntu/data`
  - `MODEL_DIR=/home/ubuntu/models`
  - `TMP_DIR=/home/ubuntu/tmp`

`EPOCHS` is currently set to `4`.

> Important: `LOCATION` is accessed with `os.environ['LOCATION']`, so it must be defined before you run the project.

Example:

```bash
export LOCATION=LAPTOP
mkdir -p /mnt/c/tmp outputs
```

## Installation

This repository does not currently include a `requirements.txt` or `pyproject.toml`, so install dependencies from the imports used in the codebase.

A reasonable starting point is:

```bash
pip install torch transformers datasets pandas numpy scikit-learn tqdm textstats
```

Depending on your environment, you may also want GPU-enabled PyTorch and any accelerator-specific dependencies.

## Training

Each training script follows the same pattern:

- load ASAP 2
- instantiate a model and tokenizer
- rename dataset columns to `text` and `label`
- fine-tune for 4 epochs
- score the test set
- write predictions to CSV

### Individual runs

Run one experiment directly:

```bash
python -m scripts.train_mbert --run-id 0 --output-csv outputs/mbert_0.csv
python -m scripts.train_mbert_trunc --run-id 0 --output-csv outputs/mbert_trunc_0.csv
python -m scripts.train_mamba --run-id 0 --output-csv outputs/mamba_0.csv
python -m scripts.train_mamba_trunc --run-id 0 --output-csv outputs/mamba_trunc_0.csv
python -m scripts.train_mamba2 --run-id 0 --output-csv outputs/mamba2_0.csv
python -m scripts.train_mamba2_trunc --run-id 0 --output-csv outputs/mamba2_trunc_0.csv
```

### Full experiment sweep

`main.py` runs all six experiment configurations and, for each one, launches 10 repeated runs with distinct output CSV names.

```bash
python main.py
```

By default, outputs are written to:

```text
outputs/
├── mbert_0.csv
├── mbert_1.csv
├── ...
├── mamba_0.csv
├── ...
└── mamba2_trunc_9.csv
```

## Training defaults

The shared trainer uses these defaults unless overridden by a script:

- `num_train_epochs = 4`
- `learning_rate = 5e-5`
- `warmup_ratio = 0.0` in the training scripts
- `per_device_train_batch_size = 8`
- `eval_strategy = "epoch"`
- `save_strategy = "epoch"`
- `metric_for_best_model = "QWK"`
- mixed precision (`fp16`) is enabled automatically when CUDA is available

Metrics:

- **QWK** (quadratic weighted kappa)
- **accuracy**
- **precision / recall / F1** only for binary-label tasks

## Model-specific notes

### ModernBERT

The ModernBERT experiments use `AutoModelForSequenceClassification.from_pretrained(...)` with `num_labels=7`.

### Mamba / Mamba2

The Mamba experiments use a custom `MambaForSequenceClassification` wrapper defined in `modeling/mamba_modeling.py`.

That wrapper:

- loads the backbone through Hugging Face `AutoModel`
- adds dropout plus a linear classification head
- uses `pooler_output` if the backbone provides it
- otherwise falls back to masked mean pooling over token embeddings
- computes cross-entropy loss when labels are present

## Benchmarking inference speed

`testing/speed.py` benchmarks inference latency for ModernBERT, Mamba, and Mamba2 across token lengths.

Example:

```bash
python -m testing.speed \
  --modern_model answerdotai/ModernBERT-base \
  --mamba_model state-spaces/mamba-130m-hf \
  --mamba2_model AntonV/mamba2-130m-hf \
  --min_len 64 \
  --max_len 8192 \
  --step 64 \
  --repeats 10 \
  --device cuda \
  --out bench_results.csv
```

The benchmark writes a CSV with columns:

- `length`
- `modern_ms`
- `mamba_ms`
- `mamba2_ms`

## Bias and agreement analysis

The repository also includes utilities for post-hoc analysis.

`utils/bias.py` provides:

- `standardized_mean_difference(...)`
- `matched_pred_kl_test(...)`

These are intended for subgroup or distribution-shift style checks on prediction behavior.

`testing/bias.py` loads prediction CSVs, compares runs, and computes agreement statistics based on quadratic weighted kappa.

## License

Apache-2.0.