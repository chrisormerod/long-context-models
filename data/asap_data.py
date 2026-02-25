"""
ASAP Dataset Loader

Provides a clean interface for loading the ASAP 2 dataset
from local CSV files into a HuggingFace DatasetDict.

Expected directory structure:

DATA_DIR/
    asap2/
        ASAP_2_Final_github_train.csv
        ASAP_2_Final_github_test.csv
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict

from config import DATA_DIR


def _load_split(csv_path: Path) -> Dataset:
    """
    Load a single CSV file into a HuggingFace Dataset.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    Dataset
        HuggingFace Dataset created from the CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Remove pandas index column if accidentally saved
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])

    return Dataset.from_pandas(df, preserve_index=False)


def get_asap(data_dir: Optional[str] = None) -> DatasetDict:
    """
    Load the ASAP 2 dataset as a HuggingFace DatasetDict.

    Parameters
    ----------
    data_dir : Optional[str]
        Optional override of the base DATA_DIR.
        If None, uses config.DATA_DIR.

    Returns
    -------
    DatasetDict
        Dictionary with:
            - "train"
            - "test"

    Example
    -------
    >>> dataset = get_asap()
    >>> dataset["train"]
    >>> dataset["test"]
    """
    base_dir = Path(data_dir or DATA_DIR) / "asap2"

    train_path = base_dir / "ASAP_2_Final_github_train.csv"
    test_path = base_dir / "ASAP_2_Final_github_test.csv"

    train_dataset = _load_split(train_path)
    test_dataset = _load_split(test_path)

    return DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )