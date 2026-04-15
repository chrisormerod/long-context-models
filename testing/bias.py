#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:56:35 2026

@author: cormerod
"""
import pandas as pd
import numpy as np
from textstats import TextStats
from sklearn.metrics import cohen_kappa_score
# from utils.bias import SMD, 

from typing import Any, Dict, Iterable, List
import math


def qwk_list_to_tikz(data: List[Dict[str, Any]]) -> str:
    """
    Convert a list of dicts into TikZ/PGFPlots code for a line chart of
    essay length vs. QWK values.

    Expected keys in each dict:
      - "length"
      - "mbert"
      - "mamba"
      - "mbert_tr"
      - "mamba_tr"

    Returns
    -------
    str
        A complete tikzpicture/axis block as a string.

    Example
    -------
    tikz_code = qwk_list_to_tikz(results)
    print(tikz_code)
    """
    required_keys = {"length", "mbert", "mamba", "mbert_tr", "mamba_tr"}
    series_names = ["mbert", "mamba", "mbert_tr", "mamba_tr"]

    if not isinstance(data, list):
        raise TypeError("Input must be a list of dictionaries.")

    if not data:
        raise ValueError("Input list is empty.")

    cleaned = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"Element at index {i} is not a dictionary.")

        missing = required_keys - row.keys()
        if missing:
            raise ValueError(f"Element at index {i} is missing keys: {sorted(missing)}")

        try:
            length = int(row["length"])
        except Exception as e:
            raise ValueError(f"Invalid 'length' at index {i}: {row['length']!r}") from e

        cleaned_row = {"length": length}
        for key in series_names:
            try:
                value = float(row[key])
            except Exception as e:
                raise ValueError(f"Invalid '{key}' at index {i}: {row[key]!r}") from e

            if not math.isfinite(value):
                raise ValueError(f"Non-finite '{key}' at index {i}: {value!r}")

            cleaned_row[key] = value

        cleaned.append(cleaned_row)

    # Sort by length so lines are drawn left-to-right.
    cleaned.sort(key=lambda x: x["length"])

    def coords_for(series: str) -> str:
        return " ".join(f"({row['length']}, {row[series]:.6f})" for row in cleaned)

    tikz = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
    width=12cm,
    height=8cm,
    xlabel={{Length}},
    ylabel={{QWK}},
    xmin={min(row['length'] for row in cleaned)},
    xmax={max(row['length'] for row in cleaned)},
    ymin=0,
    ymax=1,
    grid=major,
    legend pos=south east,
    legend cell align=left,
    line width=1pt,
    mark size=2pt,
]
\addplot+[mark=*] coordinates {{{coords_for("mbert")}}};
\addlegendentry{{mbert}}

\addplot+[mark=square*] coordinates {{{coords_for("mamba")}}};
\addlegendentry{{mamba}}

\addplot+[mark=triangle*] coordinates {{{coords_for("mbert_tr")}}};
\addlegendentry{{mbert\_tr}}

\addplot+[mark=diamond*] coordinates {{{coords_for("mamba_tr")}}};
\addlegendentry{{mamba\_tr}}

\end{{axis}}
\end{{tikzpicture}}
""".strip()

    return tikz

def standardized_mean_difference(y_true, y_pred, ddof=1):
    """
    Compute standardized mean difference between y_true and y_pred.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    ddof : int
        Delta degrees of freedom for variance (default=1 for sample variance).

    Returns
    -------
    float
        Standardized mean difference.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    mean_diff = y_true.mean() - y_pred.mean()

    var_true = y_true.var(ddof=ddof)
    var_pred = y_pred.var(ddof=ddof)

    pooled_sd = np.sqrt(0.5 * (var_true + var_pred))

    if pooled_sd == 0:
        return np.nan  # or 0.0 depending on preference

    return mean_diff / pooled_sd

mbert = pd.read_csv("/mnt/c/data/asap2/long/mbert.csv")
mamba = pd.read_csv("/mnt/c/data/asap2/long/mamba.csv")

mbert_tr = pd.read_csv("/mnt/c/data/asap2/long/mbert_trunc.csv")
mamba_tr = pd.read_csv("/mnt/c/data/asap2/long/mamba_trunc.csv")

for idx, row in mbert.iterrows():
    if np.isnan(row['essay_word_count']):
        mbert.loc[idx,'essay_word_count'] = TextStats.getWordCount(None,row['full_text'])[0]
mbert_tr['essay_word_count'], mamba['essay_word_count'], mamba_tr['essay_word_count'] = mbert['essay_word_count'],mbert['essay_word_count'],mbert['essay_word_count']
lengths = np.arange(100,801,10)
bin_sizes = [ sum(mbert['essay_word_count'].map(lambda x:x <= l+200 and x >= l-200)) for l in lengths]

smd_vals = []
qwk_vals = []

for l in lengths:
    indices = mbert['essay_word_count'].map(lambda x:x >= l-100 and x <= l+100)
    smd_vals.append({"length":l, 
                     'mbert':standardized_mean_difference(mbert.loc[indices]['score'], mbert.loc[indices]['pred_score']),
                     'mamba':standardized_mean_difference(mamba.loc[indices]['score'], mamba.loc[indices]['pred_score']),
                     'mbert_tr':standardized_mean_difference(mbert_tr.loc[indices]['score'], mbert_tr.loc[indices]['pred_score']),
                     'mamba_tr':standardized_mean_difference(mamba_tr.loc[indices]['score'], mamba_tr.loc[indices]['pred_score'])
            })
    qwk_vals.append({"length":l, 
                     'mbert':cohen_kappa_score(mbert.loc[indices]['score'], mbert.loc[indices]['pred_score'],weights="quadratic"),
                     'mamba':cohen_kappa_score(mamba.loc[indices]['score'], mamba.loc[indices]['pred_score'],weights="quadratic"),
                     'mbert_tr':cohen_kappa_score(mbert_tr.loc[indices]['score'], mbert_tr.loc[indices]['pred_score'],weights="quadratic"),
                     'mamba_tr':cohen_kappa_score(mamba_tr.loc[indices]['score'], mamba_tr.loc[indices]['pred_score'],weights="quadratic"),
                    })
    
    
races = list(set(mbert['race_ethnicity']))

race_bias = []

for r in races:
    row = {'race':r}
    row.update(matched_pred_kl_test(mbert, mbert[mbert['race_ethnicity']=="White"], pred_col="pred_score"))
    race_bias.append(row)