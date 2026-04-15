#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:42:13 2026

@author: cormerod
"""

import numpy as np
import pandas as pd

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

    mean_diff = y_pred.mean() - y_true.mean()

    var_true = y_true.var(ddof=ddof)
    var_pred = y_pred.var(ddof=ddof)

    pooled_sd = np.sqrt(0.5 * (var_true + var_pred))

    if pooled_sd == 0:
        return np.nan  # or 0.0 depending on preference

    return mean_diff / pooled_sd


def matched_pred_kl_test(
    data,
    subdata,
    score_col="score",
    pred_col="pred",
    n_resamples=5000,
    smoothing=1e-8,
    random_state=None,
    sample_with_replacement=False,
):
    """
    Test whether subdata's machine-score distribution is consistent with the
    machine-score distribution expected from the full population `data`,
    conditional on subdata having its observed human-score distribution.

    Parameters
    ----------
    data : pd.DataFrame
        Full population. Must contain score_col and pred_col.
    subdata : pd.DataFrame
        Subgroup of interest. Must contain score_col and pred_col.
    score_col : str
        Column with human-assigned scores.
    pred_col : str
        Column with machine-assigned scores.
    n_resamples : int
        Number of Monte Carlo null resamples.
    smoothing : float
        Small additive constant to stabilize KL when probabilities are zero.
    random_state : int or None
        Random seed.
    sample_with_replacement : bool
        Whether matched null samples are drawn with replacement within each
        score stratum.

    Returns
    -------
    dict with:
        observed_kl : float
            KL(subgroup pred dist || expected pred dist under null)
        expected_kl : float
            Mean null KL from matched resamples
        p_value : float
            Monte Carlo p-value for observed_kl
        null_kls : np.ndarray
            KL values from matched null resamples
        expected_pred_distribution : pd.Series
            Null expected machine-score distribution
        subgroup_pred_distribution : pd.Series
            Observed subgroup machine-score distribution
        subgroup_score_counts : pd.Series
            Human-score counts in subgroup
    """
    rng = np.random.default_rng(random_state)

    # Basic checks
    required = {score_col, pred_col}
    if not required.issubset(data.columns):
        raise ValueError(f"`data` must contain columns {required}")
    if not required.issubset(subdata.columns):
        raise ValueError(f"`subdata` must contain columns {required}")

    data = data[[score_col, pred_col]].dropna().copy()
    subdata = subdata[[score_col, pred_col]].dropna().copy()

    if len(subdata) == 0:
        raise ValueError("`subdata` is empty after dropping NA rows.")
    if len(data) == 0:
        raise ValueError("`data` is empty after dropping NA rows.")

    # Human-score composition of subgroup
    subgroup_score_counts = subdata[score_col].value_counts().sort_index()

    # Make sure full data has support for all subgroup human scores
    missing_scores = subgroup_score_counts.index.difference(data[score_col].unique())
    if len(missing_scores) > 0:
        raise ValueError(
            f"`data` has no rows for subgroup score values: {list(missing_scores)}"
        )

    # Universe of machine-score categories
    pred_levels = pd.Index(
        sorted(pd.unique(pd.concat([data[pred_col], subdata[pred_col]], ignore_index=True)))
    )

    def prob_vector_from_series(x, levels):
        counts = x.value_counts().reindex(levels, fill_value=0).astype(float)
        probs = counts + smoothing
        probs = probs / probs.sum()
        return probs

    def kl_divergence(p, q):
        # p and q are aligned probability vectors
        return float(np.sum(p * np.log(p / q)))

    # Null expected pred distribution:
    # q(pred) = sum_score P(pred | score, data) * P_sub(score)
    total_sub = subgroup_score_counts.sum()
    score_weights = subgroup_score_counts / total_sub

    expected_pred = pd.Series(0.0, index=pred_levels)
    for score_value, weight in score_weights.items():
        stratum = data.loc[data[score_col] == score_value, pred_col]
        stratum_probs = prob_vector_from_series(stratum, pred_levels)
        expected_pred += weight * stratum_probs

    expected_pred = expected_pred / expected_pred.sum()

    # Observed subgroup pred distribution
    subgroup_pred = prob_vector_from_series(subdata[pred_col], pred_levels)

    observed_kl = kl_divergence(subgroup_pred.values, expected_pred.values)

    # Monte Carlo null distribution:
    # draw matched samples from data with the same human-score counts as subgroup
    null_kls = np.empty(n_resamples, dtype=float)

    # Pre-split data by human score for speed
    grouped = {
        score_value: grp[pred_col].to_numpy()
        for score_value, grp in data.groupby(score_col, sort=False)
    }

    for b in range(n_resamples):
        sampled_preds = []

        for score_value, n_needed in subgroup_score_counts.items():
            pool = grouped[score_value]

            if (not sample_with_replacement) and (n_needed > len(pool)):
                raise ValueError(
                    f"Cannot sample {n_needed} rows without replacement from "
                    f"score={score_value}, only {len(pool)} available. "
                    f"Set sample_with_replacement=True."
                )

            draw = rng.choice(pool, size=n_needed, replace=sample_with_replacement)
            sampled_preds.append(draw)

        sampled_preds = np.concatenate(sampled_preds)
        sampled_pred_dist = prob_vector_from_series(pd.Series(sampled_preds), pred_levels)
        null_kls[b] = kl_divergence(sampled_pred_dist.values, expected_pred.values)

    # One-sided Monte Carlo p-value:
    # probability that a matched subgroup drawn under the null has KL at least as large
    p_value = (1 + np.sum(null_kls >= observed_kl)) / (n_resamples + 1)

    return {
        "observed_kl": observed_kl,
        "expected_kl": float(null_kls.mean()),
        "p_value": float(p_value),
        "null_kls": null_kls,
        "expected_pred_distribution": expected_pred,
        "subgroup_pred_distribution": subgroup_pred,
        "subgroup_score_counts": subgroup_score_counts,
    }