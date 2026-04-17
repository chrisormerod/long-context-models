#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:56:35 2026

@author: cormerod
"""
import os 

os.chdir("..")

import pandas as pd
import numpy as np
from textstats import TextStats
from sklearn.metrics import cohen_kappa_score
from utils.bias import standardized_mean_difference
import re
from collections import defaultdict

def family(col):
    # keeps 'score' as-is, strips trailing _<digit+>
    return re.sub(r'_\d+$', '', col)

scores = pd.read_csv("/mnt/c/data/asap2/long/mbert_0.csv").drop("pred_score",axis=1)

cols = ['score']
for m in ["mbert","mbert_trunc",'mamba', "mamba_trunc", "mamba2", "mamba2_trunc"]:
    for i in range(10):
        try:
            scores[f'{m}_{i}'] = pd.read_csv(f"/mnt/c/data/asap2/long/{m}_{i}.csv")['pred_score']
            cols.append(f"{m}_{i}")
        except:
            pass
        
qwk_mat = {c1:{c2: cohen_kappa_score(scores[c1],scores[c2],weights="quadratic") for c2 in cols} for c1 in cols}

grouped = defaultdict(list)

for c1 in cols:
    for c2 in cols:
        if c1 == c2:
            continue
        
        g1 = family(c1)
        g2 = family(c2)
        grouped[(g1, g2)].append(qwk_mat[c1][c2])

avg_qwk = {
    (g1, g2): np.mean(vals)
    for (g1, g2), vals in grouped.items()
}