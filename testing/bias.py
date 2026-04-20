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
from sklearn.metrics import cohen_kappa_score, accuracy_score
from utils.bias import (standardized_mean_difference,
                        matched_pred_kl_test)
import re
from collections import defaultdict

models = ["mbert","mbert_trunc",'mamba', "mamba_trunc"]

def family(col):
    # keeps 'score' as-is, strips trailing _<digit+>
    return re.sub(r'_\d+$', '', col)

scores = pd.read_csv("/mnt/c/data/asap2/long/mbert_0.csv").drop("pred_score",axis=1)
scores['len'] = scores.apply(lambda x:x['essay_word_count'] if not np.isnan(x['essay_word_count']) else len(x['full_text'].split()),axis=1)

cols = ['score']
for m in models:
    for i in range(10):
        try:
            scores[f'{m}_{i}'] = pd.read_csv(f"/mnt/c/data/asap2/long/{m}_{i}.csv")['pred_score']
            cols.append(f"{m}_{i}")
        except:
            pass
        
qwk_mat = {c1:{c2: cohen_kappa_score(scores[c1],scores[c2],weights="quadratic") for c2 in cols} for c1 in cols}
smd_mat = {c1:{c2: standardized_mean_difference(scores[c1],scores[c2]) for c2 in cols} for c1 in cols}
exa_mat = {c1:{c2: accuracy_score(scores[c1],scores[c2]) for c2 in cols} for c1 in cols}


grouped_qwk = defaultdict(list)
grouped_smd = defaultdict(list)
grouped_exa = defaultdict(list)


for c1 in cols:
    for c2 in cols:
        if c1 == c2:
            continue
        
        g1 = family(c1)
        g2 = family(c2)
        
        grouped_qwk[(g1, g2)].append(qwk_mat[c1][c2])
        grouped_smd[(g1, g2)].append(smd_mat[c1][c2])
        grouped_exa[(g1, g2)].append(exa_mat[c1][c2])


avg_qwk = {
    (g1, g2): np.mean(vals)
    for (g1, g2), vals in grouped_qwk.items()
}

print("QWK Averages")
print(pd.DataFrame({c1:{c2:avg_qwk[c1,c2] for c2 in models} for c1 in ['score']+models}).to_latex(float_format="%.3f"))

avg_smd = {
    (g1, g2): np.mean(vals)
    for (g1, g2), vals in grouped_smd.items()
}

print("SMD Averages")
print(pd.DataFrame({c1:{c2:avg_smd[c1,c2] for c2 in models} for c1 in ['score']+models}).to_latex(float_format="%.3f"))

avg_exa = {
    (g1, g2): np.mean(vals)
    for (g1, g2), vals in grouped_exa.items()
}

print("EXA Averages")
print(pd.DataFrame({c1:{c2:avg_exa[c1,c2] for c2 in models} for c1 in ['score']+models}).to_latex(float_format="%.3f"))

lengths = np.arange(100, 900, 10)

rows_qwk = []
rows_smd = []
rows_exa = []


from tqdm import tqdm

for l in tqdm(lengths):
    sub = scores[scores['len'].map(lambda x: x >= l-100 and x <= l+100)]
    qwk_sub = {c1:{c2: cohen_kappa_score(sub[c1],sub[c2],weights="quadratic") for c2 in ['score']} for c1 in cols}
    smd_sub = {c1:{c2: standardized_mean_difference(sub[c1],sub[c2]) for c2 in ['score']} for c1 in cols}
    exa_sub = {c1:{c2: accuracy_score(sub[c1],sub[c2]) for c2 in ['score']} for c1 in cols}
    
    row_qwk = {family(x): np.mean([qwk_sub[k]['score'] for k in qwk_sub if family(k) == family(x)]) for x in qwk_sub}
    row_smd = {family(x): np.mean([smd_sub[k]['score'] for k in smd_sub if family(k) == family(x)]) for x in smd_sub}
    row_exa = {family(x): np.mean([exa_sub[k]['score'] for k in smd_sub if family(k) == family(x)]) for x in smd_sub}
    
    row_qwk['length'] = l
    rows_qwk.append(row_qwk)
    row_smd['length'] = l
    rows_smd.append(row_smd)
    row_exa['length'] = l
    rows_exa.append(row_exa)

qwk_df = pd.DataFrame(rows_qwk)
smd_df = pd.DataFrame(rows_smd)
exa_df = pd.DataFrame(rows_exa)


races = scores['race_ethnicity'].unique()
bias_rows = []
kl_rows = []


idx = {r: scores['race_ethnicity']==r for r in races}
idx['ELL'] = scores['ell_status'] == "Yes"
idx['Eco'] = scores['economically_disadvantaged'] == "Economically disadvantaged"
idx['Dis'] = scores['student_disability_status'] == 'Identified as having disability'


for g,i in idx.items():
    sub = scores[i]
    if len(sub) > 0:
        smd_sub = {c1:{c2: standardized_mean_difference(sub[c1],sub[c2]) for c2 in ['score']} for c1 in cols}
        row_smd = {"group":g}
        row_smd.update({family(x): np.mean([smd_sub[k]['score'] for k in smd_sub if family(k) == family(x)]) for x in smd_sub})
        bias_rows.append(row_smd)
    
    
        row_kl = {"group":g}
        row_kl.update(matched_pred_kl_test(scores, sub, ))
    
print(pd.DataFrame(bias_rows).drop("score",axis=1).to_latex(float_format="%.3f",index=False))


