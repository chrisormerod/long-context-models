#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:02:47 2026

@author: cormerod
"""

from data.asap_data import get_asap
from config import DATA_DIR, TMP_DIR
from modeling.train_model import train
from modeling.mamba_modeling import MambaForSequenceClassification
from transformers import AutoTokenizer
import torch

model_id = "state-spaces/mamba-130m-hf"

data = get_asap(DATA_DIR)
model = MambaForSequenceClassification.from_pretrained(model_id, num_labels=7)
# model.classifier = torch.nn.Linear(in_features = 768,out_features=7, bias=False)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# for L in model.mamba.layers:
#     L.mixer.A_log.requires_grad_(False)
#     L.mixer.D.requires_grad_(False)
    
for x in model.backbone.layers: x.requires_grad_(False)
for x in model.backbone.layers: x.mixer.x_proj.requires_grad_(True)
for x in model.backbone.layers: x.mixer.dt_proj.requires_grad_(True)
for x in model.backbone.layers: x.mixer.in_proj.requires_grad_(True)
for x in model.backbone.layers: x.mixer.out_proj.requires_grad_(True)

trainer, train_result, eval_metrics = train(model, tokenizer, 
                                             data['train'].rename_columns({"full_text":"text","score":"label"}), 
                                             data['test'].rename_columns({"full_text":"text","score":"label"}), 
                                             output_dir = TMP_DIR,
                                             warmup_ratio = 0.0,
                                             per_device_train_batch_size=16)
