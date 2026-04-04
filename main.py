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

model_id = "state-spaces/mamba-130m-hf"

data = get_asap(DATA_DIR)
model = MambaForSequenceClassification.from_pretrained(model_id, num_labels=7)
tokenizer = AutoTokenizer.from_pretrained(model_id)



trainer, train_result, eval_metrics = train(model, tokenizer, 
                                             data['train'].rename_columns({"full_text":"text","score":"label"}), 
                                             data['test'].rename_columns({"full_text":"text","score":"label"}), 
                                             output_dir = TMP_DIR)
