#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 19:23:56 2026

@author: cormerod
"""
import os
os.environ['CURL_CA_BUNDLE'] = '/home/cormerod/pem/Zscaler-AWS-Feb2025.pem'
from datasets import load_dataset

dataset_path = "ruudra1/PERSUADE"

data = load_dataset(dataset_path)