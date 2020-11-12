#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:55:31 2020

@author: bruce
"""

import pickle
import os
from collections import Counter

label_path = '/media/bruce/2T/data/UI_PRMD/st-gcn/cv_cs/ang'

sample_names = []
labels = []

folds = ['1', '2', '3', '4', '5']
actions = [2]#,3,4,5,6,7,8,9]
for action in actions:
    for fold in folds:
        with open(os.path.join(label_path,str(action),fold, 'eval_label.pkl'), 'rb') as f:
           sample_name, label = pickle.load(f)
           sample_names.extend(sample_name)
           labels.extend(label)
    duplicate_list = [k for k,v in Counter(sample_names).items() if v>1]
    if len(duplicate_list) > 0:
        print(action)
