#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:55:31 2020

@author: bruce
"""

import pickle
import os
import numpy as np
from collections import Counter

results_save = np.zeros([8, 2])
for index, eval_criteria in enumerate(['cv_cs','cv_rd']):
    label_path = '/media/bruce/2T/data/KiMoRe/'+ eval_criteria +'/xyz'

    result_path_xyz = '/media/bruce/2T/data/KiMoRe/work_st_gcn/kinect/'+ eval_criteria +'/xyz'
    result_path_ang = '/media/bruce/2T/data/KiMoRe/work_st_gcn/kinect/'+ eval_criteria +'/ang'

    folds = ['1', '2', '3', '4', '5']
    actions = [2,3,4,5,6,7,8,9]
    for action in actions:
        labels = []
        results_1 = []
        results_2 = []
        fold_acc = []
        for fold in folds:
            label = open(os.path.join(label_path,str(action),fold, 'eval_label.pkl'), 'rb')
            label = np.array(pickle.load(label))
            label = label.T.tolist()
            if not os.path.exists(os.path.join(result_path_xyz,str(action),fold, 'best_result.pkl')):
                fold_acc.append(0)
                #print('no result: ', os.path.join(result_path,str(action),fold))
                continue
            r1 = open(os.path.join(result_path_xyz,str(action),fold, 'best_result.pkl'), 'rb')
            r2 = open(os.path.join(result_path_ang,str(action),fold, 'best_result.pkl'), 'rb')
            r1 = list(pickle.load(r1).items())
            r2 = list(pickle.load(r2).items())
            right_num_11 = 0
            for i in range(len(label)):
                l = label[i][1]
                _, r11 = r1[i]
                _, r22 = r2[i]

                r11 = r11 + r22
                r11 = np.argmax(r11)
                right_num_11 += int(r11 == int(l))

            acc = right_num_11/len(label)
            fold_acc.append(acc)

            labels.extend(label)
            results_1.extend(r1)
            results_2.extend(r2)

        right_num_action = 0
        for i in range(len(labels)):
            l = labels[i][1]
            _, r_a_1 = results_1[i]
            _, r_a_2 = results_2[i]
            r_a_1 = r_a_1 + r_a_2
            r_a_1 = np.argmax(r_a_1)
            right_num_action += int(r_a_1 == int(l))

        acc_action = right_num_action/len(labels)
        print("f1: {:0.4f}, f2: {:0.4f}, f3: {:0.4f}, f4: {:0.4f}, f5: {:0.4f}, action_all:{:0.4f}".format(fold_acc[0],fold_acc[1],fold_acc[2],fold_acc[3],fold_acc[4],acc_action))
        results_save[action-2][index] = acc_action

np.savetxt('results_add.csv', results_save, fmt='%1.4f', delimiter=",")
