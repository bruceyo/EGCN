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

results_save = np.zeros([8, 10])

eval_criterias = ['cv_cs','cv_rd']
for eval_criteria in eval_criterias:
    label_path = '/media/bruce/2T/data/KiMoRe/'+ eval_criteria +'/xyz'
    for index, method in enumerate(['xyz']):#  'xyz','ang','xyzang','cat','add','lit','mul'
        print(index,method)
        result_path = '/media/bruce/2T/data/KiMoRe/work_st_gcn/kinect/'+ eval_criteria +'/' + method

        folds = ['1', '2', '3', '4', '5']
        actions = [2,3,4,5,6,7,8,9]
        for action in actions:
            labels = []
            results = []
            fold_acc = []
            for fold in folds:
                label = open(os.path.join(label_path,str(action),fold, 'eval_label.pkl'), 'rb')
                label = np.array(pickle.load(label))
                label = label.T.tolist()
                if not os.path.exists(os.path.join(result_path,str(action),fold, 'best_result.pkl')):
                    fold_acc.append(0)
                    #print('no result: ', os.path.join(result_path,str(action),fold))
                    continue
                r1 = open(os.path.join(result_path,str(action),fold, 'best_result.pkl'), 'rb')
                r1 = list(pickle.load(r1).items())
                right_num_11 = 0
                for i in range(len(label)):
                    l = label[i][1]
                    _, r11 = r1[i]

                    r11 = np.argmax(r11)
                    right_num_11 += int(r11 == int(l))

                acc = right_num_11/len(label)
                fold_acc.append(acc)

                labels.extend(label)
                results.extend(r1)
                if eval_criteria == 'cv_cs':
                    results_save[action-2][int(fold)-1] = len(r1)
                else:
                    results_save[action-2][int(fold)-1 + 5] = len(r1)

            right_num_action = 0
            for i in range(len(labels)):
                l = labels[i][1]
                _, r_a = results[i]

                r_a = np.argmax(r_a)
                right_num_action += int(r_a == int(l))

            acc_action = right_num_action/len(labels)
            print("f1: {:0.4f}, f2: {:0.4f}, f3: {:0.4f}, f4: {:0.4f}, f5: {:0.4f}, action_all:{:0.4f}".format(fold_acc[0],fold_acc[1],fold_acc[2],fold_acc[3],fold_acc[4],acc_action))
            #results_save[action-2][index] = acc_action

    np.savetxt('kimore_folds_statistics.csv', results_save, fmt='%d', delimiter=",")
