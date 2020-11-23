#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:55:31 2020

@author: xxx
"""

import pickle
import os
import numpy as np
from collections import Counter
import shutil

EGCN_strategies = {
'xyz':'Single Modal Position',
'ang':'Single Modal Orientation',
'xyzang':'MultiModal SLE',
'cat':'MultiModal FLE-1',
'lit':'MultiModal FLE-2',
'add':'MultiModal DLE-1',
'add_1':'MultiModal DLE-2',
'mul':'MultiModal MLE'
}

kimore_exercises = {
2:'Es1,   ',
3:'Es2(L),',
4:'Es2(R),',
5:'Es3(L),',
6:'Es3(R),',
7:'Es4(L),',
8:'Es4(R),',
9:'Es5,   ',
}

eval_criterias = ['cv_cs','cv_rd']
for eval_criteria in eval_criterias:

    label_path = './data/KiMoRe/'+ eval_criteria +'/xyz'
    for index, method in enumerate(['xyz','ang','xyzang','cat','lit','add','add_1','mul']):#  'xyz','ang','xyzang','cat','lit','add','add_1','mul'
        print('\nThe cross-validation ('+eval_criteria+') results of ' + EGCN_strategies[method] + ':')

        if method != 'add_1':

            result_path = './work_dir/kimore/'+ eval_criteria +'/' + method

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
                    if not os.path.exists(os.path.join(result_path,str(action),fold, 'result.pkl')):
                        fold_acc.append(0)
                        #print('no result: ', os.path.join(result_path,str(action),fold))
                        continue
                    r1 = open(os.path.join(result_path,str(action),fold, 'result.pkl'), 'rb')

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

                right_num_action = 0
                for i in range(len(labels)):
                    l = labels[i][1]
                    _, r_a = results[i]

                    r_a = np.argmax(r_a)
                    right_num_action += int(r_a == int(l))

                acc_action = right_num_action/len(labels)
                print("Exercise "+kimore_exercises[action]+" f1: {:0.4f}, f2: {:0.4f}, f3: {:0.4f}, f4: {:0.4f}, f5: {:0.4f}, all:{:0.4f}".format(fold_acc[0],fold_acc[1],fold_acc[2],fold_acc[3],fold_acc[4],acc_action))

        else:
            result_path_xyz = './work_dir/kimore/'+ eval_criteria +'/xyz'
            result_path_ang = './work_dir/kimore/'+ eval_criteria +'/ang'

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
                    if not os.path.exists(os.path.join(result_path_xyz,str(action),fold, 'result.pkl')):
                        fold_acc.append(0)
                        #print('no result: ', os.path.join(result_path,str(action),fold))
                        continue
                    r1 = open(os.path.join(result_path_xyz,str(action),fold, 'result.pkl'), 'rb')
                    r2 = open(os.path.join(result_path_ang,str(action),fold, 'result.pkl'), 'rb')
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
                print("Exercise "+kimore_exercises[action]+" f1: {:0.4f}, f2: {:0.4f}, f3: {:0.4f}, f4: {:0.4f}, f5: {:0.4f}, all:{:0.4f}".format(fold_acc[0],fold_acc[1],fold_acc[2],fold_acc[3],fold_acc[4],acc_action))
