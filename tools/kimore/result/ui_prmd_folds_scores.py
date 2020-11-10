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

#results_save = np.zeros([10, 7])
eval_criteria = 'cv_rd'
label_path = '/media/bruce/2T/data/UI_PRMD/st-gcn/'+ eval_criteria +'/xyz'
for index, method in enumerate(['mul']):#'xyz','ang','xyzang','cat','add','lit','mul'
    print(index,method)
    result_path = '/media/bruce/2T/data/UI_PRMD/work_st_gcn/kinect/'+ eval_criteria +'/' + method

    folds = ['1', '2', '3', '4', '5']
    actions = [1,2,3,4,5,6,7,8,9,10]
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

        results_score = []
        right_num_action = 0
        for i in range(len(labels)):
            l = labels[i][1]
            index_, r_a = results[i]
            score = r_a.tolist()
            
            r_a = np.argmax(r_a)
            right_num_action += int(r_a == int(l))

            score.insert(0,index_)
            #score = np.asarray(score)
            results_score.append(score)
        acc_action = right_num_action/len(labels)
        print("f1: {:0.4f}, f2: {:0.4f}, f3: {:0.4f}, f4: {:0.4f}, f5: {:0.4f}, action_all:{:0.4f}".format(fold_acc[0],fold_acc[1],fold_acc[2],fold_acc[3],fold_acc[4],acc_action))
        #results_save[action-1][index] = acc_action
        np.savetxt('/media/bruce/2T/data/UI_PRMD/work_st_gcn/kinect/scores_rd/'+ method +'_' + str(action) + '.csv', results_score, fmt='%s', delimiter=",", header="Index, A1, A2")
#np.savetxt('results.csv', results_save, fmt='%1.4f', delimiter=",")
