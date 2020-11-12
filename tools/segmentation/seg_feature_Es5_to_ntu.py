#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:44:17 2020

@author: bruce
"""

import pandas as pd
import re
import os
import math
# file_ang = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Raw/JointOrientation300516_124705.csv'
# file_pos = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Raw/JointPosition300516_124705.csv'
# file_time = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Raw/TimeStamp300516_124705.csv'
# file_label = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Label/ClinicalAssessment_P_ID1.xlsx'
debug = False
r_pos = re.compile("JointPosition.*")
r_ang = re.compile("JointOrientation.*")
r_time = re.compile("TimeStamp.*")

group_dic = {
    "CG/Expert/E_ID":17,
    "CG/NotExpert/NE_ID":27,
    "GPP/BackPain/B_ID":8,
    "GPP/Parkinson/P_ID":16,
    "GPP/Stroke/S_ID":10,
}
group_label_dic = {
    "CG/Expert/E_ID": 'G001',
    "CG/NotExpert/NE_ID": 'G002',
    "GPP/BackPain/B_ID": 'G003',
    "GPP/Parkinson/P_ID": 'G004',
    "GPP/Stroke/S_ID": 'G005',
}

Es = '/Es5'

seg_labels = 'Es5_labels.csv'
seg_labels = pd.read_csv(seg_labels, header=None, index_col=0)


for group, ids in group_dic.items():
    for id in range(1,ids+1):

        group_id = group + str(id)
        if group_id in seg_labels.index:
            labels = seg_labels.loc[group_id]
        else:
            continue

        file_folder = '/home/bruce/Downloads/KiMoRe/'+ group_id + Es
        files = os.listdir(os.path.join(file_folder,'Raw'))
        if len(list(filter(r_pos.match, files)))<1:
            print('pos not exist: ',group_id+Es)
            continue
        file_pos = os.path.join(file_folder,'Raw', list(filter(r_pos.match, files))[0])
        if len(list(filter(r_ang.match, files)))<1:
            print('ang not exist: ',group_id+Es)
            continue
        file_ang = os.path.join(file_folder,'Raw', list(filter(r_ang.match, files))[0])
        file_time = os.path.join(file_folder,'Raw', list(filter(r_time.match, files))[0])
        #file_label = file_folder + '/Label/ClinicalAssessment_' + P_ID + '.xlsx'

        joint_position = pd.read_csv(file_pos,header = None)
        joint_angular = pd.read_csv(file_ang,header = None)
        time_stamp = pd.read_csv(file_time,header = None)
        #label = pd.read_excel(file_label, index_col=0)

        for i in range(0,14,2): # range(0,42,2)
            sample_id = i
            if(math.isnan(labels.iat[sample_id])):
                print(str(int((i+2)/2 - 1)) + " samples of action " + group_id + " retrieved!")
                break

            sample_lines = ""
            #print(labels.iat[sample_id], labels.iat[sample_id+1])
            sample_frame_count = int(labels.iat[sample_id+1]) - int(labels.iat[sample_id]) + 1
            for f_id in range(int(labels.iat[sample_id])-1, int(labels.iat[sample_id+1])):
                sample_lines += "1\n" #+ str(frame_count) +"\n"
                #build body_info_key
                body_info_key = str(time_stamp.iat[f_id, 0]) + ' 0 0 0 1 1 1 2'
                sample_lines += body_info_key + "\n25\n"
                 #build skeleton frame info
                joint_lines = ''
                for joint in range(0,25):
                    joint_lines += str(round(joint_position.iloc[f_id][4*joint],8)) + ' ' # x
                    joint_lines += str(round(joint_position.iloc[f_id][4*joint+1],8)) + ' ' # y
                    joint_lines += str(round(joint_position.iloc[f_id][4*joint+2],8)) + ' ' # z
                    joint_lines += str(joint_position.iloc[f_id][4*joint+3]) + ' '
                    joint_lines += str(round(joint_angular.iloc[f_id][4*joint],8)) + ' '
                    joint_lines += str(round(joint_angular.iloc[f_id][4*joint+1],8)) + ' '
                    joint_lines += str(round(joint_angular.iloc[f_id][4*joint+2],8)) + ' '
                    joint_lines += str(round(joint_angular.iloc[f_id][4*joint+3],8)) + '\n'

                sample_lines += joint_lines

            # save file to xxx.skeleton
            label_num = int((sample_id+2)/2)
            label = 'R0' + str(label_num) if label_num > 9 else 'R00'+ str(label_num)
            subject_id = 'S0' + str(id) if id > 9 else 'S00'+ str(id)
            save_name = group_label_dic[group] + subject_id + 'E009' + label
            save_path = '/media/bruce/2T/data/KiMoRe/skeleton/' + save_name
            sample_lines = str(sample_frame_count) + "\n" + sample_lines
            with open(save_path + ".skeleton", "w") as text_file:
                text_file.write(sample_lines)

            # Update to the next labeled frame span
            sample_lines = ""
            sample_frame_count = 0

            if debug:
                break
        if debug:
            break
    if debug:
        break
