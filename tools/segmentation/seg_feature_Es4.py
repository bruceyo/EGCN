#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:44:17 2020

@author: bruce
"""

import pandas as pd
import re
import os
from scipy.spatial import distance
import csv
import math
import numpy as np

# file_ang = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Raw/JointOrientation300516_124705.csv'
# file_pos = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Raw/JointPosition300516_124705.csv'
# file_time = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Raw/TimeStamp300516_124705.csv'
# file_label = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Label/ClinicalAssessment_P_ID1.xlsx'

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
Es = '/Es4'

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

dists = [] #pd.DataFrame()
for group, ids in group_dic.items():
    for id in range(1,ids+1):
#Group='GPP' #['GPP','CG/Expert']
#P_ID = 'P_ID1' #['E_ID1-17','NE_ID1-27','B_ID1-8','P_ID1-16','S_ID1-10']
        group_id = group + str(id)

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

        # retrieve feature for segmentation
        dist=[]
        joint_1 = 3 # index of head joint
        joint_2 = 19 # index of right_foot joint
        joint_3 = 15 # index of right_foot joint
        direction = []
        for f in range(0, joint_position.shape[0]):
            line_1_4 = np.array([joint_position.iloc[f][4*joint_1],joint_position.iloc[f][4*joint_1+1],joint_position.iloc[f][4*joint_1+2]]) - \
                       np.array([joint_position.iloc[f][4*0],joint_position.iloc[f][4*0+1],joint_position.iloc[f][4*0+2]])
            foot_middle = np.array([joint_position.iloc[f][4*joint_2],joint_position.iloc[f][4*joint_2+1],joint_position.iloc[f][4*joint_2+2]])/2 + \
                           np.array([joint_position.iloc[f][4*joint_3],joint_position.iloc[f][4*joint_3+1],joint_position.iloc[f][4*joint_3+2]])/2
            line_1_f = foot_middle - np.array([joint_position.iloc[f][4*0],joint_position.iloc[f][4*0+1],joint_position.iloc[f][4*0+2]])

            dist.append(angle(line_1_4, line_1_f))
            if joint_position.iloc[f][4*joint_1] + foot_middle[0] - 2*joint_position.iloc[f][4*0] > 0:
                direction.append(1)
            else:
                direction.append(0)
        #dists[group_id] = dist
        dist.insert(0, group_id)
        direction.insert(0, group_id)
        dists.append(dist)
        dists.append(direction)

        #break
    #break
#dists.to_csv('dists.csv', index=False)
#np.savetxt(os.path.join('.','dist.csv'), dist, fmt='%0.6f', delimiter=",")
with open('dists_Es4.csv', 'w', newline='') as f:
    writer=csv.writer(f)
    for dist in dists:
        writer.writerow(dist)
