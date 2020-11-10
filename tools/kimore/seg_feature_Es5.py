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
Es = '/Es5'

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
        for f in range(0, joint_position.shape[0]):
            dist.append(distance.euclidean((joint_position.iloc[f][4*joint_1],joint_position.iloc[f][4*joint_1+1],joint_position.iloc[f][4*joint_1+2]),
                        (joint_position.iloc[f][4*joint_2],joint_position.iloc[f][4*joint_2+1],joint_position.iloc[f][4*joint_2+2])))

        #dists[group_id] = dist
        dist.insert(0, group_id)
        dists.append(dist)

#dists.to_csv('dists.csv', index=False)
#np.savetxt(os.path.join('.','dist.csv'), dist, fmt='%0.6f', delimiter=",")
with open('dists.csv', 'w', newline='') as f:
    writer=csv.writer(f)
    for dist in dists:
        writer.writerow(dist)
