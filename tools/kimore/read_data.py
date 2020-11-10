#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:44:17 2020

@author: bruce
"""

import pandas
import re
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance

bones = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
         (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
         (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
         (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
         (22, 23), (23, 8), (24, 25), (25, 12))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 5)

# file_ang = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Raw/JointOrientation300516_124705.csv'
# file_pos = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Raw/JointPosition300516_124705.csv'
# file_time = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Raw/TimeStamp300516_124705.csv'
# file_label = '/home/bruce/Downloads/KiMoRe/GPP/Parkinson/P_ID1/Es5/Label/ClinicalAssessment_P_ID1.xlsx'

r_pos = re.compile("JointPosition.*")
r_ang = re.compile("JointOrientation.*")
r_time = re.compile("TimeStamp.*")
P_ID = 'S_ID6'
Es = '/Es5'

file_folder = '/home/bruce/Downloads/KiMoRe/GPP/Stroke/'+ P_ID + Es
files = os.listdir(os.path.join(file_folder,'Raw'))
file_pos = os.path.join(file_folder,'Raw', list(filter(r_pos.match, files))[0])
file_ang = os.path.join(file_folder,'Raw', list(filter(r_ang.match, files))[0])
file_time = os.path.join(file_folder,'Raw', list(filter(r_time.match, files))[0])
file_label = file_folder + '/Label/ClinicalAssessment_' + P_ID + '.xlsx'

joint_position = pandas.read_csv(file_pos,header = None)
joint_angular = pandas.read_csv(file_ang,header = None)
time_stamp = pandas.read_csv(file_time,header = None)
label = pandas.read_excel(file_label, index_col=0)

for f in range(0, joint_position.shape[0], 50):
    for i in range(0,25):
        x = joint_position.iloc[f][4*i]
        y = joint_position.iloc[f][4*i+1]
        z = joint_position.iloc[f][4*i+2]
        if i==3:
            ax.scatter(z, x, y, color='#000000')
        else:
            ax.scatter(z, x, y, color='#40ff00')

# retrieve feature for segmentation
dist=[]
joint_1 = 3 # index of head joint
joint_2 = 19 # index of right_foot joint
for f in range(0, joint_position.shape[0]):
    dist.append(distance.euclidean((joint_position.iloc[f][4*joint_1],joint_position.iloc[f][4*joint_1+1],joint_position.iloc[f][4*joint_1+2]),
                (joint_position.iloc[f][4*joint_2],joint_position.iloc[f][4*joint_2+1],joint_position.iloc[f][4*joint_2+2])))

'''
for i in range(0,25): # plot each point + it's index as text above
    x = xs[i+p*25] - xs[0]
    y = ys[i+p*25] - ys[0]
    z = zs[i+p*25] - zs[0]
    ax.scatter(z, x, y, color='#40ff00')
'''
np.savetxt(os.path.join('.','dist.csv'), dist, fmt='%0.6f', delimiter=",")
