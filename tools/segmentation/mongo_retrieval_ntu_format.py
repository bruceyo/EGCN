# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:27:46 2019

@author: PQ504-DEMO
"""

# Requires pymongo 3.6.0+
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
import pandas
import math

client = MongoClient("mongodb://158.132.255.154:27017/")
database = client["test"]
collection = database["grandma_set_rgbd"]

# Load the RGB labels
rgb_labels = 'Y:/PolyU_RGBD/case_study/train/rgb_labels.csv'
df_rgb_labels = pandas.read_csv(rgb_labels)  # df_rgb_labels.iat[0,i]
action_id_adjust = 0
sample_frame_count = 0

try:
    for a in range(0,12):
        action_id = a + 1
        # Load Skeleton Data from MongoDB
        query = {}
        if action_id in [2,5]:
            action_id_adjust +=1
            query["action_id"] = action_id - action_id_adjust   
        else:
            query["action_id"] = action_id - action_id_adjust
        #query["frame_count"] = 2
        #query["count"] = 1
        
        handState = {'Unknown':'0 ','NotTracked':'1 ','Open':'2 ','Closed':'3 ','Lasso':'4 '}
        trackingState = {'NotTracked':' 0','Inferred':' 1','Tracked':' 2'}
        name_key = ""
        
        sub_id = 'S001'
        action = 'A0' + str(action_id) if action_id > 9 else 'A00'+ str(action_id)
        
        sample_lines = ""

        for i in range(0,42,2): # range(0,42,2)
            sample_id = i
            if(math.isnan(df_rgb_labels.iat[action_id-1,sample_id])):
                print(str(int((i+2)/2 - 1)) + " samples of action " + str(action_id) + " retrieved!")
                break
            
            cursor = collection.find(query)
            #sample_lines += str(df_rgb_labels.iat[action_id-1,sample_id+1] - df_rgb_labels.iat[action_id-1,sample_id] + 1) + "\n"
            for doc in cursor:
                #build name_key
                #sub_id = 'S' + sub_id if sub_id > 9 else 'S0' + sub_id
                frame_count = doc['Frame_count']
                if df_rgb_labels.iat[action_id-1,sample_id] <= frame_count <= df_rgb_labels.iat[action_id-1,sample_id+1]:
                    sample_frame_count += 1
                    # save all samples that in the label segment to the .skeleton
                    sample_lines += "1\n" #+ str(frame_count) +"\n"
                    #build body_info_key
                    lefthand = doc['skeletons'][0]['handstates'][0]['lefthand']
                    righthand = doc['skeletons'][0]['handstates'][1]['righthand']
                    leantrackingstate = doc['skeletons'][0]['lean']['leantrackingstate']
                    lefthand = handState[lefthand]
                    righthand = handState[righthand]
                    leantrackingstate = trackingState[leantrackingstate]
                    body_info_key = (
                                        sub_id + 
                                        (' 0 ' if doc['skeletons'][0]['ClippedEdge'] == 'None' else ' 1 ') + 
                                        ('0 ' if doc['skeletons'][0]['handstates'][0]['confidence'] == 'Low' else '1 ') + 
                                        lefthand  + 
                                        ('0 ' if doc['skeletons'][0]['handstates'][1]['confidence'] == 'Low' else '1 ') + 
                                        righthand  + 
                                        ('1 ' if doc['skeletons'][0]['isrestricted'] == 'True' else '0 ') + 
                                        str(doc['skeletons'][0]['lean']['x']) + ' ' + 
                                        str(doc['skeletons'][0]['lean']['y']) + 
                                        leantrackingstate
                                )
                    sample_lines += body_info_key + "\n25\n"
                     #build skeleton frame info
                    frame_skl = doc['skeletons'][0]
                    joint_lines = ''
                    for joint, joint_ori in zip(frame_skl['joints'], frame_skl['jointOrientations']):
                        joint_lines += str(joint['3d']['x']) + ' '
                        joint_lines += str(joint['3d']['y']) + ' '
                        joint_lines += str(joint['3d']['z']) + ' '
                        joint_lines += str(joint['2dDepth']['x']) + ' '
                        joint_lines += str(joint['2dDepth']['y']) + ' '
                        joint_lines += str(joint['2dColor']['x']) + ' '
                        joint_lines += str(joint['2dColor']['y']) + ' '
                        joint_lines += str(joint_ori['orientation']['w']) + ' '
                        joint_lines += str(joint_ori['orientation']['x']) + ' '
                        joint_lines += str(joint_ori['orientation']['y']) + ' '
                        joint_lines += str(joint_ori['orientation']['z'])
                        joint_lines += trackingState[joint['TrackingState']] + '\n'
                    sample_lines += joint_lines
                    
                elif frame_count > df_rgb_labels.iat[action_id-1,sample_id+1]:
                    # loop to the next segment
                    #print(">")
                    break
                else:
                    # drop the frame not in the label segment span
                    #continue
                    # print("<")
                    continue
                
            # save file to xxx.skeleton
            label_num = int((sample_id+2)/2)
            label = 'L0' + str(label_num) if label_num > 9 else 'L00'+ str(label_num)
            save_name = sub_id + action + label
            save_path = 'D:/data/PolyU_RGBD/case_study_skeleton/' + save_name
            sample_lines = str(sample_frame_count) + "\n" + sample_lines
            with open(save_path + ".skeleton", "w") as text_file:
                text_file.write(sample_lines)
                
            # Update to the next labeled frame span
            sample_lines = ""
            sample_frame_count = 0  

finally:
    client.close()

