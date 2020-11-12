# python tools/ntu_gendata_hard_cases.py
import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap
import re

from ui_prmd_read import read_ang, read_xyzang, read_xyz

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 1
num_joint = 22
max_frame = 150
toolbar_width = 30

files_ = os.listdir('./data/UI_PRMD/skl_whole')

number_of_folds = 5

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")

def gendata(data_path,
            out_path,
            action,
            fold,
            benchmark='xview'):

    sample_name = []
    sample_label = []
    if action < 10:
        r = re.compile("A0"+str(action)+".*.skeleton")
    else:
        r = re.compile("A"+str(action)+".*.skeleton")
    files = list(filter(r.match, files_))
    #print('len(files):', len(files))
    #samples_count = 0
    training_list = []
    testing_list = []
    training_list_label = []
    testing_list_label = []
    #print('files len: ', len(files))
    for filename in files:
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 3])
        if action_class != action:
            print('not action',action, action_class)
            continue
        subject_id = int(
            filename[filename.find('S') + 1:filename.find('S') + 3])
        episode_id = int(
            filename[filename.find('E') + 1:filename.find('E') + 3])
        if_correct = int(
            filename[filename.find('C') + 1:filename.find('C') + 3])

        label = if_correct - 1

        mod = subject_id % number_of_folds
        istraining = False
        if int(fold) == mod + 1:
            istraining = False
        else:
            istraining = True

        if istraining:
            training_list.append(filename)
            training_list_label.append(label)
        else:
            testing_list.append(filename)
            testing_list_label.append(label)
    #print('testing_list len: ', len(testing_list))
    for part in ['train', 'eval']:
        if part == 'train':
            sample_name = training_list
            sample_label = training_list_label
        else:
            sample_name = testing_list
            sample_label = testing_list_label

        with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
            pickle.dump((sample_name, list(sample_label)), f, protocol=2)
        # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

        fp = open_memmap(
            '{}/{}_data.npy'.format(out_path, part),
            dtype='float32',
            mode='w+',
            #shape=(len(sample_label), 3, max_frame, num_joint, max_body))
            shape=(len(sample_label), 3, max_frame, num_joint, max_body))

        for i, s in enumerate(sample_name):
            print_toolbar(i * 1.0 / len(sample_label),
                          '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                              i + 1, len(sample_name), benchmark, part))
            data = read_ang(
                os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)

            fp[i, :, 0:data.shape[1], :, :] = data
        end_toolbar()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='/mnt/nas/ntu-rgbd/other_Datasets/UI_PRMD/skl_whole')
    #parser.add_argument('--out_folder', default='/media/bruce/2T/data/UI_PRMD/st-gcn/kinect/sd_1_1/pos')
    parser.add_argument('--out_folder', default='/media/bruce/2T/data/UI_PRMD/st-gcn/cv_cs/ang')
    #parser.add_argument('--out_folder', default='/media/bruce/2T/data/UI_PRMD/st-gcn/vicon/sd_1_1/ang')
    folds = ['1', '2', '3', '4', '5']
    arg = parser.parse_args()
    for act in [1,2,3,4,5,6,7,8,9,10]:
        for fold in folds:
            #for b in benchmark:
            out_path = os.path.join(arg.out_folder, str(act),fold)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            gendata(
                arg.data_path,
                out_path,
                act,
                fold,
                benchmark='c_inc'
                )

    # 1. line 118
    # 2. line 104
    # 3. line 114
