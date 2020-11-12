# python tools/ntu_gendata_hard_cases.py
import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from ui_prmd_read import read_ang, read_xyzang, read_xyz

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 1
num_joint = 22
max_frame = 150
toolbar_width = 30

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
            action=1,
            benchmark='xview',
            part='eval'):

    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):

        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 3])

        #if action_class != action:
        #    continue

        subject_id = int(
            filename[filename.find('S') + 1:filename.find('S') + 3])
        episode_id = int(
            filename[filename.find('E') + 1:filename.find('E') + 3])
        if_correct = int(
            filename[filename.find('C') + 1:filename.find('C') + 3])

        ''' 7_3 train
        if episode_id in [9,10]:
            continue
        if subject_id not in [9,10] and episode_id in [8]:
            continue
        #'''

        '''7_3 eval
        if episode_id not in [8,9,10]:
            continue
        if subject_id in [9,10] and episode_id in [8]:
            continue
        #'''

        sample_name.append(filename)
        sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f, protocol=2)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))
        #shape=(len(sample_label), 6, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='./data/UI_PRMD/skl_whole')
    #parser.add_argument('--out_folder', default='/media/bruce/2T/data/UI_PRMD/st-gcn/kinect/sd_1_1/pos')
    parser.add_argument('--out_folder', default='./data/UI_PRMD/st-gcn/kinect/cls/pos_all_4_cv')
    #parser.add_argument('--out_folder', default='/media/bruce/2T/data/UI_PRMD/st-gcn/vicon/sd_1_1/ang')

    #benchmark = ['xsub', 'xview']
    part = ['train']#eval
    arg = parser.parse_args()
    #for act in [1,2,3,4,5,6,7,8,9,10]:
        #for b in benchmark:
    for p in part:
        out_path = os.path.join(arg.out_folder)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        gendata(
            arg.data_path,
            out_path,
            1,
            benchmark='c_inc',
            part=p)

    # 1. line 118
    # 2. line 104
    # 3. line 114
