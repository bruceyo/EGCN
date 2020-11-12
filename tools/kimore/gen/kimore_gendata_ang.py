# python tools/ntu_gendata_hard_cases.py
import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap
import re

from kimore_read import read_ang, read_xyzang, read_xyz

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 1
num_joint = 25
max_frame = 150
toolbar_width = 30


files_ = os.listdir('./data/KiMoRe/skeleton')

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
            benchmark='xview'):

    sample_name = []
    sample_label = []

    r = re.compile(".*"+"E00"+str(action)+".*.skeleton")
    files = list(filter(r.match, files_))
    #print('len(files):', len(files))
    #samples_count = 0
    training_list = []
    testing_list = []
    training_list_label = []
    testing_list_label = []
    for group in ['G001', 'G003', 'G004', 'G005']:
        #r = re.compile(group + ".*.skeleton")
        #files_g = list(filter(r.match, files))
        if group == 'G001':
            label = 1
        else:
            label = 0
        for subject in range(1,28):
            sub_str = 'S00' + str(subject) if subject < 10 else 'S0' + str(subject)
            r = re.compile(group + sub_str + ".*.skeleton")
            files_g_s = list(filter(r.match, files))
            files_g_s.sort()

            if len(files_g_s) == 0:
                continue

            #samples_count = samples_count + len(files_g_s)
            testing_split = round(len(files_g_s) * 3.0/10)
            training_split = round(len(files_g_s) * 7.0/10)
            if training_split + testing_split > len(files_g_s):
                training_split = training_split - 1

            for i in range(0, len(files_g_s)):
                if i < training_split:
                    training_list.append(files_g_s[i])
                    training_list_label.append(label)
                else:
                    testing_list.append(files_g_s[i])
                    testing_list_label.append(label)

        #print('samples_count: ', samples_count)
        #print('training_list len:', len(training_list))
        #print('testing_split len:', len(testing_list))
        #break
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
            shape=(len(sample_label), 6, max_frame, num_joint, max_body))

        for i, s in enumerate(sample_name):
            print_toolbar(i * 1.0 / len(sample_label),
                          '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                              i + 1, len(sample_name), benchmark, part))
            data = read_xyzang(
                os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)

            fp[i, :, 0:data.shape[1], :, :] = data
        end_toolbar()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='/media/bruce/2T/data/KiMoRe/skeleton')
    #parser.add_argument('--out_folder', default='/media/bruce/2T/data/UI_PRMD/st-gcn/kinect/sd_1_1/pos')
    parser.add_argument('--out_folder', default='/media/bruce/2T/data/KiMoRe/pre_7_3/xyzang')
    #parser.add_argument('--out_folder', default='/media/bruce/2T/data/UI_PRMD/st-gcn/vicon/sd_1_1/ang')

    arg = parser.parse_args()
    for act in [2,3,4,5,6,7,8,9]:
        #for b in benchmark:
        out_path = os.path.join(arg.out_folder, str(act))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        gendata(
            arg.data_path,
            out_path,
            act,
            benchmark='c_inc'
            )

    # 1. line 118
    # 2. line 104
    # 3. line 114
