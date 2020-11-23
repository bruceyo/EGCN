import os
import sys
import pickle
import argparse
import numpy as np
from numpy.lib.format import open_memmap
import re
from kimore_read import read_ang, read_xyzang, read_xyz

max_body = 1
num_joint = 25
max_frame = 150
toolbar_width = 30

files_ = os.listdir('./data/KiMoRe/skeleton')

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
            feature='both',
            benchmark='xview'):

    sample_name = []
    sample_label = []

    r = re.compile(".*"+"E00"+str(action)+".*.skeleton")
    files = list(filter(r.match, files_))
    training_list = []
    testing_list = []
    training_list_label = []
    testing_list_label = []
    for group in ['G001', 'G003', 'G004', 'G005']:
        if group == 'G001':
            label = 1
        else:
            label = 0
        for subject in range(1,28):
            sub_str = 'S00' + str(subject) if subject < 10 else 'S0' + str(subject)
            r = re.compile(group + sub_str + ".*.skeleton")
            files_g_s = list(filter(r.match, files))
            if len(files_g_s) == 0:
                continue
            files_g_s.sort()

            for i in range(0, len(files_g_s)):
                mod = (i+1) % number_of_folds
                istraining = False
                if int(fold) == mod + 1:
                    istraining = False
                else:
                    istraining = True
                if istraining:
                    training_list.append(files_g_s[i])
                    training_list_label.append(label)
                else:
                    testing_list.append(files_g_s[i])
                    testing_list_label.append(label)

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

        num_channel = 3
        if feature=='both':
            num_channel = 6

        fp = open_memmap(
            '{}/{}_data.npy'.format(out_path, part),
            dtype='float32',
            mode='w+',
            shape=(len(sample_label), num_channel, max_frame, num_joint, max_body))

        for i, s in enumerate(sample_name):
            print_toolbar(i * 1.0 / len(sample_label),
                          '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                              i + 1, len(sample_name), benchmark, part))
            if feature=='position':
                data = read_xyz(
                    os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            elif feature=='angle':
                data = read_ang(
                    os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            else:
                data = read_xyzang(
                    os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)

            fp[i, :, 0:data.shape[1], :, :] = data
        end_toolbar()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='./data/KiMoRe/skeleton', help='the path of the skeleton data')
    parser.add_argument('--joint_feature', default='angle', choices=['angle','position','both'], help='the feature of the skeleton data')
    parser.add_argument('--out_folder', default='./data/KiMoRe/cv_rd/ang')

    folds = ['1', '2', '3', '4', '5']
    arg = parser.parse_args()
    for act in [2,3,4,5,6,7,8,9]:
        for fold in folds:
            out_path = os.path.join(arg.out_folder, str(act),fold)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            gendata(
                arg.data_path,
                out_path,
                act,
                fold,
                arg.joint_feature,
                benchmark='cv_rd'
                )
