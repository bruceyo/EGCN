# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data_pos = np.load(self.data_path, mmap_mode='r')
            self.data_ang = np.load(self.data_path.replace('pos','ang'), mmap_mode='r')
            print("################self.data: " +str(len(self.data_ang)))#107
        else:
            self.data_pos = np.load(self.data_path)
            self.data_ang = np.load(self.self.data_path.replace('pos','ang'))

        if self.debug:
            self.label = self.label[0:100]
            self.data_pos = self.data_pos[0:100]
            self.data_ang = self.data_ang[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data_pos.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy_pos = np.array(self.data_pos[index])
        data_numpy_ang = np.array(self.data_ang[index])
        label = self.label[index]

        # processing
        if self.random_choose:
            data_numpy_pos = tools.random_choose(data_numpy_pos, self.window_size)
            data_numpy_ang = tools.random_choose(data_numpy_ang, self.window_size)
        elif self.window_size > 0:
            data_numpy_pos = tools.auto_pading(data_numpy_pos, self.window_size)
            data_numpy_ang = tools.auto_pading(data_numpy_ang, self.window_size)
        if self.random_move:
            data_numpy_pos = tools.random_move(data_numpy_pos)
            data_numpy_ang = tools.random_move(data_numpy_ang)

        # to do centralization  --bruce
        #print('????????????')
        return data_numpy_pos, data_numpy_ang, label
