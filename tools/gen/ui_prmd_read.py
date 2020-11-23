import numpy as np
import os

def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z',
                        'ang_x', 'ang_y', 'ang_z',
                        'ab_x', 'ab_y', 'ab_z'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    if seq_info['numFrame'] > 150:
        #https://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
        f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        sample_indexs = f(150, seq_info['numFrame'])
        seq_info['frameInfo'] = [f for n, f in enumerate(seq_info['frameInfo']) if n in sample_indexs]#range(start,end)]
        seq_info['numFrame'] = 150
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data

def read_ang(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    if seq_info['numFrame'] > 150:
        #https://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
        f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        sample_indexs = f(150, seq_info['numFrame'])
        seq_info['frameInfo'] = [f for n, f in enumerate(seq_info['frameInfo']) if n in sample_indexs]#range(start,end)]
        seq_info['numFrame'] = 150
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['ang_x'], v['ang_y'], v['ang_z']]
                else:
                    pass
    return data

def read_xyzang(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    if seq_info['numFrame'] > 150:
        #https://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
        f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        sample_indexs = f(150, seq_info['numFrame'])
        seq_info['frameInfo'] = [f for n, f in enumerate(seq_info['frameInfo']) if n in sample_indexs]#range(start,end)]
        seq_info['numFrame'] = 150
    data = np.zeros((6, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z'], v['ang_x'], v['ang_y'], v['ang_z']]
                else:
                    pass
    return data
