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

def read_xy(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((2, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            b_to_xy = xyz_to_xy(b['jointInfo'])
            for j, v in enumerate(b_to_xy):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y']]
                else:
                    pass
    return data

def xyz_to_xy(jointInfo):
    #frame = np.array([0.22039,	0.25156,	0.28316,	0.23403,	0.12856,	0.10396,	0.13252,	0.19971,	0.34047,	0.24075,	0.17264,	0.16746,	0.16678,	0.20114,	0.24125,	0.189,	0.27047,	0.32045,	0.34952,	0.30343,	0.27528,	0.20348,	0.19339,	0.12768,	0.11403,	0.16784,	0.44405,	0.71587,	0.83867,	0.59848,	0.38757,	0.34519,	0.30468,	0.58881,	0.53316,	0.70089,	0.76815,	0.17231,	-0.13684,	-0.41342,	-0.47015,	0.16014,	-0.12052,	-0.43461,	-0.4978,	0.64847,	0.29061,	0.28534,	0.80711,	0.76011,	3.7888,	3.7114,	3.6236,	3.5884,	3.7016,	3.8219,	3.6066,	3.6158,	3.5615,	3.3892,	3.407,	3.4304,	3.7867,	3.9053,	4.0533,	3.9608,	3.727,	3.7912,	3.9496,	3.9696,	3.6473,	3.6045,	3.6,	3.4339,	3.444])

    frame = np.zeros((1,75))
    joint_count = 0
    for i,v in enumerate(jointInfo):
        joint_count = joint_count +1
        frame[0,i] = v['x']
        frame[0,i+25] = v['y']
        frame[0,i+50] = v['z']

    #P1 = np.array([0.2204,0.1678,3.7888])
    #P2 = np.array([0.3405,0.5888,3.5615])
    #P3 = np.array([0.1286,0.5985,3.7016])
    # Calculate the plane
    P1 = np.array([frame[0,0],frame[0,25],frame[0,50]])
    P2 = np.array([frame[0,8],frame[0,8+25],frame[0,8+50]])
    P3 = np.array([frame[0,4],frame[0,4+25],frame[0,4+50]])
    normal = np.cross(P1-P2, P1-P3)
    unit_normal = normal/np.linalg.norm(normal)

    frame_plane = np.zeros((1,75))
    for i in range(25):
        # https://stackoverflow.com/questions/8942950/how-do-i-find-the-orthogonal-projection-of-a-point-onto-a-plane/8944143
        q = np.array([frame[0,i],frame[0,i+25],frame[0,i+50]])
        q_proj = q - np.dot(q - P1, unit_normal) * unit_normal
        frame_plane[0,i] = q_proj[0]
        frame_plane[0,i+25] = q_proj[1]
        frame_plane[0,i+50] = q_proj[2]

    #%{ reference: https://www.mathworks.com/matlabcentral/answers/81694-rotate-3d-plane-to-a-new-2d-coordinate-system

    Coor_3D = np.array([frame_plane[0,0:25].T, frame_plane[0,25:50].T, frame_plane[0,50:75].T])
    #assume points are Nx3, where N is number of points.
    #let first point origin in 2D coordinate system (can be shifted later)
    #Calculate out-of-plane vector (local z)
    N = Coor_3D.shape[1]
    #origin = (Coor_3D(9,:) - Coor_3D(5,:))/2;
    origin = Coor_3D[:,0]
    localz = np.cross(Coor_3D[:,4]-origin, Coor_3D[:,8]-origin)
    #normalize it
    unitz = localz/np.linalg.norm(localz,2)
    #calculate local x vector in plane
    #localx = Coor_3D(2,:)-origin;
    localx = Coor_3D[:,20]-Coor_3D[:,0]
    unitx = localx/np.linalg.norm(localx,2)
    #calculate local y
    localy = np.cross(localz, localx)
    unity = localy/np.linalg.norm(localy,2)
    #assume transformation matrix
    T = np.transpose(np.append(np.array([unitx[:], unity[:], unitz[:], origin[:]]), [[0], [0], [0], [1]],axis=1))
    C = np.append(np.transpose(Coor_3D), np.ones((N,1), dtype=np.float64), axis=1)
    Coor_2D = np.linalg.solve(T, np.transpose(C))
    Coor_2D = np.transpose(Coor_2D[0:3,:])

    twoDpnts  = Coor_2D[:,0:2]

    skl_2d = np.around(twoDpnts*100) + 1000
    j = np.argsort([1,0])
    skl_2d = skl_2d[:,j]

    joints = []
    for x,y in skl_2d:
        joint_info_key = ['x', 'y']
        joint_info = {
            k: float(v)
            for k, v in zip(joint_info_key, [x,y])
        }
        joints.append(joint_info)

    return joints
