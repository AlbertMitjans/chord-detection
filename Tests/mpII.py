import os
import re
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import shutil
import copy

directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'data/mpii/mpii_human_pose_v1_u12_1.mat'))

directory2 = os.path.abspath(os.path.join(os.getcwd(), '..', 'data/mpii/images'))

file = sio.loadmat(directory)['RELEASE']['annolist'][0][0][0]

for i in range(file.shape[0]):
    print(i)
    name = os.path.splitext(file['image'][i][0][0][0][0])[0]
    try:
        coord = file['annorect'][i]['annopoints']
    except (ValueError, IndexError):
        print('NO DATA')
        continue
    joints = {}
    final_joints = {}
    for j in range(coord.shape[1]):
        try:
            all_coord = coord[0][j]['point'][0][0][0]
        except IndexError:
            continue
        num = final_joints.__len__()
        final_joints.setdefault('joint{num}'.format(num=num), []).append([[-1, -1] for i in range(16)])
        final_joints['joint{num}'.format(num=num)] = final_joints['joint{num}'.format(num=num)][0]
        num2 = joints.__len__()
        for joint in range(all_coord.shape[0]):
            joints.setdefault('joint{num}'.format(num=num2), []).append([all_coord[joint][0][0][0], all_coord[joint][1][0][0], all_coord[joint][2][0][0]])

    if joints['joint0'][0][0] == min(joints['joint0'][0]):
        for y in range(joints.__len__()):
            for x in range(joints['joint{num}'.format(num=y)].__len__()):
                order = [1, 2, 0]
                joints['joint{num}'.format(num=y)][x] = [joints['joint{num}'.format(num=y)][x][i] for i in order]

    for joint in range(joints.__len__()):
        for x, y, i in joints['joint{num}'.format(num=joint)]:
            final_joints['joint{num}'.format(num=joint)][i] = [x, y]

        np.savetxt(os.path.join(directory2, '{top1}_{person}.csv'.format(top1=name, person=joint)), np.asarray(final_joints['joint{num}'.format(num=joint)]), delimiter=',',
               fmt='%.3f')


