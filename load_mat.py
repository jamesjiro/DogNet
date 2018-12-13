import numpy as np
import scipy.io as sio
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
test_mat = sio.loadmat(os.path.join(dir_path, 'data/lists/test_list.mat'))
train_mat = sio.loadmat(os.path.join(dir_path, 'data/lists/train_list.mat'))

# f = h5py.File(os.path.join(dir_path, 'lists/test_list.mat'), 'r')
# data = f.get('data/variable1')
# data = np.array(data)
print(test_mat)
# print(len(test_mat['file_list']))
# print(len(train_mat['file_list']))
# print(len(test_mat['file_list']) + len(train_mat['file_list']))
# for file in test_mat['file_list']:
#     print(file[0][0])
