import numpy as np
import os
import scipy.io as sio

# 假设你有9个被试
subject_indices = range(1, 10)

PAT = 'T'

for subject_index in subject_indices:
    # 构建文件名
    npy_file = f'D:/EEG_dataset/BCICIV_2a_npy/A0{subject_index}{PAT}.npy'
    mat_file = f'D:/EEG_dataset/BCICIV_2a_gdf/true_labels/A0{subject_index}{PAT}.mat'

    # 读取.npy文件
    data = np.load(npy_file)

    # 读取.mat文件中的标签
    mat_data = sio.loadmat(mat_file)
    label = mat_data # 假设.mat文件中有一个名为'label'的键

    # 创建字典
    data_dict = {'data': data, 'label': label}

    # 保存为.mat文件
    save_path = f'D:/EEG_dataset/BCICIV_2a_mat/A0{subject_index}{PAT}.mat'
    sio.savemat(save_path, data_dict)

    print(f'Saved {save_path}')
