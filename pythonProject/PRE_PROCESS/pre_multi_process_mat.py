import mne
import scipy.io
import numpy as np
from scipy.signal import cheby2, filtfilt

def rd_raw(subject_index, session_type):

    dir_2 = f'D:/EEG_dataset/BCICIV_2a_npy/A0{subject_index}{session_type}.npy'
    labeldir_2 = f'D:/EEG_dataset/BCICIV_2a_gdf/true_labels/A0{subject_index}{session_type}.mat'
    data = np.load(dir_2)
    labels = scipy.io.loadmat(labeldir_2)
    return data, labels

def mat_merge(session_type):

    for subject_index in range(1, 10):
        data, labels = rd_raw(subject_index, session_type)
        print("labels.shape is ")
        print(len(labels))
        combined_data = {
            'data': data,
            'labels': labels['classlabel']  # 假设标签在 .mat 文件中存储为 'true_labels'
        }
        # 保存到新的 .mat 文件
        scipy.io.savemat(f'D:/EEG_dataset/BCICIV_2a_mat/A0{subject_index}{session_type}.mat', combined_data)

def main():
    session_type = 'E'
    mat_merge(session_type)

if __name__ == "__main__":
    main()


# 读取GDF文件
#
# s = raw.get_data().T
# HDR = raw.info
#
# # 读取标签
# # label = HDR['events'][1]
# mat = scipy.io.loadmat(labeldir_2)
# label_2 = mat.flatten()



# 构建样本数据 1000*22*288
# Pos = HDR['events'][0][:, 0]
# Typ = HDR['events'][0][:, 2]
#
# k = 0
# data_2 = np.zeros((1000, 22, 288))
# for j in range(len(Typ)):
#     if Typ[j] == 768:
#         k += 1
#         data_2[:, :, k-1] = s[(Pos[j]+500):(Pos[j]+1500), :22]
#
# # 去除NaN值
# data_2[np.isnan(data_2)] = 0
#
# # 预处理 - 带通滤波
# fc = 250  # 采样率
# Wl = 7
# Wh = 35  # 通带
# Wn = [Wl*2/fc, Wh*2/fc]
# b, a = cheby2(6, 60, Wn, btype='bandpass')
#
# for j in range(288):
#     data_2[:, :, j] = filtfilt(b, a, data_2[:, :, j], axis=0)
#
# print("Data shape:", data_2.shape)
# print("Labels shape:", label_2.shape)