import os

import mne
import scipy.io
import numpy as np
from scipy.signal import cheby2, filtfilt

class EEG_PREPRO():
    def __init__(self, tmin, tmax, frq_low, frq_high, serie_flag):
        self.tmin = tmin
        self.tmax = tmax
        self.frq_low = frq_low
        self.frq_high = frq_high
        self.serie_flag = serie_flag

    def rd_raw(self, subject_index, session_type):
        labeldir_2 = f'D:/EEG_dataset/BCICIV_2a_gdf/true_labels/A0{subject_index}{session_type}.mat'
        folder_name = f"D:/EEG_dataset/asynchronous/BCICIV_2a_npy/{self.tmin}-{self.tmax}s_{self.frq_low}-{self.frq_high}hz"
        dir_2 = f'/A0{subject_index}{session_type}.npy'
        dir_0 = folder_name + dir_2
        print(dir_0)
        data = np.load(dir_0)
        labels = scipy.io.loadmat(labeldir_2)

        return data, labels

    def mat_merge(self):
        for subject_index in range(1, 10):
            data, labels = self.rd_raw(subject_index, self.serie_flag)
            print("labels.shape is ")
            print(len(labels))
            combined_data = {
                'data': data,
                'labels': labels['classlabel']  # 假设标签在 .mat 文件中存储为 'true_labels'
            }
            folder_path = f'D:/EEG_dataset/asynchronous/BCICIV_2a_mat/{self.tmin}-{self.tmax}s_{self.frq_low}-{self.frq_high}hz/'
            os.makedirs(folder_path, exist_ok=True)
            save_name = folder_path + f'A0{subject_index}{self.serie_flag}.mat'
            scipy.io.savemat(save_name, combined_data)


def main():
    session_type = 'E'
    EEG_pro = EEG_PREPRO(0, 3, 4,40, session_type)
    EEG_pro.mat_merge()

if __name__ == "__main__":
    main()

