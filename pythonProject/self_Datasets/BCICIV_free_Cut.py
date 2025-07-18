import glob

import mne
import numpy as np
import matplotlib.pyplot as plt
import os

class EEG_PREPRO():
    def __init__(self, file_path, tmin, tmax, frq_low, frq_high, serie_flag):
        self.file_path = file_path
        self.tmin = tmin
        self.tmax = tmax
        self.frq_low = frq_low
        self.frq_high = frq_high
        self.serie_flag = serie_flag
        # self.p_num = p_num

    def self_get_data(self, raw, events, event_id):
        raw.load_data()
        # print(raw.info) # raw data information
        frq_low = self.frq_low
        frq_high = self.frq_high
        raw.filter(frq_low, frq_high, fir_design='firwin')
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
        tmin = self.tmin
        tmax = self.tmax

        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
        print(epochs)  # event names
        epochs_data = epochs.get_data()
        return  epochs_data


    def create_folder_and_save_file(self, time_lower, time_upper, filter_lower, filter_upper, base_dir='D:/EEG_dataset/asynchronous/BCICIV_2a_npy/'):
        folder_name = f"{time_lower}-{time_upper}s_{filter_lower}-{filter_upper}hz"
        folder_path = os.path.join(base_dir, folder_name)

        # Create the directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        return folder_path


    def process(self):
        series_name = 'A*' + self.serie_flag + '.gdf'
        file_list = glob.glob(os.path.join(self.file_path, series_name))

        for filename in file_list:
            # raw = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR')
            raw = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR',
                                      exclude=(["EOG-left", "EOG-central", "EOG-right"]))

            events, _ = mne.events_from_annotations(raw)

            if 'E' in filename:
                event_id = dict({'783': 7, })
            elif 'A04' in filename:
                event_id = dict({'769': 5, '770': 6, '771': 7, '772': 8})
            else:
                event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})

            print(filename)

            epochs_data = self.self_get_data(raw, events, event_id)
            print(epochs_data.shape)
            # print(type(epochs_data))
            data = epochs_data[:, :, :-1]
            # labels = epochs.events[:, -1]
            print(data.shape)
            # print(labels.shape)
            base_name = os.path.basename(filename).replace('.gdf', '.npy')
            # base_name2 = os.path.basename(filename).replace('.gdf', '.npy')
            print(f'base name is {base_name}')
            target_path = self.create_folder_and_save_file(self.tmin, self.tmax, self.frq_low, self.frq_high)
            save_name = os.path.join(target_path, base_name)
            # save_name2 = os.path.join(self.target_path2, base_name2)
            np.save(save_name, data)
            # np.save(save_name2, labels)

if __name__ == "__main__":
    file_name = "D:/EEG_dataset/BCICIV_2a_gdf/"
    # target_name = "D:/EEG_dataset/BCICIV_2a_npy/"
    # target_name = "D:/EEG_dataset/asynchronous/0-4s_4-40hz/BCICIV_2a_npy/"
    # target_name2 = "D:/EEG_dataset/BCICIV_2a_npy/train_test/test/labels/"
    eeg_pro = EEG_PREPRO(file_path=file_name, tmin=0, tmax=3, frq_low=4, frq_high=40, serie_flag='T')
    eeg_pro.process()