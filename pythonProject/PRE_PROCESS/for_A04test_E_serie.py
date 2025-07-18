import glob
from collections import Counter

import mne
import numpy as np
import matplotlib.pyplot as plt
import os

class EEG_PREPRO():
    def __init__(self, file_path, target_path, target_path2):
        self.target_path2 = target_path2
        self.file_path = file_path
        self.target_path = target_path
        # self.p_num = p_num

    def process(self):
        file_list = glob.glob(os.path.join(self.file_path, 'A04E.gdf'))

        for filename in file_list:
            raw = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR',
                                      exclude=(["EOG-left", "EOG-central", "EOG-right"]))
            # raw = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR')
            events, event_id_raw = mne.events_from_annotations(raw)

            event_ids = events[:, 2]
            event_counts = Counter(event_ids)
            print(event_counts)
            print(event_id_raw)
            print('-----------------------------------------------------------')

            # if 'A04T' in filename:
            #     print(f'Skipping file: {filename}')
            #     continue
            # print(events)
            print(filename)

            raw.load_data()
            # annotations = raw.annotations
            # np.save('../A03.npy', annotations)
            # print(annotations)
            # print(raw.info) # raw data information
            raw.filter(7, 35, fir_design='firwin')
            picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
            tmin = 0
            tmax = 4


            event_id = dict({'783': 7})
            # event_id = dict({'769': 7, '770': 8})

            epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
            # print(epochs) # event names
            epochs_data = epochs.get_data()
            # print(epochs_data.shape)
            print(type(epochs_data))

            data = epochs_data[:, :, :-1]
            # labels = epochs.events[:, -1]
            # labels = labels + 2
            print(data.shape)
            # print(labels.shape)
            base_name = os.path.basename(filename).replace('.gdf', '.npy')
            print(f'base name is {base_name}')
            save_name = os.path.join(self.target_path, base_name)
            np.save(save_name, data)

if __name__ == "__main__":
    file_name = "D:/EEG_dataset/BCICIV_2a_gdf/"
    target_name = "D:/EEG_dataset/BCICIV_2a_npy/"
    target_name2 = "D:/EEG_dataset/BCICIV_2a_npy/train_test/test/labels/"
    eeg_pro = EEG_PREPRO(file_path=file_name, target_path=target_name, target_path2=target_name2)
    eeg_pro.process()