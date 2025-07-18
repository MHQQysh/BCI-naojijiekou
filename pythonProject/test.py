import numpy as np
import mne

# data = np.load('D:/EEG_dataset/BCICIV_2a_npy/A04T.npy', allow_pickle=True)
# data = np.load('A04.npy', allow_pickle=True)
# print(data.shape)

raw = mne.io.read_raw_gdf('D:/EEG_dataset/BCICIV_2a_gdf/A03T.gdf', preload=True)

# 将注释转换为事件
events, event_id = mne.events_from_annotations(raw)

# 设置时间范围
tmin = 2  # 事件前200ms
tmax = 6   # 事件后500ms

raw.load_data()

picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)

print(event_id)

event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})
# 创建Epochs对象
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,  proj=True, picks=picks, baseline=None, preload=True)

# 统计每个事件ID对应的片段数量
event_counts = {key: len(epochs[key]) for key in event_id}
print(event_counts)