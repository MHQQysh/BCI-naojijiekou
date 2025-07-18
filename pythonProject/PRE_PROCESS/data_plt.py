import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

filename = "D:/EEG_dataset/BCICIV_2a_gdf/A02T.gdf"

raw = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR',
                              exclude=(["EOG-left", "EOG-central", "EOG-right"]))

# print(raw.info)
raw.load_data()
raw.filter(4, 40, fir_design='firwin') # 7-35Hz filter

# raw.plot(duration=8, n_channels=22, clipping=None)
# plt.show()
# raw.plot_psd(average=True)
# plt.show()
events, event_id_all = mne.events_from_annotations(raw)

picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
# tmin=0
# tmax=4

# event_id = dict({'768':6,'770':8,'771':9,'772':10})
# event_id = [7, 8, 9, 10]
# event_id = [6]
# epochs = mne.Epochs(raw, events, event_id, proj=True, picks=picks, baseline=None, preload=True)
# print(raw)

event_id = {'6': 6}  # 设置事件id为6
epochs = mne.Epochs(raw, events, event_id, proj=True, picks=picks, baseline=None, preload=True)

# 获取所有事件标签为6的时间戳
event_indices = np.where(events[:, 2] == 6)[0]  # 找到所有事件标签为6的位置
event_times = events[event_indices, 0]  # 获取事件发生的时间戳

# 2. 假设每个6之间的下一个6是我们需要提取的数据区间的结束位置
sfreq = raw.info['sfreq']  # 获取采样率

# 创建一个空列表来存储每个epoch数据
epochs_list = []

# 遍历每对连续的标签为6的事件
for i in range(len(event_times) - 1):  # 遍历每两个连续的6
    start_time = event_times[i]  # 当前6的时间戳
    end_time = event_times[i + 1]  # 下一个6的时间戳
    print(start_time)
    print(end_time)
    # 将开始和结束时间转换为采样点
    # start_sample = int(start_time * sfreq)  # 起始采样点
    # end_sample = int(end_time * sfreq)  # 结束采样点

    # 使用 raw.get_data() 提取每对6之间的数据
    epoch_data = raw.get_data(start=start_time, stop=end_time)  # 提取数据
    epochs_list.append(epoch_data)
    ep_ary = np.array(epoch_data)
    print(ep_ary.shape)

max_len = max([epoch.shape[1] for epoch in epochs_list])  # 找出最大的采样点数

# 2. 对长度较短的 epoch 数据进行填充
padded_epochs_list = []
for epoch in epochs_list:
    # 如果 epoch 的长度小于最大长度，则补充零
    if epoch.shape[1] < max_len:
        padding = max_len - epoch.shape[1]
        # 在最后一维（时间维度）上进行零填充 每通道采样结果末尾填0
        padded_epoch = np.pad(epoch, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    else:
        padded_epoch = epoch  # 如果长度已经是最大长度，则无需填充
    padded_epochs_list.append(padded_epoch)

# 3. 将填充后的 epochs_list 转换为 numpy 数组
epochs_array = np.array(padded_epochs_list)

print(epochs_array)
epochs_data = epochs_array
# epochs_data = epochs.get_data()
# print(epochs.drop_log)

# print(epochs_data.shape)
#
#
data = epochs_data[:, :, :-1]

epoch_data = data[50, :, :]  # 获取第一个epoch的所有通道数据

# 创建图形
plt.figure(figsize=(10, 6))

# 为每个通道绘制数据，所有通道都画在同一个图上
for i, channel_data in enumerate(epoch_data):
    # 每个通道的信号使用不同的颜色绘制
    t_tag = np.arange(0, len(channel_data))
    print(len(t_tag))
    tms = t_tag * 4
    plt.plot(tms, channel_data, label=epochs.info['ch_names'][i])

# 设置标题和标签
plt.title('EEG Data for First Epoch (All Channels)', fontsize=14)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude (uV)', fontsize=12)

# 添加图例
plt.legend(loc='upper right')

# 显示图形
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i, channel_data in enumerate(epoch_data):
    plt.subplot(len(epoch_data), 1, i + 1)  # 每个子图对应一个通道

    t_tag = np.arange(0, len(channel_data))
    print(len(t_tag))
    tms = t_tag * 4
    plt.plot(tms, channel_data, label=epochs.info['ch_names'][i])
    # plt.title(f"Channel {epochs.info['ch_names'][i]}")

plt.xlabel('Time (ms)')
plt.tight_layout()
plt.show()