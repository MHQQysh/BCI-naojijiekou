# import numpy as np

# # 伪造EEG信号数据
# num_trials = 100
# num_channels = 22 # 模拟22个通道
# num_samples = 1000

# # 生成随机数据，模拟EEG信号的浮点数
# # 数据范围可以在 -100 到 100 微伏之间，模拟真实EEG的幅度
# fake_eeg_data = np.random.uniform(low=-100.0, high=100.0,
#                                   size=(num_trials, num_channels, num_samples)).astype(np.float32)

# # 保存为 .npy 文件
# # 注意：你的原始代码对数据维度有转置操作，这里我们直接生成转置后的形状，
# # 以便与你地形图代码中的 `data = np.transpose(data, (2, 1, 0))` 之前兼容。
# # 即，这里生成的 `fake_eeg_data` 对应的是地形图代码中 `data` 经过 `np.transpose(data, (2, 1, 0))` 后的形状
# # 也就是 (trials, channels, samples)。
# np.save('merged_train_data.npy', fake_eeg_data)

# print(f"已创建 merged_train_data.npy，形状为: {fake_eeg_data.shape}")


import numpy as np

num_labels = 100

fake_eeg_labels = np.random.randint(low=0, high=4, size=(num_labels, 1)).astype(np.float32)
fake_eeg_labels[:20] = 1.0 # 确保有标签为1的数据

np.save('merged_train_label.npy', fake_eeg_labels)

print(f"已创建 merged_train_label.npy，形状为: {fake_eeg_labels.shape}")