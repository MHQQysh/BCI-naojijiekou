import numpy as np
import os

source_path = "D:/EEG_dataset/BCICIV_2a_npy/train_test/train/"
# 定义文件名列表
file_names = [f"{source_path}A0{i}T.npy" for i in range(1, 10)]


# 加载并合并所有文件
arrays = [np.load(file) for file in file_names]
merged_array = np.concatenate(arrays, axis=0)

target_path = "D:/EEG_dataset/BCICIV_2a_npy/merged/merged_train_data.npy"
# 保存合并后的数组
print(merged_array.shape)

np.save(target_path, merged_array)

print("合并完成并保存为 merged_label.npy")
