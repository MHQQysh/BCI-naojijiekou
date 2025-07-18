import numpy as np
import pandas as pd

# 加载npy文件
npy_file = np.load('D:\EEG_dataset\BCICIV_2a_npy\merged\merged_label.npy')

# 将npy文件的数据转换为DataFrame
df = pd.DataFrame(npy_file)

# 保存为csv文件
df.to_csv('D:\EEG_dataset\BCICIV_2a_npy\merged\merged_label.csv', index=False)