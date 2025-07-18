import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

filename = "D:/EEG_dataset/BCICIV_2a_gdf/A01T.gdf"

raw = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR',
                              exclude=(["EOG-left", "EOG-central", "EOG-right"]))

data, times= raw[:,:]
# t_idx = raw.time_as_index([1., 6.])
# data, times = raw[:, t_idx[0]:t_idx[1]]


print("data.shape is:", str(data.shape))
print("times.shape is", str(times.shape))
# print(raw.info)
# raw.load_data()
# raw.filter(7, 35, fir_design='firwin') # 7-35Hz filter

# raw.plot(duration=8, n_channels=22, clipping=None)
#
# plt.show()
# raw.plot_psd(average=True)
# plt.show()
events, events_id_all = mne.events_from_annotations(raw)
print("events.shape")
print(events[0:200, :])
print("events_id_all")
print(events_id_all)


def checkpt(events):
    lst_begin = []
    lst_end = []
    lst_event = []
    lst_after = []
    ct7 = 0
    ct8 = 0
    ct9 = 0
    ct10 = 0
    data_len = events.shape[0]
    for i in range(data_len - 1):

        if events[i, 2] == 7:
            lst_begin.append(events[i, 0])
            lst_end.append(events[i+1, 0])

            lst_event.append(events[i, 2])
            lst_after.append(events[i+1, 2])

            ct7 += 1
        elif events[i, 2] == 8:
            lst_begin.append(events[i, 0])
            lst_end.append(events[i + 1, 0])

            lst_event.append(events[i, 2])
            lst_after.append(events[i + 1, 2])

            ct8 += 1
        elif events[i, 2] == 9:
            lst_begin.append(events[i, 0])
            lst_end.append(events[i + 1, 0])

            lst_event.append(events[i, 2])
            lst_after.append(events[i + 1, 2])

            ct9 += 1
        elif events[i, 2] == 10:
            lst_begin.append(events[i, 0])
            lst_end.append(events[i + 1, 0])

            lst_event.append(events[i, 2])
            lst_after.append(events[i + 1, 2])

            ct10 += 1
        else:
            continue

    if events[data_len-1, 2] == 7:
        lst_begin.append(events[data_len-1, 0])
        lst_end.append(-1)
        lst_event.append(events[data_len-1, 2])
        lst_after.append(-1)
        ct7 += 1
    elif events[data_len-1, 2] == 8:
        lst_begin.append(events[data_len-1, 0])
        lst_end.append(-1)
        lst_event.append(events[data_len - 1, 2])
        lst_after.append(-1)
        ct8 += 1
    elif events[data_len-1, 2] == 9:
        lst_begin.append(events[data_len-1, 0])
        lst_end.append(-1)
        lst_event.append(events[data_len - 1, 2])
        lst_after.append(-1)
        ct9 += 1
    elif events[data_len-1, 2] == 10:
        lst_begin.append(events[data_len-1, 0])
        lst_end.append(-1)
        lst_event.append(events[data_len - 1, 2])
        lst_after.append(-1)
        ct10 += 1

    print(ct7, ct8, ct9, ct10)
    return lst_begin, lst_end, lst_event, lst_after


def check_start(events):
    lst_begin = []
    ct = 0
    data_len = events.shape[0]
    for i in range(data_len):
        if events[i, 2] == 6:
            lst_begin.append(events[i, 0])
            print(events[i, 0])
            ct = ct + 1
    print(ct)

    return lst_begin

def cut_data(lst_b,  data, t1, t2):
    pcked = np.zeros((288, 22, 1000))
    print("data shape 1 is :")
    # print(data.shape)
    # print(data.shape[0])
    for i, idx in enumerate(lst_b):
        # print(i)
        # 提取每个关键点索引开始后的1000个数据点
        # print(type(data))
        # print(type(pcked))
        # print(i) 问题出在最后一项
        print("idx is: ",str(idx))
        print(idx + t1 * 250)
        print(idx + t2 * 250)
        pcked[i, :, :] = data[:, idx + t1 * 250:idx + t2 * 250]

        # print(to_saved)
    # pcked[i, :, :]
    # pcked = data
    return pcked

    # for i in range(len(lst_b)):
    #     flg1 = lst_b[i]
    #     start
    #     pcked = data[22, lst_b[i]+500:lst_b[i]+1250]


def cnt_len(lstb, lste):
    mi_len = 0
    for i in range(len(lstb)-1):
        mi_len = mi_len + lste[i] - lstb[i]
    avg_len = mi_len / (len(lstb) - 1)

    return avg_len

# 测试 测量数据长度
# lst_b开始索引 _e结束索引，lst_event对应事件，_after后续事件
# lst_b, lst_e, lst_event, lst_after = checkpt(events)
# print(lst_event[40:50], lst_after[40:50])
# avg_len = cnt_len(lst_b, lst_e)
# print("avg len of data is: ", str(avg_len))
# print(lst_e[-2] - lst_b[-2])

# 为了查看每一个事件标签内数据有效长度
def check_event_length(events):
    lst_begin = []
    lst_end = []
    lst_event = []
    lst_after = []
    lst_init = []
    ct7 = 0
    ct8 = 0
    ct9 = 0
    ct10 = 0
    data_len = events.shape[0]
    for i in range(data_len - 1):

        if events[i, 2] == 7:
            lst_begin.append(events[i, 0])
            lst_end.append(events[i+1, 0])

            lst_event.append(events[i, 2])
            lst_after.append(events[i+1, 2])

            ct7 += 1
        elif events[i, 2] == 8:
            lst_begin.append(events[i, 0])
            lst_end.append(events[i + 1, 0])

            lst_event.append(events[i, 2])
            lst_after.append(events[i + 1, 2])

            ct8 += 1
        elif events[i, 2] == 9:
            lst_begin.append(events[i, 0])
            lst_end.append(events[i + 1, 0])

            lst_event.append(events[i, 2])
            lst_after.append(events[i + 1, 2])

            ct9 += 1
        elif events[i, 2] == 10:
            lst_begin.append(events[i, 0])
            lst_end.append(events[i + 1, 0])

            lst_event.append(events[i, 2])
            lst_after.append(events[i + 1, 2])

            ct10 += 1
        elif events[i, 2] == 6:
            lst_init.append(events[i, 0])

        else:
            continue

    if events[data_len-1, 2] == 7:
        lst_begin.append(events[data_len-1, 0])
        lst_end.append(-1)
        lst_event.append(events[data_len-1, 2])
        lst_after.append(-1)
        ct7 += 1
    elif events[data_len-1, 2] == 8:
        lst_begin.append(events[data_len-1, 0])
        lst_end.append(-1)
        lst_event.append(events[data_len - 1, 2])
        lst_after.append(-1)
        ct8 += 1
    elif events[data_len-1, 2] == 9:
        lst_begin.append(events[data_len-1, 0])
        lst_end.append(-1)
        lst_event.append(events[data_len - 1, 2])
        lst_after.append(-1)
        ct9 += 1
    elif events[data_len-1, 2] == 10:
        lst_begin.append(events[data_len-1, 0])
        lst_end.append(-1)
        lst_event.append(events[data_len - 1, 2])
        lst_after.append(-1)
        ct10 += 1

    elif events[data_len-1, 2] == 6:
        lst_init.append(events[i, 0])

    # print(ct7, ct8, ct9, ct10)
    len_cont = []
    for j in range(len(lst_begin)-1):
        len_cont.append(lst_end[j]-lst_init[j])

    print("最后一个开始tag：",str(lst_init[-1]))
    print("最后一个事件tag: ",str(lst_begin[-1]))
    print("单个通道数据总长度",str(data.shape[1]))

    df = pd.DataFrame(len_cont, columns=['Length'])
    df.to_excel('event_length.xlsx', index=False)

def main1():
    # 原方案从分类标记点出发,现方案,从开始实验标记点出发
    # lst_start = check_start(events)
    lst_start, lst_end, lst_event, lst_after = checkpt(events)
    pcked = cut_data(lst_start, data, 0, 4)
    print("pcked shape is:", str(pcked.shape))
    np.save('A01T_test.npy', pcked)

def main2():
    data1 = np.load('D:/EEG_dataset/BCICIV_2a_npy/A01T.npy')
    data2 = np.load('A01T_test.npy')
    test1 = data1[0,0,:]
    test2 = data2[0,0,:]
    # print(test1)
    print(len(test2))
    print(test2[:50])
    print(test1[:50])
    # contra = test2-test1
    # print(contra)
    plt.figure(figsize=(10, 5))
    plt.plot(test1, label='Data1 (1, 1, :)')
    plt.plot(test2, label='Data2 (1, 1, :)')
    plt.xlabel('Sample Points')
    plt.ylabel('Amplitude')
    plt.title('Plot of (1, 1, :) from Data1 and Data2')
    plt.legend()
    plt.show()

def main3():
    check_event_length(events)



# if __name__ == '__main__':
    # main1()
    # main2()
    # main3()


# print(events_id)

# picks = mne.pick_types(raw.info)
# picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
# tmin=5
# tmax=7

# event_id = dict({'769':7,'770':8,'771':9,'772':10})
# event_id = [7, 8, 9, 10]
# epochs = mne.Epochs(raw, events, event_id, tmin=5, tmax=7, proj=True, picks=picks, baseline=None, preload=True)
# print(raw)

# 将npy文件的数据转换为DataFrame
# df = pd.DataFrame(events)

# 保存为csv文件
# df.to_csv('test2.csv', index=False)

# print(epochs)
# print(epochs.drop_log)

# epochs_data = epochs.get_data()
# print(epochs.drop_log)

# print(epochs_data.shape)
# print(epochs_data)
# type(epochs_data)
#
#
# data = epochs_data[:, :, :]
# print(data.shape)
# BCI_IV_2a_data = np.save('A01T.npy', data)