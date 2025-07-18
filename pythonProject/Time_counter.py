import torchvision.transforms as transforms # 导入torchvision库中的transforms模块，用于图像数据预处理
from torch import optim # 导入torch库中的optim模块，用于优化器
from torch.utils.data import DataLoader,Dataset # 导入torch.utils.data模块中的DataLoader和Dataset，用于数据加载和数据集定义
import warnings # 导入warnings模块，用于控制警告信息
warnings.filterwarnings('ignore') # 忽略所有警告
from sklearn.metrics import classification_report # 导入sklearn.metrics模块中的classification_report，用于生成分类报告
from sklearn.metrics import accuracy_score # 导入sklearn.metrics模块中的accuracy_score，用于计算准确率
from sklearn import metrics # 导入sklearn库中的metrics模块，用于模型评估
import matplotlib.pyplot as plt # 导入matplotlib.pyplot模块，用于绘图
from torch import nn # 导入torch库中的nn模块，用于构建神经网络
import torch.nn.functional as F # 导入torch.nn.functional模块，通常用于激活函数、池化等无参数层
import torchvision.datasets # 导入torchvision.datasets模块，包含常用的数据集
from sklearn.preprocessing import MinMaxScaler # 导入sklearn.preprocessing模块中的MinMaxScaler，用于数据归一化
import mne,glob,os,re,torch,sklearn,warnings # 导入mne（脑电数据处理）、glob（文件路径查找）、os（操作系统交互）、re（正则表达式）、torch、sklearn、warnings等库
import numpy as np # 导入numpy库，用于数值计算
from tqdm import tqdm # 导入tqdm库，用于显示进度条
from sklearn.preprocessing import OneHotEncoder # 导入sklearn.preprocessing模块中的OneHotEncoder，用于独热编码
from sklearn import preprocessing # 导入sklearn库中的preprocessing模块，用于数据预处理
import torch # 再次导入torch库
import torchvision.transforms as transforms # 再次导入torchvision.transforms模块
transf = transforms.ToTensor() # 创建ToTensor转换对象，将PIL Image或ndarray转换为Tensor
from sklearn.preprocessing import StandardScaler # 导入sklearn.preprocessing模块中的StandardScaler，用于数据标准化
# from EEGCNN_Module import * # 注释掉的行，可能导入自定义的EEGCNN模型模块
from EEGNet import EEGNetModel # 导入自定义的EEGNet模型
import pandas as pd # 导入pandas库，用于数据分析
import os # 导入os模块
import matplotlib.pyplot as plt # 再次导入matplotlib.pyplot模块
from sklearn.metrics import classification_report,confusion_matrix # 导入classification_report和confusion_matrix
from sklearn.metrics import accuracy_score # 再次导入accuracy_score
import sklearn # 再次导入sklearn库
from sklearn import metrics # 再次导入metrics模块
import logging # 导入logging模块，用于日志记录
from scipy import signal # 导入scipy.signal模块，用于信号处理

# 帮助类
class Config: # 定义配置类
    batch_size = 32 # 批处理大小
    lr = 0.01 # 学习率
    epochs = 800 # 训练轮数

def buttferfiter(data): # 定义巴特沃斯滤波器函数
    Fs = 250 # 采样率
    b, a = signal.butter(4, [8, 30], 'bandpass',fs=Fs) # 设计4阶巴特沃斯带通滤波器，通带范围8-30Hz
    data = signal.filtfilt(b, a, data, axis=1) # 应用零相位滤波器到数据上，axis=1表示在第二个维度上进行滤波
    return data # 返回滤波后的数据

class EEGDataset(Dataset): # 定义EEG数据集类，继承自torch.utils.data.Dataset
    def __init__(self, data_dir): # 构造函数，传入数据目录
        self.data_files = glob.glob(os.path.join(data_dir, '*.npy')) # 查找数据目录下所有.npy文件
        self.data = [] # 初始化数据列表
        for file in self.data_files: # 遍历所有数据文件
            self.data.append(np.load(file)) # 加载.npy文件并添加到数据列表
        self.data = np.concatenate(self.data, axis=0)  # 沿第一个维度合并所有数据
        np.random.shuffle(self.data)  # 打乱数据

    def __len__(self): # 返回数据集的长度
        return len(self.data)

    def __getitem__(self, idx): # 根据索引获取数据项
        data = self.data[idx] # 获取对应索引的数据
        data = torch.tensor(data, dtype=torch.float32) # 将numpy数组转换为torch张量，并指定数据类型
        return data # 返回数据

class FeatureDataset(Dataset): # 定义特征数据集类，继承自torch.utils.data.Dataset
    def __init__(self, data_path, target_path, transform=None): # 构造函数，传入数据路径、目标路径和转换器
        self.data_path = data_path # 数据文件路径
        self.target_path = target_path # 目标文件路径

        self.data = self.parse_data_file(data_path) # 解析数据文件
        self.target = self.parse_target_file(target_path) # 解析目标文件
        self.transform = transform # 转换器

        # Set train/test split (注释掉的训练/测试集划分代码)
        # self.is_train = is_train
        # split_index = int(0.6 * len(self.data))

        # if self.is_train:
        #     self.data = self.data[:split_index]
        #     self.target = self.target[:split_index]
        # else:
        #     self.data = self.data[split_index:]
        #     self.target = self.target[split_index:]

    def parse_data_file(self, file_path): # 解析数据文件函数
        data = np.load(file_path)  # 从.npy格式文件加载数据
        data = np.array(data, dtype=np.float32) # 转换为float32类型的numpy数组

        # Normalize data (数据归一化)
        scaler = StandardScaler().fit(data.reshape(-1, data.shape[-1]))  # 初始化StandardScaler并拟合数据，reshape(-1, data.shape[-1])表示在最后一个维度上进行标准化
        data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape) # 对数据进行标准化并恢复原始形状

        data = torch.tensor(data) # 将numpy数组转换为torch张量
        data = data.unsqueeze(1)  # 添加一个通道维度（通常用于图像数据，这里可能是为了适应模型输入）
        # print("data shape is :") # 打印数据形状（注释掉）
        # print(data.shape) # 打印数据形状（注释掉）
        return data # 返回处理后的数据

    def parse_target_file(self, target_path): # 解析目标文件函数
        target = np.load(target_path) # 加载目标数据
        target = np.array(target, dtype=np.float32) # 转换为float32类型的numpy数组

        # One-Hot 编码
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore') # 初始化OneHotEncoder，不返回稀疏矩阵，忽略未知类别
        x_hot = target.reshape(-1, 1) # 将目标数据reshape为二维数组，以便进行独热编码
        encoder.fit(x_hot) # 拟合编码器
        x_oh = encoder.transform(x_hot) # 对目标数据进行独热编码

        # 自定义转换（假设存在 transf 函数）
        d = transf(x_oh) # 应用自定义的转换（这里是ToTensor）

        # 获取 One-Hot 标签
        x_hot_label = torch.argmax(torch.tensor(d), dim=2).long() # 将张量转换为长整型，并获取独热编码中的最大值索引作为标签

        # 调整标签形状
        # print("x_hot_label.shape is : ") # 打印标签形状（注释掉）
        # print(x_hot_label.shape) # 打印标签形状（注释掉）
        label = x_hot_label.transpose(1, 0) # 转置标签

        # print(label.shape) # 打印标签形状（注释掉）
        target = torch.squeeze(label) # 移除单维度条目
        # print(target.shape) # 打印标签形状（注释掉）

        return target # 返回处理后的目标标签

    def __len__(self): # 返回数据集的长度
        return len(self.data)

    def __getitem__(self, index): # 根据索引获取数据项
        x = self.data[index] # 获取输入数据
        # print(x.shape) # 打印输入数据形状（注释掉）
        target = self.target[index] # 获取目标标签
        # print("label is like") # 打印标签信息（注释掉）
        # print(target) # 打印标签（注释掉）
        return x, target # 返回输入数据和目标标签

def set_seed(seed=0): # 定义设置随机种子函数
    import random # 导入random模块
    random.seed(seed) # 设置random模块的随机种子
    np.random.seed(seed) # 设置numpy的随机种子
    torch.manual_seed(seed) # 设置CPU的随机种子
    torch.cuda.manual_seed_all(seed) # 设置所有GPU的随机种子

# 定义L1正则化函数
def l1_regularizer(weight, lambda_l1): # 定义L1正则化函数，传入权重和L1系数
    return lambda_l1 * torch.norm(weight, 1) # 返回L1范数乘以L1系数

# 定义L2正则化函数
def l2_regularizer(weight, lambda_l2): # 定义L2正则化函数，传入权重和L2系数
    return lambda_l2 * torch.norm(weight, 2) # 返回L2范数乘以L2系数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设置设备为GPU（如果可用）或CPU
config = Config() # 实例化配置类

train_transforms = transforms.Compose([transforms.ToTensor()]) # 训练数据的转换，只包含ToTensor
train_dataset = FeatureDataset(data_path = r'D:\EEG_dataset\BCICIV_2a_npy\merged\merged_train_data.npy', # 实例化训练数据集
                               target_path=r'D:\EEG_dataset\BCICIV_2a_npy\merged\merged_train_label.npy')
train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=config.batch_size) # 实例化训练数据加载器，shuffle为True表示打乱数据

val_transforms = transforms.Compose([transforms.ToTensor()]) # 验证数据的转换，只包含ToTensor
val_dataset = FeatureDataset(data_path = r'D:\EEG_dataset\BCICIV_2a_npy\merged\merged_test_data.npy', # 实例化验证数据集
                               target_path=r'D:\EEG_dataset\BCICIV_2a_npy\merged\merged_test_label.npy')
val_dataloader = DataLoader(val_dataset,batch_size=config.batch_size)  # 实例化验证数据加载器
set_seed() # 设置随机种子

def show_plot(accuracy_history,loss_history,test_accuracy): # 定义显示绘图函数，传入准确率历史、损失历史和测试准确率
    plt.figure(figsize=(20,10)) # 创建一个20x10英寸的图形
    #fig2
    plt.subplot(121) # 创建第一个子图，1行2列的第1个
    plt.plot(loss_history,marker=".",color="c") # 绘制训练损失曲线，点标记，青色
    plt.title('train loss') # 设置子图标题
    #fig3
    plt.subplot(122) # 创建第二个子图，1行2列的第2个
    plt.plot(accuracy_history,marker="o",label="train_acc") # 绘制训练准确率曲线，圆圈标记，标签为"train_acc"
    plt.plot(test_accuracy, marker='o', label="test_acc") # 绘制测试准确率曲线，圆圈标记，标签为"test_acc"
    plt.title("ACC") # 设置子图标题
    plt.legend(loc="best") # 显示图例，位置为最佳
    plt.savefig('acc_loss.png') # 保存图形为acc_loss.png
    plt.show() # 显示图形
    
def plot_recall(epoch_list,recall1,recall2,recall3,recall4): # 定义绘制召回率函数，传入epoch列表和四个类别的召回率
    plt.figure(figsize=(15,8)) # 创建一个15x8英寸的图形
    plt.plot(epoch_list,recall1, color='purple', label='Back1_Recall',marker=".") # 绘制第一个类别的召回率曲线
    plt.plot(epoch_list,recall2,color='c',label="Back2_Recall",marker=".") # 绘制第二个类别的召回率曲线
    plt.plot(epoch_list,recall3,color='g',label="Back3_Recall",marker=".") # 绘制第三个类别的召回率曲线
    plt.plot(epoch_list,recall4,color='m',label="Back4_Recall",marker=".") # 绘制第四个类别的召回率曲线
    plt.title('Recall during test') # 设置标题
    plt.xlabel('Epoch') # 设置X轴标签
    plt.ylabel('Recall_Vales') # 设置Y轴标签
    plt.legend() # 显示图例
    plt.savefig("recall.jpg") # 保存图形为recall.jpg
    plt.show() # 显示图形

def plot_precision(epoch_list,precision1,precision2,precision3,precision4): # 定义绘制精确率函数，传入epoch列表和四个类别的精确率
    plt.figure(figsize=(15,8)) # 创建一个15x8英寸的图形
    plt.plot(epoch_list,precision1, color='black', label='Back1_Precision',marker="o") # 绘制第一个类别的精确率曲线
    plt.plot(epoch_list,precision2, color='b', label='Back2_Precision',marker="o") # 绘制第二个类别的精确率曲线
    plt.plot(epoch_list,precision3, color='m', label='Back3_Precision',marker="o") # 绘制第三个类别的精确率曲线
    plt.plot(epoch_list,precision4, color='c', label='Back4_Precision',marker="o") # 绘制第四个类别的精确率曲线
    plt.xlabel('Epoch') # 设置X轴标签
    plt.ylabel('Precision_Vales') # 设置Y轴标签
    plt.title('Precision during test') # 设置标题
    plt.legend() # 显示图例
    plt.savefig("precision.jpg") # 保存图形为precision.jpg
    
    plt.show() # 显示图形

def plot_f1(epoch_list,f1_1,f1_2,f1_3,f1_4): # 定义绘制F1分数函数，传入epoch列表和四个类别的F1分数
    plt.figure(figsize=(15,8)) # 创建一个15x8英寸的图形
    plt.plot(epoch_list,f1_1, color='yellow', label='Back1_F1',marker="^") # 绘制第一个类别的F1分数曲线
    plt.plot(epoch_list,f1_2, color='g', label='Back2_F1',marker="^") # 绘制第二个类别的F1分数曲线
    plt.plot(epoch_list,f1_3, color='b', label='Back3_F1',marker="^") # 绘制第三个类别的F1分数曲线
    plt.plot(epoch_list,f1_4, color='m', label='Back4_F1',marker="^") # 绘制第四个类别的F1分数曲线
    plt.xlabel('Epoch') # 设置X轴标签
    plt.ylabel('F1_Values') # 设置Y轴标签
    plt.title('f1 during test') # 设置标题
    plt.legend() # 显示图例
    plt.savefig("f1.jpg") # 保存图形为f1.jpg
    plt.show() # 显示图形
    
def get_logger(filename, verbosity=1, name=None): # 定义获取日志记录器函数
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING} # 日志级别字典
    formatter = logging.Formatter( # 定义日志格式
        "[%(asctime)s]%(message)s"
    )
    logger = logging.getLogger(name) # 获取日志记录器实例
    logger.setLevel(level_dict[verbosity]) # 设置日志级别

    fh = logging.FileHandler(filename, "w") # 创建文件处理器，写入模式
    fh.setFormatter(formatter) # 设置文件处理器的格式
    logger.addHandler(fh) # 将文件处理器添加到日志记录器

    sh = logging.StreamHandler() # 创建流处理器（控制台输出）
    sh.setFormatter(formatter) # 设置流处理器的格式
    logger.addHandler(sh) # 将流处理器添加到日志记录器

    return logger # 返回日志记录器
    
def DrawConfusionMatrix(save_model_name,val_dataloader): # 定义绘制混淆矩阵函数，传入模型保存路径和验证数据加载器
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设置设备
    model = EEGNetModel().to(device) # 实例化EEGNet模型并将其移动到指定设备
    model.load_state_dict(torch.load(os.path.join(save_model_name,"EEGNET_MODEL.pth"))) # 加载模型权重
    model.eval() # 设置模型为评估模式
    predict = [] # 存储预测结果的列表
    gt = [] # 存储真实标签的列表
    with torch.no_grad(): # 在此上下文中不计算梯度
        for data_label in val_dataloader: # 遍历验证数据加载器
            x,target = data_label # 获取输入数据和目标标签
            x,target = x.to(device),target.to(device) # 将数据和标签移动到指定设备
            outputs = model(x) # 模型前向传播
            _, predicted = torch.max(outputs, 1) # 获取预测结果（最大概率的类别）

            tmp_predict = predicted.cpu().detach().numpy() # 将预测结果从GPU移动到CPU，转换为numpy数组
            tmp_label = target.cpu().detach().numpy() # 将真实标签从GPU移动到CPU，转换为numpy数组

            if len(predict) == 0: # 如果predict列表为空
                predict = np.copy(tmp_predict) # 复制预测结果
                gt = np.copy(tmp_label) # 复制真实标签
            else:
                predict = np.hstack((predict,tmp_predict)) # 水平拼接预测结果
                gt = np.hstack((gt,tmp_label)) # 水平拼接真实标签

    cm = confusion_matrix(y_true=gt, y_pred=predict)   # 计算混淆矩阵
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = ['left','right','feet',"tongue"]) # 创建混淆矩阵显示对象
    disp.plot() # 绘制混淆矩阵
    save_hunxiao_path = os.path.join(save_model_name,'混淆矩阵.png') # 构造保存混淆矩阵的路径
    plt.savefig(save_hunxiao_path,dpi = 1000) # 保存混淆矩阵图像，分辨率为1000dpi

    
def timec(model, x): # 定义模型推理时间测量函数
    start_event = torch.cuda.Event(enable_timing=True) # 创建CUDA事件，用于记录开始时间
    end_event = torch.cuda.Event(enable_timing=True) # 创建CUDA事件，用于记录结束时间
    time_list = [] # 存储时间的列表
    for _ in range(50): # 循环50次进行测量
        start_event.record() # 记录开始事件
        ret = model(x) # 模型前向传播
        end_event.record() # 记录结束事件
        end_event.synchronize() # 等待CUDA事件完成
        time_list.append(start_event.elapsed_time(end_event) / 1000) # 计算时间差（毫秒），并转换为秒，添加到列表中

    print(f"event avg time :{sum(time_list[5:]) / len(time_list[5:]):.5f}") # 打印平均时间（忽略前5次，因为首次运行可能存在预热开销）

def main(): # 定义主函数
    device = torch.device("cuda:0") # 设置设备为第一个CUDA GPU
    model = EEGNetModel().to(device) # 实例化EEGNet模型并将其移动到GPU
    x = torch.randn(size=(1, 1, 22, 1000), device=device) # 创建一个随机输入张量，形状为(1, 1, 22, 1000)，并移动到GPU
    timec(model, x) # 调用时间测量函数

if __name__ == '__main__': # 如果脚本作为主程序运行
    main() # 调用主函数