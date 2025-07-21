"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""
# remember to change paths

import argparse
import os
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import matplotlib.pyplot as plt
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# Import Weights & Biases
import wandb
# Import tqdm
from tqdm import tqdm

# ------------------------------------------------------------------------------------------
# Argparse Setup
parser = argparse.ArgumentParser(description='EEG Conformer Training Script')
parser.add_argument('--gpu', type=int, default=0,
                    help='CUDA device ID to use (e.g., 0, 1, 2, 3). Default is 0.')
parser.add_argument('--wandb_project', type=str, default='EEG-Conformer-Decoding',
                    help='Weights & Biases project name.')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='Weights & Biases entity name (your username or team name).')
parser.add_argument('--log_interval', type=int, default=20, 
                    help='Log metrics to WandB every N epochs. Default is 20.')
parser.add_argument('--output_base_dir', type=str, default='./final_results', # 新增参数：输出文件根目录
                    help='Base directory for all experiment outputs (logs, models, results).')
parser.add_argument('--nepoch', type=int, default=2000, # 新增参数：输出文件根目录
                    help='Base directory for all experiment outputs (logs, models, results).')
args = parser.parse_args()

# Set GPU environment variables
gpus = [args.gpu]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

# Check CUDA availability
if not torch.cuda.is_available():
    tqdm.write(f"WARNING: CUDA device {args.gpu} is not available. Training will run on CPU.")

if len(gpus) == 0:
    tqdm.write("WARNING: No GPUs specified or available. DataParallel might not work as expected.")


# Convolution module
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self, nsub, cha_tag, experiment_output_dir,nepoch): # Pass experiment_output_dir here
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = nepoch
        self.c_dim = 4 # Number of classes
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        # This is the path to your preprocessed .mat data files. KEEP THIS AS THE DATA SOURCE.
        self.root = '/data/wfy/BCI/EEG-Conformer-main/preprocessing/output/' 

        # Define the directory for logs and models specific to THIS experiment run
        self.experiment_output_dir = experiment_output_dir
        
        # Subject-specific log file path
        self.subject_log_dir = os.path.join(self.experiment_output_dir, "subject_logs")
        os.makedirs(self.subject_log_dir, exist_ok=True) # Create if it doesn't exist
        self.log_write = open(os.path.join(self.subject_log_dir, f"log_subject{self.nSub}.txt"), "w")


        if torch.cuda.is_available():
            self.Tensor = torch.cuda.FloatTensor
            self.LongTensor = torch.cuda.LongTensor
        else:
            self.Tensor = torch.FloatTensor
            self.LongTensor = torch.LongTensor
            tqdm.write(f"Subject {self.nSub}: Running on CPU, performance will be significantly lower.")

        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_l2 = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.criterion_l1.cuda()
            self.criterion_l2.cuda()
            self.criterion_cls.cuda()

        self.model = Conformer().cuda() if torch.cuda.is_available() else Conformer()
        if torch.cuda.device_count() > 1 and len(gpus) > 1:
            tqdm.write(f"Subject {self.nSub}: Using DataParallel on GPUs: {gpus}")
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)[0]
            if len(cls_idx) == 0:
                continue
            
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            num_aug_samples_per_class = int(self.batch_size / 4)
            
            if tmp_data.shape[0] == 0:
                continue
            
            tmp_aug_data = np.zeros((num_aug_samples_per_class, 1, 22, 1000))
            for ri in range(num_aug_samples_per_class):
                rand_indices = np.random.choice(tmp_data.shape[0], size=8, replace=True)
                for rj in range(8):
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = \
                        tmp_data[rand_indices[rj], :, :, rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(np.full(num_aug_samples_per_class, cls4aug + 1))


        if not aug_data:
            return torch.empty(0, 1, 22, 1000).type(self.Tensor), torch.empty(0).type(self.LongTensor)

        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).type(self.Tensor)
        aug_label = torch.from_numpy(aug_label - 1).type(self.LongTensor)
        return aug_data, aug_label

    def get_source_data(self):
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label'] 

        self.train_data = np.transpose(self.train_data, (2, 1, 0)) 
        self.train_data = np.expand_dims(self.train_data, axis=1) 
        tqdm.write(f"Subject {self.nSub} Train data shape: {self.train_data.shape}, Train label shape: {self.train_label.shape}")
        
        self.train_label = np.transpose(self.train_label) 
        self.allLabel = self.train_label[0] 
        tqdm.write(f"Subject {self.nSub} all labels shape is {self.allLabel.shape}")

        self.allData = self.train_data # Corrected: moved assignment here

        if self.allData.shape[0] != self.allLabel.shape[0]:
            raise ValueError(f"Mismatch in number of samples for training: allData has {self.allData.shape[0]} but allLabel has {self.allLabel.shape[0]}")

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]
        tqdm.write(f"Subject {self.nSub} Test data shape: {self.testData.shape}, Test label shape: {self.testLabel.shape}")

        if self.testData.shape[0] != self.testLabel.shape[0]:
             raise ValueError(f"Mismatch in number of samples for test data: testData has {self.testData.shape[0]} but testLabel has {self.testLabel.shape[0]}")

        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        
        if target_std == 0:
            tqdm.write(f"Subject {self.nSub} WARNING: Standard deviation is zero. Data will not be normalized.")
        else:
            self.allData = (self.allData - target_mean) / target_std
            self.testData = (self.testData - target_mean) / target_std

        return self.allData, self.allLabel, self.testData, self.testLabel


    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1) 

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False) 

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data_gpu = Variable(test_data.type(self.Tensor))
        test_label_gpu = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0 
        Y_true_collector = [] 
        Y_pred_collector = [] 

        log_interval = args.log_interval 

        epoch_iterator = tqdm(range(self.n_epochs), desc=f"Subject {self.nSub} Training (Epochs)", leave=False)


        for e in epoch_iterator: # Iterate over the tqdm object
            self.model.train() 
            
            train_loss_sum = 0
            train_correct_predictions = 0
            train_total_samples = 0

            for i, (img_batch, label_batch) in enumerate(self.dataloader):
                img_batch = Variable(img_batch.type(self.Tensor))
                label_batch = Variable(label_batch.type(self.LongTensor))

                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                
                if aug_data.size(0) > 0:
                    img_batch = torch.cat((img_batch, aug_data))
                    label_batch = torch.cat((label_batch, aug_label))
                
                tok, outputs = self.model(img_batch)
                loss = self.criterion_cls(outputs, label_batch) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item() * img_batch.size(0)
                train_pred_batch = torch.max(outputs, 1)[1]
                train_correct_predictions += (train_pred_batch == label_batch).sum().item()
                train_total_samples += label_batch.size(0)

            avg_train_loss = train_loss_sum / train_total_samples
            avg_train_acc = train_correct_predictions / train_total_samples

            self.model.eval() 
            
            with torch.no_grad():
                test_loss_sum = 0
                correct_predictions = 0
                total_samples = 0
                
                current_epoch_test_preds = []
                current_epoch_test_labels = []

                for batch_test_data, batch_test_label in self.test_dataloader:
                    batch_test_data_gpu = Variable(batch_test_data.type(self.Tensor))
                    batch_test_label_gpu = Variable(batch_test_label.type(self.LongTensor))

                    Tok_batch, Cls_batch = self.model(batch_test_data_gpu)
                    batch_loss_test = self.criterion_cls(Cls_batch, batch_test_label_gpu)
                    
                    test_loss_sum += batch_loss_test.item() * batch_test_data_gpu.size(0)
                    
                    y_pred_batch = torch.max(Cls_batch, 1)[1]
                    correct_predictions += (y_pred_batch == batch_test_label_gpu).sum().item()
                    total_samples += batch_test_label_gpu.size(0)

                    current_epoch_test_preds.append(y_pred_batch)
                    current_epoch_test_labels.append(batch_test_label_gpu)
                
                test_acc = correct_predictions / total_samples
                test_loss = test_loss_sum / total_samples

                epoch_iterator.set_description(f'S{self.nSub} E{e+1} | TL:{avg_train_loss:.4f} TA:{avg_train_acc:.4f} | T_loss:{test_loss:.4f} T_Acc:{test_acc:.4f}')

                if (e + 1) == 1 or (e + 1) == self.n_epochs or (e + 1) % log_interval == 0:
                    self.log_write.write(f"{e+1}    {test_acc}\n") 

                    wandb.log({
                        "Epoch": e + 1,
                        f"Subject {self.nSub}/Train Loss": avg_train_loss,
                        f"Subject {self.nSub}/Test Loss": test_loss,
                        f"Subject {self.nSub}/Train Accuracy": avg_train_acc,
                        f"Subject {self.nSub}/Test Accuracy": test_acc
                    })
                
                num += 1 
                averAcc += test_acc
                
                if test_acc > bestAcc:
                    bestAcc = test_acc
                    Y_true_collector = torch.cat(current_epoch_test_labels).cpu()
                    Y_pred_collector = torch.cat(current_epoch_test_preds).cpu()

        # Define model save directory based on experiment_output_dir
        model_save_dir = os.path.join(self.experiment_output_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True) # Create if it doesn't exist
        model_save_path = os.path.join(model_save_dir, f'model_subject{self.nSub}_epoch{self.n_epochs}_gpu{gpus[0]}.pth')

        if isinstance(self.model, nn.DataParallel):
            # If the model is wrapped by DataParallel, access the original model via .module
            torch.save(self.model.module.state_dict(), model_save_path)
        else:
            # Otherwise, it's the original model instance
            torch.save(self.model.state_dict(), model_save_path)


        averAcc = averAcc / num
        tqdm.write(f'Subject {self.nSub} - The average accuracy is: {averAcc:.6f}')
        tqdm.write(f'Subject {self.nSub} - The best accuracy is: {bestAcc:.6f}')
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        self.log_write.close()

        return bestAcc, averAcc, Y_true_collector, Y_pred_collector

# ###########################################################################################
def main():
    # Generate a unique timestamp for the experiment run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct the full experiment output directory path
    # args.output_base_dir will be like './experiment_outputs'
    # full_experiment_output_dir will be like './experiment_outputs/20250722_143000'
    full_experiment_output_dir = os.path.join(args.output_base_dir, f"run_{timestamp}")
    os.makedirs(full_experiment_output_dir, exist_ok=True) # Create the main experiment directory
    
    # Initialize Weights & Biases run
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args,
               name=f"run_{timestamp}_gpu{args.gpu}") # Add run name for easier identification
    wandb.config.update({
        "batch_size": 72,
        "n_epochs": 2000,
        "lr": 0.0002,
        "model": "Conformer",
        "tmin": 0,
        "tmax": 4,
        "frq_low": 4,
        "frq_high": 40,
        "gpus_used": gpus, 
        "experiment_output_dir": full_experiment_output_dir # Log the output directory
    })

    total_best_accuracy = 0
    total_avg_accuracy = 0
    
    # Define overall result file path within the new experiment directory
    overall_result_file_path = os.path.join(full_experiment_output_dir, "overall_results.txt")
    result_write = open(overall_result_file_path, "w")


    tmin = 0
    tmax = 4
    frq_low = 4
    frq_high = 40

    all_subjects_Y_true = []
    all_subjects_Y_pred = []

    for i in tqdm(range(9), desc="Overall Subject Progress"):
        starttime = datetime.datetime.now()

        seed_n = 3407 
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        cha_tag = f'{tmin}-{tmax}s_{frq_low}-{frq_high}hz'
        
        # Pass the experiment_output_dir to ExP constructor
        exp = ExP(i + 1, cha_tag, full_experiment_output_dir,args.nepoch) 

        bestAcc, averAcc, Y_true_subject, Y_pred_subject = exp.train()
        
        result_write.write(f'Subject {i + 1} : Seed is: {seed_n}\n')
        result_write.write(f'Subject {i + 1} : The best accuracy is: {bestAcc:.6f}\n')
        result_write.write(f'Subject {i + 1} : The average accuracy is: {averAcc:.6f}\n')

        wandb.log({
            f"Subject {i+1}/Best Test Accuracy": bestAcc,
            f"Subject {i+1}/Average Test Accuracy": averAcc,
            f"Subject {i+1}/Duration_seconds": (datetime.datetime.now() - starttime).total_seconds()
        })
        
        total_best_accuracy += bestAcc
        total_avg_accuracy += averAcc
        
        all_subjects_Y_true.append(Y_true_subject)
        all_subjects_Y_pred.append(Y_pred_subject)

    final_avg_best_acc = total_best_accuracy / 9
    final_avg_avg_acc = total_avg_accuracy / 9

    result_write.write(f'**The overall average Best accuracy is: {final_avg_best_acc:.6f}\n')
    result_write.write(f'The overall average Aver accuracy is: {final_avg_avg_acc:.6f}\n')
    result_write.close()

    wandb.log({
        "Overall Average Best Accuracy": final_avg_best_acc,
        "Overall Average Average Accuracy": final_avg_avg_acc
    })

    wandb.finish()


if __name__ == "__main__":
    tqdm.write(f"Start time: {time.asctime(time.localtime(time.time()))}")
    main()
    tqdm.write(f"End time: {time.asctime(time.localtime(time.time()))}")