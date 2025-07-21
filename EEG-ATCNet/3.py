
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
from tqdm import tqdm # Import tqdm

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%%
def load_BCI2a_data(data_path, subject, training, all_trials=True):
    """ Loading and Dividing of the data set based on the subject-specific 
    (subject-dependent) approach.
    In this approach, we used the same training and testing data as the original
    competition, i.e., 288 x 9 trials in session 1 for training, 
    and 288 x 9 trials in session 2 for testing. 
    
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts 
    """
    
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6 * 48 
    window_Length = 7 * 250 
    
    # Define MI trial window 
    fs = 250            # sampling rate
    t1 = int(1.5 * fs)  # start time_point
    t2 = int(6 * fs)    # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if(a_artifacts[trial] != 0 and not all_trials):
                continue
            data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial +=1     
    
    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)

    return data_return, class_return

#%%
def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    # For PyTorch, we typically expect [Batch, Channels, Height, Width]
    # Here, MI-tasks is 1, so [Trials, 1, Channels, Time points]
    
    # Convert to numpy for StandardScaler
    X_train_np = X_train.squeeze(1).numpy() # Remove the MI-tasks dimension for scaling
    X_test_np = X_test.squeeze(1).numpy()

    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train_np[:, j, :])
        X_train_np[:, j, :] = scaler.transform(X_train_np[:, j, :])
        X_test_np[:, j, :] = scaler.transform(X_test_np[:, j, :])

    # Add back the MI-tasks dimension and convert to tensor
    X_train = torch.from_numpy(X_train_np).unsqueeze(1)
    X_test = torch.from_numpy(X_test_np).unsqueeze(1)

    return X_train, X_test

#%%
def get_data(path, subject, dataset='BCI2a', classes_labels='all', LOSO=False, isStandard=True, isShuffle=True):
    
    # Load and split the dataset into training and testing 
    if LOSO:
        raise NotImplementedError("LOSO evaluation is not implemented in this PyTorch conversion.")
    else:
        if (dataset == 'BCI2a'):
            X_train, y_train = load_BCI2a_data(path, subject + 1, True)
            X_test, y_test = load_BCI2a_data(path, subject + 1, False)
        # elif (dataset == 'CS2R'):
        #     X_train, y_train, _, _, _ = load_CS2R_data_v2(path, subject, True, classes_labels)
        #     X_test, y_test, _, _, _ = load_CS2R_data_v2(path, subject, False, classes_labels)
        # elif (dataset == 'HGD'):
        #     X_train, y_train = load_HGD_data(path, subject+1, True)
        #     X_test, y_test = load_HGD_data(path, subject+1, False)
        else:
            raise Exception(f"'{dataset}' dataset is not supported yet!")

    # Prepare training data 
    N_tr, N_ch, T = X_train.shape 
    X_train = torch.from_numpy(X_train).float().reshape(N_tr, 1, N_ch, T) # Add MI-task dim, convert to float tensor
    y_train = torch.from_numpy(y_train).long() # Convert to long tensor for labels
    y_train_onehot = F.one_hot(y_train, num_classes=len(np.unique(y_train))).float() # Convert to one-hot, then float
    
    # Prepare testing data 
    N_ts, N_ch, T = X_test.shape 
    X_test = torch.from_numpy(X_test).float().reshape(N_ts, 1, N_ch, T) # Add MI-task dim, convert to float tensor
    y_test = torch.from_numpy(y_test).long() # Convert to long tensor for labels
    y_test_onehot = F.one_hot(y_test, num_classes=len(np.unique(y_test))).float() # Convert to one-hot, then float
    
    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    # shuffle the data (PyTorch DataLoader can handle shuffling, but we'll do it here for consistency)
    if isShuffle:
        # Create permuted indices
        train_indices = torch.randperm(X_train.size(0))
        test_indices = torch.randperm(X_test.size(0))
        
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        y_train_onehot = y_train_onehot[train_indices]
        
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]
        y_test_onehot = y_test_onehot[test_indices]

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


class ShallowConvNet(nn.Module):
    def __init__(self, nb_classes, Chans=22, Samples=1125, dropoutRate=0.5):
        super(ShallowConvNet, self).__init__()
        self.Chans = Chans
        self.Samples = Samples
        
        self.conv1 = nn.Conv2d(1, 40, (1, 25), padding='same')
        # Kernel constraint will be applied manually after optimization step for these layers
        self.conv2 = nn.Conv2d(40, 40, (Chans, 1), bias=False) 
        self.batchnorm = nn.BatchNorm2d(40, eps=1e-05, momentum=0.9)
        self.avg_pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropoutRate)
        self.dense = nn.Linear(self.get_flat_feats(), nb_classes) # Adjusted later if input size changes
        
        # Max-norm constraint for dense layer will be applied manually

    def forward(self, x):
        # x shape: [batch_size, 1, Chans, Samples]
        
        # Block 1
        x = self.conv1(x) 
        x = self.conv2(x) 
        
        x = self.batchnorm(x)
        x = torch.pow(x, 2) # Square activation
        x = self.avg_pool(x)
        x = torch.log(x.clamp(min=1e-6)) # Log activation, clamp to avoid log(0)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.dense(x)
        return x

    def get_flat_feats(self):
        # This method is to dynamically calculate the input size for the dense layer
        # Create a dummy input to trace its size through the convolutional layers
        dummy_input = torch.zeros(1, 1, self.Chans, self.Samples)
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), -1).size(1)

# Custom constraint function for max_norm. This will be applied manually.
def max_norm_constraint(model, max_val=2.0, layer_type=nn.Conv2d, dim_axis=(0,1,2)):
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            with torch.no_grad():
                if len(module.weight.shape) == 4: # Conv2d
                    # Simplified interpretation for this translation: constrain the L2 norm of the entire kernel weights tensor.
                    norm = torch.norm(module.weight.data)
                    if norm > max_val:
                        module.weight.data = module.weight.data * (max_val / norm)
                elif len(module.weight.shape) == 2: # Linear (Dense) layer
                    norm = torch.norm(module.weight.data, p=2, dim=0, keepdim=True) # Norm for each incoming connection (column)
                    if (norm > max_val).any():
                        module.weight.data = module.weight.data * (max_val / norm.clamp(min=1e-6))
                        

def square(x):
    return torch.pow(x, 2)

def log(x):
    return torch.log(x.clamp(min=1e-6)) # Clamp to avoid log(0)


def draw_learning_curves(history, sub):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f'Model accuracy - subject: {sub}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'Model loss - subject: {sub}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
    # Generate confusion matrix plot
    display_labels = classes_labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
                                  display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title(f'Confusion Matrix of Subject: {sub}')
    plt.savefig(os.path.join(results_path, f'subject_{sub}.png'))
    plt.show()
    plt.close()

def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(1, num_sub + 1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title(f'Model {label} per subject')
    ax.set_ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def train(dataset_conf, train_conf, results_path):
    # Remove the 'results' folder before training
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    os.makedirs(os.path.join(results_path, 'saved models1'))

    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    best_models_file = os.path.join(results_path, "best models1.txt")
    best_models = open(best_models_file, "w")
    # Create a file to store performance during training
    log_file = os.path.join(results_path, "log1.txt")
    log_write = open(log_file, "w")
    
    # Get dataset parameters
    dataset = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    n_classes = dataset_conf.get('n_classes')

    # Get training hyperparamters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience') # For early stopping
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves') # Plot Learning Curves?
    n_train = train_conf.get('n_train') # Number of repetitions
    model_name = train_conf.get('model')
    # from_logits is handled by CrossEntropyLoss in PyTorch

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))
    
    # Iteration over subjects 
    for sub in range(n_sub):
        print(f'\nTraining on subject {sub + 1}')
        log_write.write(f'\nTraining on subject {sub + 1}\n')
        
        # Initiating variables to save the best subject accuracy among multiple runs.
        BestSubjAcc = 0 
        bestTrainingHistory = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        
        # Get training data
        X_data, y_data, y_data_onehot, _, _, _ = get_data(
            data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard)
            
        # Divide the training data into training and validation
        X_train_split, X_val_split, y_train_split, y_val_split, y_train_onehot_split, y_val_onehot_split = train_test_split(
            X_data, y_data, y_data_onehot, test_size=0.2, random_state=42)

        # Create DataLoader for training and validation sets
        train_dataset = TensorDataset(X_train_split, y_train_split)
        val_dataset = TensorDataset(X_val_split, y_val_split)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Iteration over multiple runs 
        for run_idx in range(n_train):
            # Set the random seed for reproducibility.
            torch.manual_seed(run_idx + 1)
            np.random.seed(run_idx + 1)
            
            # Get the current 'IN' time to calculate the 'run' training time
            in_run = time.time()
            
            # Create folders and files to save trained models for all runs
            run_filepath_dir = os.path.join(results_path, f'saved models1/run-{run_idx + 1}')
            os.makedirs(run_filepath_dir, exist_ok=True)
            filepath = os.path.join(run_filepath_dir, f'subject-{sub + 1}.pth') # Using .pth for PyTorch models
            
            # Create the model
            model = getModel(model_name, dataset_conf).to(device)
            
            # Compile and train the model
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss() # PyTorch CrossEntropyLoss expects logits and integer labels
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.90, patience=20, min_lr=0.0001)

            best_val_loss = float('inf')
            epochs_no_improve = 0
            
            history_run = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

            # Wrap the epoch loop with tqdm for overall progress
            for epoch in tqdm(range(epochs), desc=f"Sub {sub+1} Run {run_idx+1} Epoch"):
                model.train()
                total_loss = 0
                correct_train = 0
                total_train = 0
                
                # Wrap the batch loop with tqdm for batch-level progress
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    # Apply max_norm constraint after optimizer step
                    max_norm_constraint(model, max_val=2.0, layer_type=nn.Conv2d)
                    max_norm_constraint(model, max_val=0.5, layer_type=nn.Linear, dim_axis=0)

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += batch_y.size(0)
                    correct_train += (predicted == batch_y).sum().item()
                
                train_accuracy = correct_train / total_train
                train_loss_avg = total_loss / len(train_loader)
                
                # Validation
                model.eval()
                val_loss = 0
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                        outputs_val = model(batch_X_val)
                        loss_val = criterion(outputs_val, batch_y_val)
                        val_loss += loss_val.item()
                        _, predicted_val = torch.max(outputs_val.data, 1)
                        total_val += batch_y_val.size(0)
                        correct_val += (predicted_val == batch_y_val).sum().item()
                
                val_accuracy = correct_val / total_val
                val_loss_avg = val_loss / len(val_loader)

                history_run['accuracy'].append(train_accuracy)
                history_run['val_accuracy'].append(val_accuracy)
                history_run['loss'].append(train_loss_avg)
                history_run['val_loss'].append(val_loss_avg)

                scheduler.step(val_loss_avg)

                # Update tqdm description with current epoch's metrics
                tqdm.write(f"Epoch {epoch+1} - Train Loss: {train_loss_avg:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}")

                # Save best model based on validation loss
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), filepath)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        tqdm.write(f"Early stopping at epoch {epoch+1} for Sub {sub+1} Run {run_idx+1}")
                        break # Early stop

            # Load the best model weights for evaluation
            model.load_state_dict(torch.load(filepath))
            model.eval()
            
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs = model(batch_X_val)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y_val.cpu().numpy())
            
            acc[sub, run_idx] = accuracy_score(all_labels, all_preds)
            kappa[sub, run_idx] = cohen_kappa_score(all_labels, all_preds)
                                
            # Get the current 'OUT' time to calculate the 'run' training time
            out_run = time.time()
            # Print & write performance measures for each run
            info = f'Subject: {sub + 1}   seed {run_idx + 1}   time: {((out_run - in_run) / 60):.1f} m   '
            info += f'valid_acc: {acc[sub, run_idx]:.4f}   valid_loss: {best_val_loss:.3f}'
            print(info)
            log_write.write(info + '\n')
            
            # If current training run is better than previous runs, save the history.
            if(BestSubjAcc < acc[sub, run_idx]):
                BestSubjAcc = acc[sub, run_idx]
                # Deep copy the history to avoid reference issues
                bestTrainingHistory = {k: list(v) for k, v in history_run.items()}
        
        # Store the path of the best model among several runs
        best_run_for_subject = np.argmax(acc[sub,:])
        filepath_best_run = os.path.join('/saved models1', f'run-{best_run_for_subject + 1}', f'subject-{sub + 1}.pth') + '\n'
        best_models.write(filepath_best_run)

        # Plot Learning curves 
        if (LearnCurves == True):
            print('Plot Learning Curves ....... ')
            draw_learning_curves(bestTrainingHistory, sub + 1)
            
    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
            
    # Print & write the validation performance using all seeds
    head1 = head2 = '         '
    for s_idx in range(n_sub): 
        head1 += f'sub_{s_idx + 1}   '
        head2 += '-----   '
    head1 += '   average'
    head2 += '   -------'
    info = '\n---------------------------------\nValidation performance (acc %):'
    info += '\n---------------------------------\n' + head1 +'\n'+ head2
    for r_idx in range(n_train): 
        info += f'\nSeed {r_idx + 1}:   '
        for s_idx in range(n_sub): 
            info += f'{acc[s_idx, r_idx]*100:.2f}   '
        info += f'   {np.average(acc[:, r_idx])*100:.2f}   '
    info += '\n---------------------------------\nAverage acc - all seeds: '
    info += f'{np.average(acc)*100:.2f} %\n\nTrain Time   - all seeds: {((out_exp - in_exp) / 60):.1f}'
    info += ' min\n---------------------------------\n'
    print(info)
    log_write.write(info + '\n')

    # Close open files 
    best_models.close()  
    log_write.close() 
    
def getModel(model_name, dataset_conf):
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

    if model_name == 'ShallowConvNet':
        model = ShallowConvNet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model





def test(dataset_conf, results_path, train_conf):
    # Open the "Log" file to write the evaluation results 
    log_write = open(os.path.join(results_path, "log.txt"), "a")
    
    # Get dataset parameters
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    classes_labels = dataset_conf.get('cl_labels')
    
    # Test the performance based on several runs (seeds)
    runs_dir = os.path.join(results_path, "saved models1") # Corrected path
    runs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]) # Ensure only directories and sorted
    
    # Initialize variables
    acc = np.zeros((n_sub, len(runs)))
    kappa = np.zeros((n_sub, len(runs)))
    cf_matrix = np.zeros([n_sub, len(runs), n_classes, n_classes])

    # Create a model instance once
    model = getModel(train_conf.get('model'), dataset_conf).to(device)

    inference_times_per_trial = [] # To store inference time for each test set

    # Iteration over subjects 
    for sub in range(n_sub):
        print(f'\nTesting on subject {sub + 1}')
        log_write.write(f'\nTesting on subject {sub + 1}\n')

        # Load test data
        _, _, _, X_test, y_test, _ = get_data(data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard)
        
        # Move test data to device
        X_test = X_test.to(device)
        y_test_np = y_test.cpu().numpy() # Convert true labels to numpy once

        # Iteration over runs (seeds) 
        for seed_idx, run_folder in enumerate(runs): 
            # Load the model weights for the current seed.
            filepath = os.path.join(runs_dir, run_folder, f'subject-{sub + 1}.pth')
            
            if not os.path.exists(filepath):
                print(f"Warning: Model weights not found for subject {sub+1} in run {run_folder}. Skipping.")
                continue

            model.load_state_dict(torch.load(filepath, map_location=device))
            model.eval() # Set model to evaluation mode

            # Predict MI task
            inference_time_start = time.time()
            with torch.no_grad():
                outputs = model(X_test)
                _, y_pred_tensor = torch.max(outputs.data, 1) # Get predicted class indices
            inference_time_end = time.time()
            
            y_pred_np = y_pred_tensor.cpu().numpy() # Move predictions to CPU and convert to numpy
            
            inference_time_per_trial = (inference_time_end - inference_time_start) / X_test.shape[0]
            inference_times_per_trial.append(inference_time_per_trial)

            # Calculate accuracy and K-score 
            acc[sub, seed_idx] = accuracy_score(y_test_np, y_pred_np)
            kappa[sub, seed_idx] = cohen_kappa_score(y_test_np, y_pred_np)
            
            # Calculate and store confusion matrix
            cf_matrix[sub, seed_idx, :, :] = confusion_matrix(y_test_np, y_pred_np, normalize='true')
            
            # Optionally draw confusion matrix for each subject/seed (commented out as per original)
            # draw_confusion_matrix(cf_matrix[sub, seed_idx, :, :], f"Sub {sub+1} Run {seed_idx+1}", results_path, classes_labels)
            
            info = f'Subject: {sub+1} Run: {run_folder} Test Acc: {acc[sub, seed_idx]:.4f} Kappa: {kappa[sub, seed_idx]:.4f}'
            print(info)
            log_write.write(info + '\n')

    # Print & write the average performance measures for all subjects 
    head1 = head2 = '                     '
    for s_idx in range(n_sub): 
        head1 += f'sub_{s_idx + 1}   '
        head2 += '-----   '
    head1 += '   average'
    head2 += '   -------'
    info = '\n---------------------------------\nTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTest performance (acc & k-score):\n'
    info += '---------------------------------\n' + head1 +'\n'+ head2
    for r_idx, run_folder in enumerate(runs): 
        info += f'\nRun {r_idx + 1}: '
        info_acc = '(acc %)   '
        info_k = '          (k-sco)   '
        for s_idx in range(n_sub): 
            info_acc += f'{acc[s_idx, r_idx]*100:.2f}   '
            info_k += f'{kappa[s_idx, r_idx]:.3f}   '
        info_acc += f'   {np.average(acc[:, r_idx])*100:.2f}   '
        info_k += f'   {np.average(kappa[:, r_idx]):.3f}   '
        info += info_acc + '\n' + info_k
    info += '\n----------------------------------\nAverage - all seeds (acc %): '
    info += f'{np.average(acc)*100:.2f}\n          (k-sco): '
    info += f'{np.average(kappa):.3f}\n\nInference time: {np.mean(inference_times_per_trial) * 1000:.2f}'
    info += ' ms per trial\n----------------------------------\n'
    print(info)
    log_write.write(info+'\n')
            
    # Draw a performance bar chart for all subjects 
    draw_performance_barChart(n_sub, acc.mean(axis=1), 'Accuracy') # Use axis=1 for mean across runs
    draw_performance_barChart(n_sub, kappa.mean(axis=1), 'k-score') # Use axis=1 for mean across runs
    
    # Draw confusion matrix for all subjects (average across subjects and runs)
    # Ensure cf_matrix is not empty before attempting to average
    if cf_matrix.size > 0:
        draw_confusion_matrix(cf_matrix.mean(axis=(0,1)), 'All', results_path, classes_labels) # Mean across subjects and runs
    else:
        print("No confusion matrix to draw, cf_matrix is empty.")
    
    # Close opened file    
    log_write.close() 









def run():
    in_samples = 1125
    n_channels = 22
    n_sub = 9
    n_classes = 4
    classes_labels = ['Left hand', 'Right hand','Foot','Tongue']
    data_path = "C:/Users/13613/Desktop/Naoji/EEG-ATCNet/data/"
    results_path = os.getcwd() + "/results_pytorch_gpu2" # Changed results folder name

    # Set dataset paramters 
    dataset_conf = { 'name': 'BCI2a', 'n_classes': n_classes, 'cl_labels': classes_labels,
                     'n_sub': n_sub, 'n_channels': n_channels, 'in_samples': in_samples,
                     'data_path': data_path, 'isStandard': True, 'LOSO': False}
    # Set training hyperparamters
    train_conf = { 'batch_size': 32, 'epochs': 500, 'patience': 100, 'lr': 0.001,'n_train': 1,
                   'LearnCurves': False, 'from_logits': False, 'model':'ShallowConvNet'}
            
    # Train the model
    train(dataset_conf, train_conf, results_path)


    test(dataset_conf, results_path, train_conf) # Pass train_conf to test for getModel


#%%
if __name__ == "__main__":
    run()