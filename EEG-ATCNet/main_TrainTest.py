""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

#%%
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score

import models 
from preprocess import get_data
# from keras.utils.vis_utils import plot_model


#%%
def draw_learning_curves(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.close()

def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
    # Generate confusion matrix plot
    display_labels = classes_labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
                                display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix of Subject: ' + sub )
    plt.savefig(results_path + '/subject_' + sub + '.png')
    plt.show()

def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject')
    ax.set_ylim([0,1])
    
    
#%% Training 
def train(dataset_conf, train_conf, results_path):
    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")
    # Create a .npz file (zipped archive) to store the accuracy and kappa metrics 
    # for all runs (to calculate average accuracy/kappa over all runs)
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')
    
    # Get dataset paramters
    dataset = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    # Get training hyperparamters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves') # Plot Learning Curves?
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))
    
    # Iteration over subjects 
    # for sub in range(n_sub-1, n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
    for sub in range(n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Get the current 'IN' time to calculate the subject training time
        in_sub = time.time()
        print('\nTraining on subject ', sub+1)
        log_write.write( '\nTraining on subject '+ str(sub+1) +'\n')
        # Initiating variables to save the best subject accuracy among multiple runs.
        BestSubjAcc = 0 
        bestTrainingHistory = [] 
        # Get training and test data
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
            data_path, sub, dataset, LOSO = LOSO, isStandard = isStandard)  
        
        # Iteration over multiple runs 
        for train in range(n_train): # How many repetitions of training for subject i.
            # Get the current 'IN' time to calculate the 'run' training time
            tf.random.set_seed(train+1)
            np.random.seed(train+1)

            in_run = time.time()
            # Create folders and files to save trained models for all runs
            filepath = results_path + '/saved models/run-{}'.format(train+1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)        
            filepath = filepath + '/subject-{}.h5'.format(sub+1)
            
            # Create the model
            model = getModel(model_name, dataset_conf)
            # Compile and train the model
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])          
            # model.summary()
            # plot_model(model, to_file='plot_model.png', show_shapes=True, show_layer_names=True)
            
            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, 
                                save_best_only=True, save_weights_only=True, mode='max'),
                
                ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=1, min_lr=0.0001),  
                
                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]
            history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), 
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

            # Evaluate the performance of the trained model. 
            # Here we load the Trained weights from the file saved in the hard 
            # disk, which should be the same as the weights of the current model.
            model.load_weights(filepath)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train]  = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)
              
            # Get the current 'OUT' time to calculate the 'run' training time
            out_run = time.time()
            # Print & write performance measures for each run
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub+1, train+1, ((out_run-in_run)/60))
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc[sub, train], kappa[sub, train])
            print(info)
            log_write.write(info +'\n')
            # If current training run is better than previous runs, save the history.
            if(BestSubjAcc < acc[sub, train]):
                 BestSubjAcc = acc[sub, train]
                 bestTrainingHistory = history
        
        # Store the path of the best model among several runs
        best_run = np.argmax(acc[sub,:])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run+1, sub+1)+'\n'
        best_models.write(filepath)
        # Get the current 'OUT' time to calculate the subject training time
        out_sub = time.time()
        # Print & write the best subject performance among multiple runs
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub+1, best_run+1, ((out_sub-in_sub)/60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]), acc[sub,:].std() )
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub,:].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info+'\n')
        # Plot Learning curves 
        if (LearnCurves == True):
            print('Plot Learning Curves ....... ')
            draw_learning_curves(bestTrainingHistory)
          
    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format( (out_exp-in_exp)/(60*60) )
    print(info)
    log_write.write(info+'\n')
    
    # Store the accuracy and kappa metrics as arrays for all runs into a .npz 
    # file format, which is an uncompressed zipped archive, to calculate average
    # accuracy/kappa over all runs.
    np.savez(perf_allRuns, acc = acc, kappa = kappa)
    
    # Close open files 
    best_models.close()   
    log_write.close() 
    perf_allRuns.close() 


#%% Evaluation 
def test(model, dataset_conf, results_path, allRuns = False):
    # Open the  "Log" file to write the evaluation results 
    log_write = open(results_path + "/log.txt", "a")
    # Open the file that stores the path of the best models among several random runs.
    best_models = open(results_path + "/best models.txt", "r")   
    
    # Get dataset paramters
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    classes_labels = dataset_conf.get('cl_labels')
    
    # Initialize variables
    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)  
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])

    # Calculate the average performance (average accuracy and K-score) for 
    # all runs (experiments) for each subject.
    if(allRuns): 
        # Load the test accuracy and kappa metrics as arrays for all runs from a .npz 
        # file format, which is an uncompressed zipped archive, to calculate average
        # accuracy/kappa over all runs.
        perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
        perf_arrays = np.load(perf_allRuns)
        acc_allRuns = perf_arrays['acc']
        kappa_allRuns = perf_arrays['kappa']
    
    # Iteration over subjects 
    # for sub in range(n_sub-1, n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
    for sub in range(n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Load data
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, dataset, LOSO, isStandard)      
        
        # Load the best model out of multiple random runs (experiments).
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1])
        # Predict MI task
        y_pred = model.predict(X_test).argmax(axis=-1)
        # Calculate accuracy and K-score
        labels = y_test_onehot.argmax(axis=-1)
        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
        # Calculate and draw confusion matrix
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='true')
        draw_confusion_matrix(cf_matrix[sub, :, :], str(sub+1), results_path, classes_labels)
        
        # Print & write performance measures for each subject
        info = 'Subject: {}   best_run: {:2}  '.format(sub+1, (filepath[filepath.find('run-')+4:filepath.find('/sub')]) )
        info = info + 'acc: {:.4f}   kappa: {:.4f}   '.format(acc_bestRun[sub], kappa_bestRun[sub] )
        if(allRuns): 
            info = info + 'avg_acc: {:.4f} +- {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(
                np.average(acc_allRuns[sub, :]), acc_allRuns[sub,:].std(),
                np.average(kappa_allRuns[sub, :]), kappa_allRuns[sub,:].std() )
        print(info)
        log_write.write('\n'+info)
      
    # Print & write the average performance measures for all subjects     
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun)) 
    if(allRuns): 
        info = info + '\nAverage of {} subjects x {} runs (average of {} experiments):\nAccuracy = {:.4f}   Kappa = {:.4f}'.format(
            n_sub, acc_allRuns.shape[1], (n_sub * acc_allRuns.shape[1]),
            np.average(acc_allRuns), np.average(kappa_allRuns)) 
    print(info)
    log_write.write(info)
    
    # Draw a performance bar chart for all subjects 
    draw_performance_barChart(n_sub, acc_bestRun, 'Accuracy')
    draw_performance_barChart(n_sub, kappa_bestRun, 'K-score')
    # Draw confusion matrix for all subjects (average)
    draw_confusion_matrix(cf_matrix.mean(0), 'All', results_path, classes_labels)
    # Close open files     
    log_write.close() 
    
    
#%%
def getModel(model_name, dataset_conf):
    
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

    # Select the model
    if(model_name == 'ATCNet'):
        # Train using the proposed ATCNet model: https://doi.org/10.1109/TII.2022.3197419
        model = models.ATCNet_( 
            # Dataset parameters
            n_classes = n_classes, 
            in_chans = n_channels, 
            in_samples = in_samples, 
            # Sliding window (SW) parameter
            n_windows = 5, 
            # Attention (AT) block parameter
            attention = 'mha', # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1 = 16,
            eegn_D = 2, 
            eegn_kernelSize = 64,
            eegn_poolSize = 7,
            eegn_dropout = 0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth = 2, 
            tcn_kernelSize = 4,
            tcn_filters = 32,
            tcn_dropout = 0.3, 
            tcn_activation='elu'
            )     
    elif(model_name == 'TCNet_Fusion'):
        # Train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
        model = models.TCNet_Fusion(n_classes = n_classes, Chans=n_channels, Samples=in_samples)      
    elif(model_name == 'EEGTCNet'):
        # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
        model = models.EEGTCNet(n_classes = n_classes, Chans=n_channels, Samples=in_samples)          
    elif(model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = models.EEGNet_classifier(n_classes = n_classes, Chans=n_channels, Samples=in_samples) 
    elif(model_name == 'EEGNeX'):
        # Train using EEGNeX: https://arxiv.org/abs/2207.12369
        model = models.EEGNeX_8_32(n_timesteps = in_samples , n_features = n_channels, n_outputs = n_classes)
    elif(model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = models.DeepConvNet(nb_classes = n_classes , Chans = n_channels, Samples = in_samples)
    elif(model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = models.ShallowConvNet(nb_classes = n_classes , Chans = n_channels, Samples = in_samples)
    elif(model_name == 'MBEEG_SENet'):
        # Train using MBEEG_SENet: https://www.mdpi.com/2075-4418/12/4/995
        model = models.MBEEG_SENet(nb_classes = n_classes , Chans = n_channels, Samples = in_samples)

    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model
    
    
#%%
def run():
    # Define dataset parameters
    dataset = 'BCI2a' # Options: 'BCI2a','HGD', 'CS2R'
    
    if dataset == 'BCI2a': 
        in_samples = 1125
        n_channels = 22
        n_sub = 9
        n_classes = 4
        classes_labels = ['Left hand', 'Right hand','Foot','Tongue']
        # data_path = os.path.expanduser('~') + '/BCI Competition IV/BCI Competition IV-2a/BCI Competition IV 2a mat/'
        data_path = 'D:/EEG_dataset/BCICIV_2a_mat_official/'
    elif dataset == 'HGD': 
        in_samples = 1125
        n_channels = 44
        n_sub = 14
        n_classes = 4
        classes_labels = ['Right Hand', 'Left Hand','Rest','Feet']     
        data_path = os.path.expanduser('~') + '/mne_data/MNE-schirrmeister2017-data/robintibor/high-gamma-dataset/raw/master/data/'
    elif dataset == 'CS2R': 
        in_samples = 1125
        # in_samples = 576
        n_channels = 32
        n_sub = 18
        n_classes = 3
        # classes_labels = ['Fingers', 'Wrist','Elbow','Rest']     
        classes_labels = ['Fingers', 'Wrist','Elbow']     
        # classes_labels = ['Fingers', 'Elbow']     
        data_path = os.path.expanduser('~') + '/CS2R MI EEG dataset/all/EDF - Cleaned - phase one (remove extra runs)/two sessions/'
    else:
        raise Exception("'{}' dataset is not supported yet!".format(dataset))
        
    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not  os.path.exists(results_path):
      os.makedirs(results_path)   # Create a new directory if it does not exist 
      
    # Set dataset paramters 
    dataset_conf = { 'name': dataset, 'n_classes': n_classes, 'cl_labels': classes_labels,
                    'n_sub': n_sub, 'n_channels': n_channels, 'in_samples': in_samples,
                    'data_path': data_path, 'isStandard': True, 'LOSO': False}
    # Set training hyperparamters
    train_conf = { 'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.001,
                  'LearnCurves': True, 'n_train': 10, 'model':'ATCNet'}
           
    # Train the model
    # train(dataset_conf, train_conf, results_path)

    # Evaluate the model based on the weights saved in the '/results' folder
    model = getModel(train_conf.get('model'), dataset_conf)
    test(model, dataset_conf, results_path)    
    
#%%
if __name__ == "__main__":
    run()
    
