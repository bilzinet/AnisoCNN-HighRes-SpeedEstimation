# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:41:26 2020

@author: btt1

Testing Deep CNN model (based on Summer 2019 work)

"""

# =============================================================================
# Import libraries
# =============================================================================

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['text.usetex'] = False
np.set_printoptions(precision=4, suppress=True)

# =============================================================================
# Some helper functions
# =============================================================================

def rmse_cal(y_test, predY):
    
    rmse_array = []
    for i in range(y_test.shape[0]):
        true_y = y_test[i,:h,:,:].reshape(-1)
        pred_y = predY[i,:h,:,:].reshape(-1)
        rmse = np.round(np.sqrt(np.mean(np.power(true_y-pred_y, 2)))*max_speed, 3)
        rmse_array.append(rmse)
        
    return rmse_array


# =============================================================================
# Load model and data
# =============================================================================

# Load data
c=3; h=80; w=60
max_speed=100
x_test = np.array([]); y_test = np.array([])
datasets = ['Wide_free_sp5p','Wide_cong_sp5p','Wide_cong_more_sp5p']
for df in datasets:
    for i in [1,2,3,4,5,6]:    
        if (i>3) and (df!=datasets[0]):
            continue
        input_X = np.load('./5_Testing data/{}/inp_data_{}{}.npy'.format(df,df,i))
        nonempty_indx = np.load('./5_Testing data/{}/noempty_indx_{}{}.npy'.format(df,df,i))
        output = np.load('./5_Testing data/{}/out_data_{}.npy'.format(df,df[:-5]))
        output_Y = output[nonempty_indx.reshape(-1),:,:,:]
        if len(x_test) == 0:
            x_test = input_X[:, :, :, :3]
            y_test = output_Y[:, :, :, :]
        else:
            x_test = np.append(x_test, input_X[:, :, :, :3], axis=0)
            y_test = np.append(y_test, output_Y[:, :, :, :], axis=0)

# =============================================================================
# Single model summary and performance
# =============================================================================

# Load models
file_name = 'Expt22_CNNModel_Isotropic-hypopt-1.h5'
tse_cnn = keras.models.load_model("./3_Training experiments/expt-21 (hyperopt)/Run 5/{}".format(file_name),
                                   compile=False)
# Model summary
print('\nNumber of filters in each conv layer:')
for i in range(0,13,2):
    print('\t layer {}: '.format(i), tse_cnn.layers[i].filters)
print('\nKernel size in each conv layer:')
for i in range(0,13,2):
    print('\t layer {}: '.format(i), tse_cnn.layers[i].kernel_size)
print('\nModel summary')
tse_cnn.summary()

# Predict test samples
predY = tse_cnn.predict(x_test)

# Root mean squared error
rmse_array = rmse_cal(y_test, predY)
print('Root mean squared error: {} (+/-{})'.format(np.mean(rmse_array), np.std(rmse_array)))


# =============================================================================
# Multiple models analysis:
# =============================================================================

# Model names
file_names = ['Expt22_CNNModel_Isotropic-hypopt-1.h5',
              'Expt22_CNNModel_Isotropic-hypopt-2.h5',
              'Expt22_CNNModel_Isotropic-hypopt-3.h5',
              'Expt22_CNNModel_Isotropic-hypopt-4.h5',
              'Expt22_CNNModel_Isotropic-hypopt-5.h5',
              'Expt22_CNNModel_Isotropic-hypopt-6.h5']

# Compare models:
print('\nCompare all models')
model_filters = []; model_kernels = []
fig, axs = plt.subplots(2,1, figsize=(6,6), sharex=True)
x_coords = np.array(list(range(0,13,2))); x_w=-0.2
axs[0].grid(); axs[1].grid()
for f in file_names:
    tse_cnn = keras.models.load_model("./3_Training experiments/expt-21 (hyperopt)/Run 5/{}".format(f),
                                      compile=False)
    filters = []; kernels = []
    for i in range(0,13,2):
        filters.append(tse_cnn.layers[i].filters)
        kernels.append(tse_cnn.layers[i].kernel_size[0])
    axs[0].bar(x_coords+x_w, filters, width=0.2)
    axs[1].bar(x_coords+x_w, kernels, width=0.2)
    model_filters.append(filters); model_kernels.append(kernels)
    x_w += 0.2
axs[0].set_ylabel('Number of filters', fontsize=12)
axs[1].set_ylabel('Kernel size', fontsize=12)
axs[1].set_xlabel('Convolutional layers', fontsize=12)
fig.tight_layout()

# Add average to nearest multplies
model_filters = np.array(model_filters)
model_kernels = np.array(model_kernels)
print(model_filters.mean(axis=0))
print(model_kernels.mean(axis=0))
filter_base = 8
kernel_base = 2
avg_filters = [40, 48, 32, 48, 40, 56, 1]
avg_kernels = [5, 7, 7, 5, 5, 9, 7]

# RMSE for all models
print('\nRMSE for all models:')
models = []; model_preds = []; models_rmse = []
for i,f in enumerate(file_names):
    tse_cnn = keras.models.load_model("./3_Training experiments/expt-21 (hyperopt)/Run 5/{}".format(f),
                                      compile=False)
    predY = tse_cnn.predict(x_test)
    rmse = rmse_cal(y_test, predY)
    models.append(tse_cnn)
    model_preds.append(predY)
    models_rmse.append(rmse)
    print('\tModel {}: {} (+/- {})'.format(i+1, np.mean(rmse), np.std(rmse)))

plt.figure(figsize=(6,5))
plt.xlabel('Models', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.errorbar(list(range(1,5)), [np.mean(m_rmse) for m_rmse in models_rmse], 
             yerr=[np.std(m_rmse) for m_rmse in models_rmse],
             capsize=1)
plt.ylim([0,20])



