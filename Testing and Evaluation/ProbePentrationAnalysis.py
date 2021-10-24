# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:41:26 2020

@author: btt1

Probe penetration analysis:
    
    Measure performance of CNN models trained on each probe penetration
    Compare with generic models trained on:
        (i) All the data in equal proportions
        (ii) All the data in unequal proportions
        (iii) Ensemble average

"""

# =============================================================================
# Import libraries
# =============================================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
np.set_printoptions(precision=4, suppress=True)

from matplotlib import rcParams
plt.rcParams["font.family"] = "Times New Roman"
rcParams['mathtext.fontset']='cm'


# =============================================================================
# Model classes and functions
# =============================================================================

class Tilted_kernel(tf.keras.constraints.Constraint):
    '''
    Class of new tilter kernels
    '''
    def __init__(self, mask):
        self.mask = mask
    
    def __call__(self, w):
        return w * math_ops.cast(self.mask, K.floatx())
    
    def get_config(self):
        return {'mask':self.mask}

class Group_regularizer(tf.keras.regularizers.Regularizer):
    '''
    Class of Group LASSO regularizer for CNN structures - kernels and channels.
    '''
    def __init__(self, l2_ker=0., l2_cha=0.):
        self.l2_ker = l2_ker
        self.l2_cha = l2_cha
        
    def __call__(self, w):
        ker_norm = tf.math.reduce_sum(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(w),axis=[0,1,2])))
        cha_norm = tf.math.reduce_sum(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(w),axis=[0,1,3])))
        ker_reg = self.l2_ker*ker_norm
        cha_reg = self.l2_cha*cha_norm
        return ker_reg + cha_reg

    def get_config(self):
        return {'l2_ker': float(self.l2_ker),'l2_cha': float(self.l2_cha)}


# =============================================================================
# Load testing data
# =============================================================================

# Dataset parameters
datasets = ['free','cong','cong_more']
probe_per = np.array([5,10,20,30,40,50,60,70])
dataset_num = np.array([1,2,3,4,5,6])
inputs = []; outputs = []

# Loop over each probe penetration
for p_i, p in enumerate(probe_per):
    
    x_test = np.array([])
    y_test = np.array([])

    # Loop over each data type (cong, free-flow)
    for d_i, d in enumerate(datasets):
        
        # Load the output data
        output = np.load('./5_Testing data/Wide_{0}_sp5p/out_data_Wide_{0}.npy'.format(d))
    
        # Loop over different data instances of same type.
        for n_i, n in enumerate(dataset_num):
            
            # Only for free-flow datasets
            if (n>3) and (d!=datasets[0]):
                continue
            if (n>3) and (p>70) and (d==datasets[0]):
                continue
            
            # Load input and output data for the required sample size
            input_X = np.load('./5_Testing data/Wide_{0}_sp{1}p/inp_data_Wide_{0}_sp{1}p{2}.npy'.format(d,p,n))
            noempty_indx = np.load('./5_Testing data/Wide_{0}_sp{1}p/noempty_indx_Wide_{0}_sp{1}p{2}.npy'.format(d,p,n))
            output_Y = output[noempty_indx.reshape(-1),:,:,:]
            
            # Record the data in the training and testing array.
            if len(x_test) == 0:
                x_test = input_X[:, :, :, :3]
                y_test = output_Y[:, :, :, :]
            else:
                x_test = np.append(x_test, input_X[:, :, :, :3], axis=0)
                y_test = np.append(y_test, output_Y[:, :, :, :], axis=0)

    inputs.append(x_test)
    outputs.append(y_test)


# =============================================================================
# Load models and predict over test samples
# =============================================================================

# -------------- Load model and predict -------------- #

# Empty arrays
model_p_preds = []      # For probe specific model
model_g1_preds = []     # For generic model trained on equal proportioned data
model_g2_preds = []     # For generic model trained on unequal proportioned data
model_g3_preds = []     # For ensemble model


# Load the two generic model (trained over all dataset - equal and unequal proportion)
model_equprobe = load_model('./3_Training experiments/expt-27 (probe-complete)/Expt27_anisomodel-equprobe.h5',
                            custom_objects={'Tilted_kernel':Tilted_kernel}, 
                            compile=False)
model_invprobe = load_model('./3_Training experiments/expt-27 (probe-complete)/Expt27_anisomodel-invprobe.h5',
                            custom_objects={'Tilted_kernel':Tilted_kernel}, 
                            compile=False)


# ----------- Generic model prediction ------------- #
for p_i, p in enumerate(probe_per):
    
    # Load the probe specific model.
    model_probe = load_model('./3_Training experiments/expt-26 (probe10-80p)/Expt26_anisomodel_{}p.h5'.format(p),
                             custom_objects={'Tilted_kernel':Tilted_kernel}, 
                             compile=False)
    
    # Prediction
    model_p_predY = model_probe.predict(inputs[p_i])
    model_g1_predY = model_equprobe.predict(inputs[p_i])
    model_g2_predY = model_invprobe.predict(inputs[p_i])
    model_p_preds.append(model_p_predY)
    model_g1_preds.append(model_g1_predY)
    model_g2_preds.append(model_g2_predY)


# ------------- Ensemble average model prediction ------------- #

# For each probe test data
for p_i in range(len(probe_per)):
    
    model_avg_predYs = []
    # prediction over each probe specific model
    for p in probe_per:    
        # Load the probe specific model.
        model_p = load_model('./3_Training experiments//expt-26 (probe10-80p)/Expt26_anisomodel_{}p.h5'.format(p),
                                custom_objects={'Tilted_kernel':Tilted_kernel}, 
                                compile=False)
    
        # Prediction
        model_avg_predY = model_p.predict(inputs[p_i])
        model_avg_predYs.append(model_avg_predY)
    
    # Average predictions from multiple models
    model_avg_predYs = np.array(model_avg_predYs)
    model_g3_preds.append(np.mean(model_avg_predYs, axis=0))


# =============================================================================
# Performance analysis
# =============================================================================


# ----------------- Calculate RMSE ----------------- #

max_speed=100; h_act=80
model_p_rmse = []; model_g1_rmse = []
model_g2_rmse = []; model_g3_rmse = []
for i in range(len(model_p_preds)):
    true_array = outputs[i]
    pred_array1 = model_p_preds[i]; pred_array2 = model_g1_preds[i]
    pred_array3 = model_g2_preds[i]; pred_array4 = model_g3_preds[i]
    rmse_array1 = []; rmse_array2 = []
    rmse_array3 = []; rmse_array4 = []
    for s in range(true_array.shape[0]):
        true_y = true_array[s,:h_act,:,:].reshape(-1)
        pred_y1 = pred_array1[s,:h_act,:,:].reshape(-1)
        pred_y2 = pred_array2[s,:h_act,:,:].reshape(-1)
        pred_y3 = pred_array3[s,:h_act,:,:].reshape(-1)
        pred_y4 = pred_array4[s,:h_act,:,:].reshape(-1)
        rmse1 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y1)*max_speed, 2))), 3)
        rmse2 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y2)*max_speed, 2))), 3)
        rmse3 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y3)*max_speed, 2))), 3)
        rmse4 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y4)*max_speed, 2))), 3)
        rmse_array1.append(rmse1); rmse_array2.append(rmse2)
        rmse_array3.append(rmse3); rmse_array4.append(rmse4)
    model_p_rmse.append(rmse_array1)
    model_g1_rmse.append(rmse_array2)
    model_g2_rmse.append(rmse_array3)
    model_g3_rmse.append(rmse_array4)
    
    
# ----------------- Visualize mean + SD ----------------- #
    
# Plot mean + standard deviation rmse
mean_p = [np.mean(i) for i in model_p_rmse]
mean_g1 = [np.mean(i) for i in model_g1_rmse]
mean_g2 = [np.mean(i) for i in model_g2_rmse]
mean_g3 = [np.mean(i) for i in model_g3_rmse]
std_p = [np.std(i)/2 for i in model_p_rmse]
std_g1 = [np.std(i)/2 for i in model_g1_rmse]
std_g2 = [np.std(i)/2 for i in model_g2_rmse]
std_g3 = [np.std(i)/2 for i in model_g3_rmse]

x_p = np.linspace(1, 15, len(probe_per))
x_g1 = x_p + 0.6
x_g2 = x_p + 0.9
x_g3 = x_p + 0.3
x_loc = (x_p+x_g1+x_g2+x_g3)/4
# x_lab = [str(p)+'%' for p in probe_per]
x_lab = [str(p) for p in probe_per]


plt.figure(figsize=(4,4))
plt.errorbar(x_p, mean_p, yerr=std_p, 
             ecolor='tab:blue', elinewidth=1.3, capsize=3, 
             marker='o', mfc='white', mec='tab:blue', ms=5, mew=1.3, ls='')
plt.errorbar(x_g3, mean_g3, yerr=std_g3, capsize=3,
             ecolor='tab:orange', elinewidth=1.3, 
             marker='*', mfc='white', mec='tab:orange', ms=5, mew=1.3, ls='')
plt.errorbar(x_g2, mean_g2, yerr=std_g2, capsize=3,
             ecolor='tab:green', elinewidth=1.3, 
             marker='^', mfc='white', mec='tab:green', ms=5, mew=1.3, ls='')
plt.errorbar(x_g1, mean_g1, yerr=std_g1, capsize=3,
             ecolor='tab:red', elinewidth=1.3, 
             marker='+', mfc='white', mec='tab:red', ms=5, mew=1.3, ls='')
plt.xticks(x_loc, x_lab, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Test data PV penetration rate (%)", fontsize=18)
plt.ylabel("RMSE (km/hr)", fontsize=18)
plt.legend(['Probe specific model', 
            'Ensemble average model',
            'Generic model-uneq',
            'Generic model-eq'], fontsize=11, 
            loc='best')
# plt.title('Root Mean Squared Error', fontsize=14)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('Probe_analysis.pdf', bbox_inches='tight')


# =============================================================================
# Analysis with probe specific models
# =============================================================================

# Load models
model_5p = load_model('./3_Training experiments/expt-26 (probe10-80p)/Expt26_anisomodel_5p.h5',
                      custom_objects={'Tilted_kernel':Tilted_kernel}, 
                      compile=False)
model_70p = load_model('./3_Training experiments/expt-26 (probe10-80p)/Expt26_anisomodel_70p.h5',
                       custom_objects={'Tilted_kernel':Tilted_kernel}, 
                       compile=False)
model_30p = load_model('./3_Training experiments/expt-26 (probe10-80p)/Expt26_anisomodel_70p.h5',
                       custom_objects={'Tilted_kernel':Tilted_kernel}, 
                       compile=False)



# Predict samples
model_p_preds = []      # For probe specific model
model_p1_preds = []     # For 5p probe model
model_p2_preds = []     # For 30p probe model
model_p3_preds = []     # For 70p probe model


for p_i, p in enumerate(probe_per):
    
    # Load the probe specific model.
    model_probe = load_model('./3_Training experiments/expt-26 (probe10-80p)/Expt26_anisomodel_{}p.h5'.format(p),
                             custom_objects={'Tilted_kernel':Tilted_kernel}, 
                             compile=False)
    
    # Prediction
    model_p_predY = model_probe.predict(inputs[p_i])
    model_p1_predY = model_5p.predict(inputs[p_i])
    model_p2_predY = model_30p.predict(inputs[p_i])
    model_p3_predY = model_70p.predict(inputs[p_i])
    model_p_preds.append(model_p_predY)
    model_p1_preds.append(model_p1_predY)
    model_p2_preds.append(model_p2_predY)
    model_p3_preds.append(model_p3_predY)
    
    
# Calculate RMSE
max_speed=100; h_act=80
model_p_rmse = []; model_p1_rmse = []
model_p2_rmse = []; model_p3_rmse = []
for i in range(len(model_p_preds)):
    true_array = outputs[i]
    pred_array1 = model_p_preds[i]; pred_array2 = model_p1_preds[i]
    pred_array3 = model_p2_preds[i]; pred_array4 = model_p3_preds[i]
    rmse_array1 = []; rmse_array2 = []
    rmse_array3 = []; rmse_array4 = []
    for s in range(true_array.shape[0]):
        true_y = true_array[s,:h_act,:,:].reshape(-1)
        pred_y1 = pred_array1[s,:h_act,:,:].reshape(-1)
        pred_y2 = pred_array2[s,:h_act,:,:].reshape(-1)
        pred_y3 = pred_array3[s,:h_act,:,:].reshape(-1)
        pred_y4 = pred_array4[s,:h_act,:,:].reshape(-1)
        rmse1 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y1)*max_speed, 2))), 3)
        rmse2 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y2)*max_speed, 2))), 3)
        rmse3 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y3)*max_speed, 2))), 3)
        rmse4 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y4)*max_speed, 2))), 3)
        rmse_array1.append(rmse1); rmse_array2.append(rmse2)
        rmse_array3.append(rmse3); rmse_array4.append(rmse4)
    model_p_rmse.append(rmse_array1)
    model_p1_rmse.append(rmse_array2)
    model_p2_rmse.append(rmse_array3)
    model_p3_rmse.append(rmse_array4)

# Plot mean + standard deviation rmse
mean_p = [np.mean(i) for i in model_p_rmse]
mean_p1 = [np.mean(i) for i in model_p1_rmse]
mean_p2 = [np.mean(i) for i in model_p2_rmse]
mean_p3 = [np.mean(i) for i in model_p3_rmse]
std_p = [np.std(i)/2 for i in model_p_rmse]
std_p1 = [np.std(i)/2 for i in model_p1_rmse]
std_p2 = [np.std(i)/2 for i in model_p2_rmse]
std_p3 = [np.std(i)/2 for i in model_p3_rmse]

x_p = np.linspace(1, 15, len(probe_per))
x_p1 = x_p + 0.3
# x_p2 = x_p + 0.
x_p3 = x_p + 0.6
x_loc = (x_p+x_p1+x_p3)/3
# x_lab = [str(p)+'%' for p in probe_per]
x_lab = [str(p) for p in probe_per]


plt.figure(figsize=(4,4))
plt.errorbar(x_p, mean_p, yerr=std_p, 
             ecolor='tab:blue', elinewidth=1.3, capsize=3, 
             marker='o', mfc='white', mec='tab:blue', ms=5, mew=1.3, ls='')
plt.errorbar(x_p1, mean_p1, yerr=std_p1, capsize=3,
             ecolor='tab:orange', elinewidth=1.3, 
             marker='*', mfc='white', mec='tab:orange', ms=7, mew=1.3, ls='')
# plt.errorbar(x_p2, mean_p2, yerr=std_p2, capsize=3,
#              ecolor='tab:green', elinewidth=1.3, 
#              marker='*', mfc='white', mec='tab:green', ms=5, mew=1.3, ls='')
plt.errorbar(x_p3, mean_p3, yerr=std_p3, capsize=3,
             ecolor='tab:green', elinewidth=1.3, 
             marker='^', mfc='white', mec='tab:green', ms=6, mew=1.3, ls='')
plt.xticks(x_loc, x_lab, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Test data PV penetration rate (%)", fontsize=18)
plt.ylabel("RMSE (km/hr)", fontsize=18)
plt.legend(['Probe specific model', 
            '5% probe model',
            # '30% probe model',
            '70% probe model'], fontsize=12, 
            loc='best')
# plt.title('Root Mean Squared Error', fontsize=14)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('Probe_analysis1.pdf', bbox_inches='tight')

