# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:41:26 2020
@author: btt1
Training of Deep CNN model (based on Summer 2019 work)

Hyper-parameter tuning
"""

# =============================================================================
# Import Libraries
# =============================================================================

import IPython
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import Hyperband
# from kerastuner.tuners import BayesianOptimization

# =============================================================================
# Tilted filter configuration
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
    
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)

def mask_def(slope, width, ker_size=7):
    '''
    Output mask matrix as a function of slope and width
    '''
    mask = np.zeros((ker_size,ker_size))
    ts = np.arange(ker_size) - (ker_size-1)/2 
    xs = ts * -10
    i = 0; j = 0
    for x in xs:
        j = 0
        for t in ts:
            if slope > 0:
                if slope*(t-width) <= x and x <= slope*(t+width):
                    mask[i,j] = 1
            else:
                if slope*(t-width) >= x and x >= slope*(t+width):
                    mask[i,j] = 1
            j = j + 1
        i = i + 1
    return mask

def create_mask(s1,w1,s2,w2,k1,k2,inp_dep,out_dep):
    
    ff_mask = mask_def(s1, w1, k1)
    cg_mask = mask_def(s2, w2, k2)
    
    com_mask = np.zeros((max(k1, k2), max(k1, k2)))
    com_mask = np.maximum(com_mask, ff_mask)
    com_mask = np.maximum(com_mask, cg_mask)
    
    ker_mask = np.zeros((max(k1,k2),max(k1,k2),inp_dep,out_dep), np.int32)
    for i in range(ker_mask.shape[2]):
        for j in range(ker_mask.shape[3]):
            ker_mask[:,:,i,j] = com_mask
    
    return ker_mask

# =============================================================================
# CNN Reconstruction model (Improved version)
# =============================================================================

def build_cnn_model(hp):

    # ----------- Input layer ----------- #
    h = 30; w = 60; inp_dep = 3
    cnn_model = keras.Sequential(name='TSECNN_Hypopt')
    cnn_model.add(layers.Input(shape=(h, w, inp_dep)))
    
    # --------- Encoder model ----------- #
    # Layer 1
    nrows = 3
    cnn_model.add(layers.Conv2D(filters=hp.Int('conv1_filters',min_value=16, max_value=48, step=8), 
                                kernel_size=hp.Choice('conv1_kernels', values=[3,5,7], default=nrows),
                                activation='relu', padding='same'))
    cnn_model.add(layers.MaxPooling2D((3, 3), padding='same'))
    
    # Layer 2
    nrows = 3
    cnn_model.add(layers.Conv2D(filters=hp.Int('conv2_filters',min_value=16, max_value=48, step=8), 
                                kernel_size=hp.Choice('conv2_kernels', values=[3,5,7], default=nrows),
                                activation='relu', padding='same'))
    cnn_model.add(layers.MaxPooling2D((2, 2), padding='same'))
    
    # # Layer 3
    # nrows = 3
    # cnn_model.add(layers.Conv2D(filters=hp.Int('conv3_filters',min_value=16, max_value=48, step=8), 
    #                             kernel_size=hp.Choice('conv3_kernels', values=[3,5,7], default=nrows),
    #                             activation='relu', padding='same'))
    # cnn_model.add(layers.MaxPooling2D((1, 2), padding='same'))
    
    # --------- Decoder model --------- #
    # # Layer 4
    # nrows = 3
    # cnn_model.add(layers.Conv2D(filters=hp.Int('conv4_filters',min_value=16, max_value=48, step=8), 
    #                             kernel_size=hp.Choice('conv4_kernels', values=[3,5,7], default=nrows),
    #                             activation='relu', padding='same'))
    # cnn_model.add(layers.UpSampling2D((1, 2)))
    
    
    # Layer 5
    nrows = 3
    cnn_model.add(layers.Conv2D(filters=hp.Int('conv5_filters',min_value=16, max_value=48, step=8), 
                                kernel_size=hp.Choice('conv5_kernels', values=[3,5,7], default=nrows),
                                activation='relu', padding='same'))
    cnn_model.add(layers.UpSampling2D((2, 2)))
    
    # Layer 6
    nrows = 3
    cnn_model.add(layers.Conv2D(filters=hp.Int('conv6_filters',min_value=16, max_value=48, step=8), 
                                kernel_size=hp.Choice('conv6_kernels', values=[3,5,7], default=nrows),
                                activation='relu', padding='same'))
    cnn_model.add(layers.UpSampling2D((3, 3)))
    
    # Output layer
    nrows = 3
    cnn_model.add(layers.Conv2D(filters=hp.Int('conv7_filters',min_value=1, max_value=1), 
                                kernel_size=hp.Choice('conv7_kernels', values=[3,5,7], default=nrows),
                                activation='sigmoid', padding='same'))

    # ----------- Compile model ------------ #
    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True), 
        loss='binary_crossentropy',
        metrics=['mse'])
    
    return cnn_model

h = 30; w = 60;
tuner = Hyperband(build_cnn_model,
                  objective = 'val_mean_squared_error', 
                  max_epochs = 100,
                  factor = 3,
                  directory = 'cnn_hyperopt-hb-case7',
                  project_name = 'hyper_opt-case7')

# tuner = BayesianOptimization(build_cnn_model,
#                              objective = 'val_loss', 
#                              max_trials = 40,
#                              num_initial_points = 50,
#                              seed = 3,
#                              directory = 'cnn_hyperopt',
#                              project_name = 'hyper_opt-Run2')


# =============================================================================
# Loading training data
# =============================================================================

train_per = 0.95
x_train = np.array([])
x_test = np.array([])
y_train = np.array([])
y_test = np.array([])
datasets = ['Wide_free_sp','Wide_cong_sp','Wide_cong_more_sp']

for df in datasets:
    for i in [1,2,3,4,5,6]:
        if (i>3) and (df!=datasets[0]):
            continue
        input_X = np.load('../data/{}/inp_data_{}{}.npy'.format(df,df,i))
        noempty_indx = np.load('../data/{}/noempty_indx_{}{}.npy'.format(df,df,i))
        output = np.load('../data/{}/out_data_{}.npy'.format(df,df))
        output_Y = output[noempty_indx.reshape(-1),:,:,:]
        train_nums = int(output_Y.shape[0]*train_per)
        
        if len(x_train) == 0:
            x_train = input_X[:train_nums, :, :, :3]
            x_test = input_X[train_nums:, :, :, :3]
            y_train = output_Y[:train_nums, :, :, :]
            y_test = output_Y[train_nums:, :, :, :]
        else:
            x_train = np.append(x_train, input_X[:train_nums, :, :, :3], axis=0)
            x_test = np.append(x_test, input_X[train_nums:, :, :, :3], axis=0)
            y_train = np.append(y_train, output_Y[:train_nums, :, :, :], axis=0)
            y_test = np.append(y_test, output_Y[train_nums:, :, :, :], axis=0)


# sample data
sam_ratio = 20
num_samples = x_train.shape[0]
num_rand_samples = int(num_samples*sam_ratio/100)
sam_idx = np.arange(0, num_samples, dtype=np.int)
rand_sam_idx = np.random.choice(sam_idx, num_rand_samples, replace=False)
x_train = x_train[rand_sam_idx,:,:,:]
y_train = y_train[rand_sam_idx,:,:,:]

# crop space-time plane
x_train = x_train[:,:h,:w,:]
y_train = y_train[:,:h,:w,:]
x_test = x_test[:,:h,:w,:]
y_test = y_test[:,:h,:w,:]


# =============================================================================
# Tune the model
# =============================================================================

# Hyper parameter search
tuner.search(x_train, y_train, epochs = 75, batch_size=512,
             validation_data = (x_test, y_test), 
             callbacks = [ClearTrainingOutput()])
tuner.results_summary()

# best params and model
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""The hyperparameter search is complete. The optimal number of units are: \\
      Layer 1: filters={best_hps.get('conv1_filters')}, kernels={best_hps.get('conv1_kernels')}, \\
      Layer 2: filters={best_hps.get('conv2_filters')}, kernels={best_hps.get('conv2_kernels')}, \\
      Layer 5: filters={best_hps.get('conv5_filters')}, kernels={best_hps.get('conv5_kernels')}, \\
      Layer 6: filters={best_hps.get('conv6_filters')}, kernels={best_hps.get('conv6_kernels')}, \\
      Layer 7: filters={best_hps.get('conv7_filters')}, kernels={best_hps.get('conv7_kernels')} """)
best_model.save('./Expt30_bestmodel_5p-7.h5')

# # Training the best model
# loss = []; val_loss = []; mse = []; val_mse = []; loss_fn='bce'
# history = best_model.fit(x_train, y_train, 
#                          epochs=300, batch_size=512, shuffle=True, 
#                          validation_data=(x_test, y_test), verbose=0)

# loss_fn = 'bce'
# best_model.save('./Expt30_isomodel_5p-2.h5'.format(loss_fn))
# try:
#     val_loss = val_loss+history.history['val_loss']
#     np.save('./Expt21_val_loss-{}.npy'.format(loss_fn), np.array(val_loss))
# except:
#     pass
# try:
#     val_mse = val_mse+history.history['val_mse']
#     np.save('./Expt21_val_mse-{}.npy'.format(loss_fn), np.array(val_mse))
# except:
#     pass
    

