# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:41:26 2020

@author: btt1

Training of Deep CNN model (based on Summer 2019 work)

Improved CNN: Training Kernel desgin 2

"""

# =============================================================================
# Import Libraries
# =============================================================================

import sys
import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

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

def mask_def1(slope, width, ker_size=7):
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

def mask_def2(m1, m2, s=7):
    mask = np.zeros((s,s))
    ts = np.arange(s) - (s-1)/2 
    xs = ts * -10
    i = 0
    j = 0
    for x in xs:
        j = 0
        for t in ts:
            if (m1*t<=x and x<=m2*t) or (m1*t>=x and x>=m2*t):
                mask[i,j] = 1
            j = j + 1
        i = i + 1
    return mask

def create_mask(s1,w1,s2,w2,k1,k2,inp_dep,out_dep,mask_def):
    
    ff_mask = mask_def(s1, w1, k1)
    cg_mask = mask_def1(s2, w2, k2)
    if k1==7:
        ff_mask[2,3]=1; ff_mask[4,3]=1;
    elif k1==5:
        ff_mask[1,2]=1; ff_mask[3,2]=1;
    elif k1==9:
        ff_mask[3,4]=1; ff_mask[5,4]=1;
    
    com_mask = np.zeros((max(k1, k2), max(k1, k2)))
    com_mask = np.maximum(com_mask, ff_mask)
    com_mask = np.maximum(com_mask, cg_mask)
    
    ker_mask = np.zeros((max(k1,k2),max(k1,k2),inp_dep,out_dep), np.int32)
    for i in range(ker_mask.shape[2]):
        for j in range(ker_mask.shape[3]):
            ker_mask[:,:,i,j] = com_mask
    
    return ker_mask

def metric_var(y_true, y_pred):
    
    return keras.backend.var(y_pred - y_true)


# =============================================================================
# CNN Reconstruction model (Improved version)
# =============================================================================

# ----------- Input layer ----------- #

h = 80; w = 60; inp_dep = 3
input_img = Input(shape=(h, w, inp_dep))
max_v = 30; max_w = -5.3; min_v = 9; w_w = 1.2

# --------- Encoder model ----------- #

# Layer 1
nrows = 5; ncols = 5; inp_dep = 3; out_dep = 40;
mask1 = create_mask(max_v,min_v,max_w,w_w,nrows,ncols,inp_dep,out_dep,mask_def2)
cnn_1 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
               kernel_constraint=Tilted_kernel(mask1))(input_img)
mpl_1 = MaxPooling2D((2, 3), padding='same')(cnn_1)

# Layer 2
nrows = 7; ncols = 7; inp_dep = 40; out_dep = 48;
mask2 = create_mask(max_v,min_v,max_w,w_w,nrows,ncols,inp_dep,out_dep,mask_def2)
cnn_2 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
               kernel_constraint=Tilted_kernel(mask2))(mpl_1)
mpl_2 = MaxPooling2D((2, 2), padding='same')(cnn_2)

# Layer 3
nrows = 7; ncols = 7; inp_dep = 48; out_dep = 32;
mask3 = create_mask(max_v,min_v,max_w,w_w,nrows,ncols,inp_dep,out_dep,mask_def2)
cnn_3 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
               kernel_constraint=Tilted_kernel(mask3))(mpl_2)
encoder = MaxPooling2D((2, 2), padding='same')(cnn_3)


# --------- Decoder model --------- #

# Layer 4
nrows = 5; ncols = 5; inp_dep = 32; out_dep = 48;
mask4 = create_mask(max_v,min_v,max_w,w_w,nrows,ncols,inp_dep,out_dep,mask_def2)
cnn_4 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
               kernel_constraint=Tilted_kernel(mask4))(encoder)
usl_4 = UpSampling2D((2, 2))(cnn_4)

# Layer 5
nrows = 5; ncols = 5; inp_dep = 48; out_dep = 40;
mask5 = create_mask(max_v,min_v,max_w,w_w,nrows,ncols,inp_dep,out_dep,mask_def2)
cnn_5 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
               kernel_constraint=Tilted_kernel(mask5))(usl_4)
usl_5 = UpSampling2D((2, 2))(cnn_5)

# Layer 6
nrows = 9; ncols = 9; inp_dep = 40; out_dep = 56;
mask6 = create_mask(max_v,min_v,max_w,w_w,nrows,ncols,inp_dep,out_dep,mask_def2)
cnn_6 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
               kernel_constraint=Tilted_kernel(mask6))(usl_5)
usl_6 = UpSampling2D((2, 3))(cnn_6)

# Output layer
nrows = 7; ncols = 7; inp_dep = 56; out_dep = 1;
mask7 = create_mask(max_v,min_v,max_w,w_w,nrows,ncols,inp_dep,out_dep,mask_def2)
decoder = Conv2D(out_dep, (nrows, ncols), activation='sigmoid', padding='same',
                 kernel_constraint=Tilted_kernel(mask7))(usl_6)


# ----------- Reconstruction model ------------ #

tse_cnn_model = Model(input_img, decoder, name='TSE_CNN_AnisotropicModel')
optmzr = keras.optimizers.Adam(lr=0.001, amsgrad=True)
tse_cnn_model.compile(optimizer=optmzr, loss='binary_crossentropy', 
                    metrics=[keras.metrics.RootMeanSquaredError(),
                             metric_var])
tse_cnn_model.summary()

# =============================================================================
# Loading training data
# =============================================================================

# params
itr = int(sys.argv[1])
train_per = 0.95
x_train = np.array([])
x_test = np.array([])
y_train = np.array([])
y_test = np.array([])
datasets = ['Wide_cong_sp','Wide_free_sp','Wide_cong_more_sp']

# load dataset
for df in datasets:
    for i in [1,2,3,4,5,6]:
        if (i>3) and (df!='Wide_free_sp'):
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


# sample subset of dataset
if itr == 1:
    sam_ratio = 100
elif itr == 2:
    sam_ratio = 80
elif itr == 3:
    sam_ratio = 60
elif itr == 4:
    sam_ratio = 40
elif itr == 5:
    sam_ratio = 20
elif itr == 6:
    sam_ratio = 10
elif itr == 7:
    sam_ratio = 5
elif itr == 8:
    sam_ratio = 2.5
elif itr == 9:
    sam_ratio = 125
elif itr == 10:
    sam_ratio = 150

num_samples = x_train.shape[0]
num_rand_samples = int(num_samples*sam_ratio/100)
if (itr == 9) or (itr==10):
    num_add_samples = num_rand_samples-num_samples
    sam_idx = np.arange(0, num_samples, dtype=np.int)
    rand_sam_idx = np.random.choice(sam_idx, num_add_samples, replace=False)
    x_train_add = x_train[rand_sam_idx,:,:,:]
    y_train_add = y_train[rand_sam_idx,:,:,:]
    x_train = np.append(x_train, x_train_add, axis=0)
    y_train = np.append(y_train, y_train_add, axis=0)
elif itr != 1:
    sam_idx = np.arange(0, num_samples, dtype=np.int)
    rand_sam_idx = np.random.choice(sam_idx, num_rand_samples, replace=False)
    x_train = x_train[rand_sam_idx,:,:,:]
    y_train = y_train[rand_sam_idx,:,:,:]
    

# =============================================================================
# Train the model
# =============================================================================

# Model fit
history = tse_cnn_model.fit(x_train, y_train, 
                            epochs=300, batch_size=128, shuffle=True, 
                            validation_data=(x_test, y_test), verbose=0)

# Save model
tse_cnn_model.save('./Expt28_anisomodel_5p-{}.h5'.format(itr))

# Saving training results
loss = history.history['loss']
rmse = history.history['root_mean_squared_error']
var = history.history['metric_var']
val_loss = history.history['val_loss']
val_rmse = history.history['val_root_mean_squared_error']
val_var = history.history['val_metric_var']

np.save('./Expt28_anisomodel_loss-{}.npy'.format(itr), np.array(loss))
np.save('./Expt28_anisomodel_val_loss-{}.npy'.format(itr), np.array(val_loss))
np.save('./Expt28_anisomodel_rmse-{}.npy'.format(itr), np.array(rmse))
np.save('./Expt28_anisomodel_val_rmse-{}.npy'.format(itr), np.array(val_rmse))
# np.save('./Expt28_anisomodel_var-{}.npy'.format(itr), np.array(var))
# np.save('./Expt28_anisomodel_val_var-{}.npy'.format(itr), np.array(val_var))

# Save evaluation results
predy_train = tse_cnn_model.predict(x_train)
predy_test = tse_cnn_model.predict(x_test)
rmse_train = np.sqrt(np.mean(np.power(predy_train-y_train, 2), axis=(1,2,3)))
rmse_test = np.sqrt(np.mean(np.power(predy_test-y_test, 2), axis=(1,2,3)))
np.save('./Expt28_anisomodel_trainevals-{}.npy'.format(itr), np.array(rmse_train))
np.save('./Expt28_anisomodel_testevals-{}.npy'.format(itr), np.array(rmse_test))
