# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:41:26 2020

@author: btt1

Training of Deep CNN model (based on Summer 2019 work)

"""

# =============================================================================
# Import Libraries
# =============================================================================

import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
# from sklearn.model_selection import train_test_split
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

# ---------- Input layer ---------- #

h = 80; w = 60; inp_dep = 3
input_img = Input(shape=(h, w, inp_dep))

# --------- Encoder model ----------- #

# Layer 1
nrows = 5; ncols = 5; inp_dep = 3; out_dep = 8;
mask1 = np.random.randint(1, 2, [nrows,ncols,inp_dep,out_dep], np.int32)
# mask1 = create_mask(16.67,2,-4.67,3,nrows,ncols,inp_dep,out_dep)
cnn_1 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
               kernel_constraint=Tilted_kernel(mask1))(input_img)
mpl_1 = MaxPooling2D((2, 3), padding='same')(cnn_1)

# Layer 2
nrows = 5; ncols = 5; inp_dep = 8; out_dep = 32;
mask2 = np.random.randint(1, 2, [nrows,ncols,inp_dep,out_dep], np.int32)
# mask2 = create_mask(16.67,2,-4.67,3,nrows,ncols,inp_dep,out_dep)
cnn_2 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
               kernel_constraint=Tilted_kernel(mask2))(mpl_1)
mpl_2 = MaxPooling2D((2, 2), padding='same')(cnn_2)

# Layer 3
nrows = 3; ncols = 3; inp_dep = 32; out_dep = 64;
mask3 = np.random.randint(1, 2, [nrows,ncols,inp_dep,out_dep], np.int32)
# mask3 = create_mask(16.67,2,-4.67,3,nrows,ncols,inp_dep,out_dep)
cnn_3 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
               kernel_constraint=Tilted_kernel(mask3))(mpl_2)
encoder = MaxPooling2D((2, 2), padding='same')(cnn_3)


# --------- Decoder model --------- #

# Layer 4
nrows = 3; ncols = 3; inp_dep = 64; out_dep = 64;
mask4 = np.random.randint(1, 2, [nrows,ncols,inp_dep,out_dep], np.int32)
# mask4 = create_mask(16.67,2,-4.67,3,nrows,ncols,inp_dep,out_dep)
cnn_4 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
                kernel_constraint=Tilted_kernel(mask4))(encoder)
usl_4 = UpSampling2D((2, 2))(cnn_4)


# Layer 5
nrows = 5; ncols = 5; inp_dep = 64; out_dep = 32;
mask5 = np.random.randint(1, 2, [nrows,ncols,inp_dep,out_dep], np.int32)
# mask5 = create_mask(16.67,2,-4.67,3,nrows,ncols,inp_dep,out_dep)
cnn_5 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
                kernel_constraint=Tilted_kernel(mask5))(usl_4)
usl_5 = UpSampling2D((2, 2))(cnn_5)

# Layer 6
nrows = 5; ncols = 5; inp_dep = 32; out_dep = 8;
mask6 = np.random.randint(1, 2, [nrows,ncols,inp_dep,out_dep], np.int32)
# mask6 = create_mask(16.67,2,-4.67,3,nrows,ncols,inp_dep,out_dep)
cnn_6 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same',
                kernel_constraint=Tilted_kernel(mask6))(usl_5)
usl_6 = UpSampling2D((2, 3))(cnn_6)

# Output layer
nrows = 5; ncols = 5; inp_dep = 8; out_dep = 1;
mask7 = np.random.randint(1, 2, [nrows,ncols,inp_dep,out_dep], np.int32)
# mask7 = create_mask(16.67,2,-4.67,3,nrows,ncols,inp_dep,out_dep)
decoder = Conv2D(out_dep, (nrows, ncols), activation='sigmoid', padding='same',
                 kernel_constraint=Tilted_kernel(mask7))(usl_6)


# ----------- Reconstruction model ------------ #

autoencoder = Model(input_img, decoder)
optmzr = keras.optimizers.Adam(lr=0.00001, amsgrad=True)
autoencoder.compile(optimizer=optmzr, loss='binary_crossentropy', metrics=['mse','mae'])
autoencoder.summary()

# =============================================================================
# Loading training data
# =============================================================================

# Load input and output data
# output_Y = np.load('./out_data.npy')[:,:80,:,:]
# input_X = np.load('./inp_data.npy')[:,:80,:,:]

# Split data into training and testing sets
#x_train,x_test,y_train,y_test = train_test_split(input_X,
#                                                 output_Y, 
#                                                 test_size=0.1, 
#                                                 random_state=13)
#x_train,x_test,y_train,y_test = 

train_per = 0.95
x_train = np.array([])
x_test = np.array([])
y_train = np.array([])
y_test = np.array([])
datasets = ['Wide_cong','Wide_free','Wide_cong_more']

for df in datasets:
    
    for i in [1,2,3,4,5,6]:
        
        if (i>3) and (df!='Wide_free'):
            continue
        
        output_Y = np.load('../data/{}/out_data_{}{}.npy'.format(df,df,i))
        input_X = np.load('../data/{}/inp_data_{}{}.npy'.format(df,df,i))
        train_nums = int(output_Y.shape[0]*train_per)
        
        if len(x_train) == 0:
            x_train = input_X[:train_nums, :, :, :]
            x_test = input_X[train_nums:, :, :, :]
            y_train = output_Y[:train_nums, :, :, :]
            y_test = output_Y[train_nums:, :, :, :]
        else:
            x_train = np.append(x_train, input_X[:train_nums, :, :, :], axis=0)
            x_test = np.append(x_test, input_X[train_nums:, :, :, :], axis=0)
            y_train = np.append(y_train, output_Y[:train_nums, :, :, :], axis=0)
            y_test = np.append(y_test, output_Y[train_nums:, :, :, :], axis=0)

# =============================================================================
# Train the model
# =============================================================================

# Learning step by step
loss = []; val_loss = []; mse = []; val_mse = []; loss_fn='bce'
history = autoencoder.fit(x_train, y_train, epochs=500, batch_size=32, shuffle=True, 
                validation_data=(x_test, y_test), verbose=0)
loss = loss+history.history['loss']
mse = mse+history.history['mse']
val_loss = val_loss+history.history['val_loss']
val_mse = val_mse+history.history['val_mse']

autoencoder.save('./Expt4_model_nomask-{}.h5'.format(loss_fn))
np.save('./Expt4_loss-{}.npy'.format(loss_fn), np.array(loss))
np.save('./Expt4_val_loss-{}.npy'.format(loss_fn), np.array(val_loss))
np.save('./Expt4_mse-{}.npy'.format(loss_fn), np.array(mse))
np.save('./Expt4_val_mse-{}.npy'.format(loss_fn), np.array(val_mse))
