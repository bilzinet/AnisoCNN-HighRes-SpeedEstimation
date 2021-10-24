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
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import load_model, Model
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
np.set_printoptions(precision=4, suppress=True)

from matplotlib import rcParams
plt.rcParams["font.family"] = "Times New Roman"
rcParams['mathtext.fontset']='cm'

# =============================================================================
# Model class and functions
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

def CNN_model(h=80, w=60):
    
    inp_dep = 3
    input_img = Input(shape=(h, w, inp_dep))
    # --------- Encoder model ----------- #
    # Layer 1
    nrows = 5; ncols = 5; inp_dep = 3; out_dep = 40;
    cnn_1 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(input_img)
    mpl_1 = MaxPooling2D((2, 3), padding='same')(cnn_1)
    # Layer 2
    nrows = 7; ncols = 7; inp_dep = 40; out_dep = 48;
    cnn_2 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(mpl_1)
    mpl_2 = MaxPooling2D((2, 2), padding='same')(cnn_2)
    # Layer 3
    nrows = 7; ncols = 7; inp_dep = 48; out_dep = 32;
    cnn_3 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(mpl_2)
    encoder = MaxPooling2D((2, 2), padding='same')(cnn_3)
    # --------- Decoder model --------- #
    # Layer 4
    nrows = 5; ncols = 5; inp_dep = 32; out_dep = 48;
    cnn_4 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(encoder)
    usl_4 = UpSampling2D((2, 2))(cnn_4)
    # Layer 5
    nrows = 5; ncols = 5; inp_dep = 48; out_dep = 40;
    cnn_5 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(usl_4)
    usl_5 = UpSampling2D((2, 2))(cnn_5)
    # Layer 6
    nrows = 9; ncols = 9; inp_dep = 40; out_dep = 56;
    cnn_6 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(usl_5)
    usl_6 = UpSampling2D((2, 3))(cnn_6)
    # Output layer
    nrows = 7; ncols = 7; inp_dep = 56; out_dep = 1;
    decoder = Conv2D(out_dep, (nrows, ncols), activation='sigmoid', padding='same')(usl_6)

    return Model(input_img, decoder)

# =============================================================================
# Load and create models: NGSIM and HIGHD
# =============================================================================

# Load the base trained model
base_model = load_model('./3_Training experiments/expt-24 (iso-ani-compare)/Expt24_anisomodel_5p-1.h5',
                        custom_objects={'Tilted_kernel':Tilted_kernel}, 
                        compile=False)

# NGSIM model (US-101)
# Create a dummy model and transfer weights
ngsim_model = CNN_model(h=80, w=2760)
ngsim_model.set_weights(base_model.get_weights())

# HighD model (HW-)
# Create a dummy model and transfer weights
highd_model = CNN_model(h=80, w=1140)  # w=2760
highd_model.set_weights(base_model.get_weights())

# =============================================================================
# NGSIM: Load, predict, RMSE, visualize
# =============================================================================

# Parameters
c=3; h=80; w=60
h_act=67; w_act=2760
max_speed = 95
x_test = np.array([])
y_test = np.array([])
datasets = ['Ngsim']

# Load data
for df in datasets:
    for i in ['us101_lane2']: 
        output_Y = np.load('./5_Testing data/{}/out_data_{}_{}.npy'.format(df,df,i))
        input_X = np.load('./5_Testing data/{}/inp_data_{}_{}.npy'.format(df,df,i))
        if len(x_test) == 0:
            x_test = input_X[:, :, :w_act, :]
            y_test = output_Y[:, :, :w_act, :]
        else:
            x_test = np.append(x_test, input_X[:, :, :w_act, :], axis=0)
            y_test = np.append(y_test, output_Y[:, :, :w_act, :], axis=0)
x_bin = (x_test.sum(axis=3) != 0)
y_map = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
x_map = x_bin*y_map
data_ngsim = [x_test, y_test, x_map]

# Predict (crop out the zero-padded region)
w_mod = 2400
ngsim_pred = ngsim_model.predict(data_ngsim[0])[:,:h_act,:w_mod,:]

# RMSE error
ngsim_rmse = []
true_array = data_ngsim[1][0]
for s in range(ngsim_pred.shape[0]):
    true_y = true_array[:h_act,:w_mod,:].reshape(-1)
    pred_y1 = ngsim_pred[s,:h_act,:w_mod,:].reshape(-1)
    rmse1 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y1)*max_speed, 2))), 3)
    ngsim_rmse.append(rmse1)
print('Average RMSE for prediction: ', np.mean(ngsim_rmse))
print('Standard deviation is: ', np.std(ngsim_rmse))

# Visualize estimation results
ngsim_avg_pred = ngsim_pred.mean(axis=0)
ngsim_avg_pred = ngsim_pred[np.argmin(ngsim_rmse),:,:,:]  # 43
ngsim_input = data_ngsim[0][np.argmin(ngsim_rmse),:h_act,:w_mod,:]
true_speed = data_ngsim[1][0][:h_act,:w_mod,:]
input_probe = data_ngsim[-1][np.argmin(ngsim_rmse),:h_act,:w_mod]


fig_a = plt.figure(figsize=(5, 6))
gs1 = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.5)

# display true speed field (true output)
ax = fig_a.add_subplot(gs1[0, 0])
map1 = plt.imshow(true_speed.reshape(h_act, w_mod)*max_speed, 
                  cmap='jet_r', vmin=0, vmax=max_speed, aspect='auto')
x_space = np.nonzero(input_probe)[0]
y_time = np.nonzero(input_probe)[1]
plt.scatter(y_time, x_space, s=1, c='k', marker='.', alpha=0.8)
ax.set_title('(a) True speed map', y=-0.41, fontsize=15)
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('Space (m)', fontsize=14)
ax.set_yticks([66, 46, 26, 6])
ytick_labels = ['0','200','400','600','800','1000']
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=13)
plt.setp(ax.get_yticklabels(), fontsize=13)
cbar_ax = fig_a.add_axes([0.93, 0.60, 0.02, 0.25])
cbar = fig_a.colorbar(map1, cax=cbar_ax)
cbar.ax.tick_params(labelsize=11)
cbar.ax.set_ylabel('$V$ (kmph)')

ax = fig_a.add_subplot(gs1[1, 0])
plt.imshow(ngsim_avg_pred.reshape(h_act, w_mod)*max_speed, 
           cmap='jet_r', vmin=0, vmax=max_speed, aspect='auto')
ax.set_title('(b) Estimated speed map', y=-0.41, fontsize=15)
ax.set_xlabel('Time (s)',fontsize=14)
ax.set_ylabel('Space (m)',fontsize=14)
ax.set_yticks([66, 46, 26, 6])
ytick_labels = ['0','200','400','600','800','1000']
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=13)
plt.setp(ax.get_yticklabels(), fontsize=13)

# plt.tight_layout()
plt.savefig('Ngsim_pred.pdf', bbox_inches='tight')


# True speed field
fig_a = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.1)
ax = fig_a.add_subplot(gs1[0, 0])
map1 = plt.imshow(data_ngsim[1][0,:67,120:-120,0]*95, 
                  cmap='jet_r', vmin=0, vmax=95, aspect='auto')
ax.set_title('NGSIM US-101 Lane 2', fontsize=17)
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel(r'Road length (m)', fontsize=16)
ax.set_yticks([66, 46, 26, 6])
ytick_labels = ['0','200','400','600','800','1000']
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=15)
plt.setp(ax.get_yticklabels(), fontsize=15)
cbar_ax = fig_a.add_axes([0.93, 0.50, 0.02, 0.35])
cbar = fig_a.colorbar(map1, cax=cbar_ax)
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_ylabel('Speed (kmph)', fontsize=16)
plt.savefig('US101_lane2.png', bbox_inches='tight')

true_sf = data_ngsim[1][0,:67,120:-120,0]*95
np.save('./US101_lane2.npy', true_sf)

# True speed field (Zoomed: 600 to 800 secs)
fig_a = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.1)
ax = fig_a.add_subplot(gs1[0, 0])
map1 = plt.imshow(data_ngsim[1][0,:67,600:900,0]*95, 
                  cmap='jet_r', vmin=0, vmax=70, aspect='auto')
ax.set_title('NGSIM US-101 Lane 2', fontsize=17)
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel(r'Road length (m)', fontsize=16)
ax.set_yticks([66, 46, 26, 6])
ytick_labels = ['0','200','400','600','800','1000']
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=15)
plt.setp(ax.get_yticklabels(), fontsize=15)
cbar_ax = fig_a.add_axes([0.93, 0.50, 0.02, 0.35])
cbar = fig_a.colorbar(map1, cax=cbar_ax)
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_ylabel('Speed (kmph)', fontsize=16)


plt.savefig('US101_lane2.png', bbox_inches='tight')

true_sf = data_ngsim[1][0,:67,120:-120,0]*95
np.save('./US101_lane2.npy', true_sf)

fig_a = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.1)
ax = fig_a.add_subplot(gs1[0, 0])
map1 = plt.imshow(ngsim_pred[43][:67,600:900,0]*95, 
                  cmap='jet_r', vmin=0, vmax=70, aspect='auto')
ax.set_title('NGSIM US-101 Lane 2', fontsize=17)
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel(r'Road length (m)', fontsize=16)
ax.set_yticks([66, 46, 26, 6])
ytick_labels = ['0','200','400','600','800','1000']
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=15)
plt.setp(ax.get_yticklabels(), fontsize=15)
cbar_ax = fig_a.add_axes([0.93, 0.50, 0.02, 0.35])
cbar = fig_a.colorbar(map1, cax=cbar_ax)
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_ylabel('Speed (kmph)', fontsize=16)


# =============================================================================
# HighD dataset: Load, predict, RMSE, visualize
# =============================================================================

# Parameters
c=3; h=80; w=60
h_act=40; w_act=1140
max_speed = 75  # 75 for hw25 lane 4; 165 for hw44 lane 6
x_test = np.array([])
y_test = np.array([])
datasets = ['HighD']

# Load data
for df in datasets:
    for i in ['track25_lane4-5p']:  # track44_lane6 and track25_lane4
        output_Y = np.load('./5_Testing data/{}/out_data_{}_{}.npy'.format(df,df,i))
        input_X = np.load('./5_Testing data/{}/inp_data_{}_{}.npy'.format(df,df,i))
        if len(x_test) == 0:
            x_test = input_X[:, :, :w_act, :]
            y_test = output_Y[:, :, :w_act, :]
        else:
            x_test = np.append(x_test, input_X[:, :, :w_act, :], axis=0)
            y_test = np.append(y_test, output_Y[:, :, :w_act, :], axis=0)
x_bin = (x_test.sum(axis=3) != 0)
y_map = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
x_map = x_bin*y_map
data_highd = [x_test, y_test, x_map]

# Predict (crop out the zero-padded region)
w_mod = 1140
highd_pred = highd_model.predict(data_highd[0])[:,:h_act,:w_mod,:]

# RMSE error
highd_rmse = []
true_array = data_highd[1][0]
for s in range(highd_pred.shape[0]):
    true_y = true_array[:h_act,:w_mod,:].reshape(-1)
    pred_y1 = highd_pred[s,:h_act,:w_mod,:].reshape(-1)
    rmse1 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y1)*max_speed, 2))), 3)
    highd_rmse.append(rmse1)
print('Average RMSE for prediction: ', np.mean(highd_rmse))
print('Standard deviation is: ', np.std(highd_rmse))

# Visualize estimation results
highd_avg_pred = highd_pred.mean(axis=0)
highd_avg_pred = highd_pred[10,:,:,:]  # 3 for hw-44 and 10 for hw-25
highd_input = data_highd[0][10,:h_act,:w_mod,:]
true_speed = data_highd[1][0][:h_act,:w_mod,:]
input_probe = data_highd[-1][10,:h_act,:w_mod]

fig_a = plt.figure(figsize=(5, 6))
gs1 = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.5)

# display true speed field (true output)
ax = fig_a.add_subplot(gs1[0, 0])
map1 = ax.imshow(true_speed.reshape(h_act, w_mod)*max_speed, 
                 cmap='jet_r', vmin=0, vmax=max_speed, aspect='auto')
x_space = np.nonzero(input_probe)[0]
y_time = np.nonzero(input_probe)[1]
plt.scatter(y_time, x_space, s=1, c='k', marker='o', alpha=0.8)
ax.set_title('(a) True speed map', y=-0.41, fontsize=15)
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('Space (m)', fontsize=14)
ytick_labels = ['500','400','300','200','100','0']
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=13)
plt.setp(ax.get_yticklabels(), fontsize=13)
cbar_ax = fig_a.add_axes([0.93, 0.60, 0.02, 0.25])
cbar = fig_a.colorbar(map1, cax=cbar_ax)
cbar.ax.tick_params(labelsize=11)
cbar.ax.set_ylabel('$V$ (kmph)')

# display reconstructed speed field
ax = fig_a.add_subplot(gs1[1, 0])
plt.imshow(highd_avg_pred.reshape(h_act, w_mod), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
ax.set_title('(b) Estimated speed map', y=-0.41, fontsize=15)
ytick_labels = ['500','400','300','200','100','0']
ax.set_yticklabels(ytick_labels)
ax.set_xlabel('Time (s)',fontsize=14)
ax.set_ylabel('Space (m)',fontsize=14)
plt.setp(ax.get_xticklabels(), fontsize=13)
plt.setp(ax.get_yticklabels(), fontsize=13) 

# plt.tight_layout()
plt.savefig('Highd_pred-hw25.pdf', bbox_inches='tight')


# True speed field
fig_a = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.1)
ax = fig_a.add_subplot(gs1[0, 0])
map1 = plt.imshow(data_highd[1][0,:40,:,0]*75, 
                  cmap='jet_r', vmin=0, vmax=75, aspect='auto')
ax.set_title('HighD HW 25 Lane 4', fontsize=17)
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel(r'Road length (m)', fontsize=16)
ax.set_yticks([40, 30, 20, 10, 0])
ytick_labels = ['0','100','200','300','400']
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=15)
plt.setp(ax.get_yticklabels(), fontsize=15)
cbar_ax = fig_a.add_axes([0.93, 0.50, 0.02, 0.35])
cbar = fig_a.colorbar(map1, cax=cbar_ax)
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_ylabel('Speed (kmph)', fontsize=16)
plt.savefig('HW25_lane4.png', bbox_inches='tight')

true_sf = data_highd[1][0,:40,:,0]*75
np.save('./HW25_lane4.npy', true_sf)


# =============================================================================
# NGSIM: Comparison with CNN, GASM and EnKF 
# =============================================================================

def rmse_calc(y_pred, y_act):
    rmse_sam = np.sqrt(np.nanmean(np.power(y_pred - y_act,2), axis=(1,2,3)))
    return rmse_sam

# ------------- CNN Prediction ------------ #

# NGSIM model (US-101)
# Create a dummy model and transfer weights
ngsim_model = CNN_model(h=80, w=300)
ngsim_model.set_weights(base_model.get_weights())


# Parameters
c=3; h=80; w=60; h_act=67; w_sta=600; w_act=900; max_speed = 95


# output data
df = 'Ngsim'
out_i = 'us101_lane2'
output_Y = np.load('./5_Testing data/{}/out_data_{}_{}.npy'.format(df,df,out_i))
y_test = output_Y[:, :, w_sta:w_act, :]


# input data
data_ngsim = []
for i in ['us101_lane2_3p','us101_lane2_5p','us101_lane2_10p']: # 'us101_lane2_1p','us101_lane2_3p','us101_lane2_5p','us101_lane2_10p','us101_lane2_20p'
    input_X = np.load('./5_Testing data/{}/inp_data_{}_{}.npy'.format(df,df,i))
    x_test = input_X[:, :, w_sta:w_act, :]
    data_ngsim.append(x_test)
data_ngsim.append(y_test)


# Predict (crop out the zero-padded region)
aniso_preds = []
for i in range(len(data_ngsim)-1):
    pred_i = ngsim_model.predict(data_ngsim[i])[:,:h_act,:,:]
    aniso_preds.append(pred_i)

    
# error calculations
sams = 25
aniso_rmse = []
print('\nAnisotropic Deep CNN prediction results...')
for i in range(len(data_ngsim)-1):
    print('\n\tData: ', i)
    r = rmse_calc(aniso_preds[i]*max_speed, 
                  np.repeat(y_test, sams, axis=0)[:,:h_act,:,:]*max_speed)
    ir = np.argsort(r)
    m = r[ir][:].mean()
    s = r[ir][:].std()
    aniso_rmse.append((np.nanmean(r),np.nanstd(r)))
    print(f'\tMean rmse: {m:0.03f} +- {s:0.03f}')


# save results
np.save('./5_Testing data/Ngsim/AnisoCNN_preds.npy', np.array(aniso_preds))
np.save('./5_Testing data/Ngsim/AnisoCNN_preds_rmse.npy', np.array(aniso_rmse))


# plt.figure()
# plt.imshow(data_ngsim[1][0][:h_act,:,:], cmap='jet_r', aspect='auto')
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(ngsim_preds[1][0].squeeze(), cmap='jet_r', aspect='auto', vmin=0, vmax=1)
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(y_test[0,:h_act,:,0], cmap='jet_r', aspect='auto')
# plt.colorbar()
# plt.show()



# ------------- Aniso Prediction ----------- #

# load predictions
aniso_preds = np.load('./5_Testing data/Ngsim/AnisoCNN_preds.npy')
aniso_rmse = np.load('./5_Testing data/Ngsim/AnisoCNN_preds_rmse.npy')


# ------------- GASM Prediction ------------ #

# load predictions
gasm_preds = np.load('./5_Testing data/Ngsim/GASM_preds.npy')
gasm_rmse = np.load('./5_Testing data/Ngsim/GASM_preds_rmse.npy')
gasm_comp = np.load('./5_Testing data/Ngsim/GASM_pred0.npy')

# ------------- EnKF Prediction ------------ #

# load predictions
enkf_preds = np.load('./5_Testing data/Ngsim/EnKF_preds.npy')
enkf_rmse = np.load('./5_Testing data/Ngsim/EnKF_preds_rmse.npy')


# -------------- Visualize comparison ------------ #


x_inpt = data_ngsim[1][0][:h_act,:,:]
y_true = y_test[0,:h_act,:,0]*95
y_gasm = gasm_comp
y_anis = aniso_preds[1][0].squeeze()*100

plt.figure(figsize=(4,4))
plt.imshow(x_inpt, cmap='jet_r', aspect='auto')
plt.xlabel('Time (sec)', fontsize=16)
plt.ylabel('Space (m)', fontsize=16)
plt.title('(a) Input Trajectories', fontsize=18)
plt.xticks([0,100,200,300], fontsize=14)
plt.yticks([6,26,46,66],[60,40,20,0], fontsize=14)
plt.tight_layout()
plt.show()
plt.savefig('GASM_comp-1.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(4.2,4))
map1 = plt.imshow(y_true, cmap='jet_r', aspect='auto', vmin=0, vmax=70)
plt.xlabel('Time (sec)', fontsize=16)
plt.ylabel('Space (m)', fontsize=16)
plt.title('(a) True Speed Field', fontsize=18)
plt.xticks([0,100,200,300], fontsize=14)
plt.yticks([6,26,46,66],[60,40,20,0], fontsize=14)
plt.tight_layout()
cbar = plt.colorbar(map1)
cbar.set_label('Speed (km/hr)', rotation=90, fontsize=14)
plt.show()
plt.savefig('GASM_comp-2.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(4,4))
map2 = plt.imshow(y_gasm, cmap='jet_r', aspect='auto', vmin=0, vmax=70)
plt.xlabel('Time (sec)', fontsize=16)
plt.ylabel('Space (m)', fontsize=16)
plt.title('(a) GASM Estimation', fontsize=18)
plt.xticks([0,100,200,300], fontsize=14)
plt.yticks([6,26,46,66],[60,40,20,0], fontsize=14)
plt.tight_layout()
cbar = plt.colorbar(map2)
cbar.set_label('Speed (km/hr)', rotation=90, fontsize=14)
plt.show()
plt.savefig('GASM_comp-3.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(4.2,4))
map3 = plt.imshow(y_anis, cmap='jet_r', aspect='auto', vmin=0, vmax=70)
plt.xlabel('Time (sec)', fontsize=16)
plt.ylabel('Space (m)', fontsize=16)
plt.title('(a) Anisotropic CNN estimation', fontsize=18)
plt.xticks([0,100,200,300], fontsize=14)
plt.yticks([6,26,46,66],[60,40,20,0], fontsize=14)
plt.tight_layout()
cbar = plt.colorbar(map3)
cbar.set_label('Speed (km/hr)', rotation=90, fontsize=14)
plt.show()
plt.savefig('GASM_comp-4.png', dpi=300, bbox_inches='tight')

# =============================================================================
# For TRB Poster
# =============================================================================

# Visualize estimation results
ngsim_avg_pred = ngsim_pred.mean(axis=0)
ngsim_avg_pred = ngsim_pred[np.argmin(ngsim_rmse),:,:,:]  # 43 np.argmin(ngsim_rmse)
ngsim_input = data_ngsim[0][np.argmin(ngsim_rmse),:h_act,:w_mod,:]
true_speed = data_ngsim[1][0][:h_act,:w_mod,:]
input_probe = data_ngsim[-1][np.argmin(ngsim_rmse),:h_act,:w_mod]

fig_a = plt.figure(figsize=(8.5, 6))
gs1 = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.5)

ax = fig_a.add_subplot(gs1[0, 0])
plt.imshow(true_speed.reshape(h_act, w_mod), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
x_space = np.nonzero(input_probe)[0]
y_time = np.nonzero(input_probe)[1]
plt.scatter(y_time, x_space, s=3, c='k', marker='.', alpha=0.3)
ax.set_title('(a) True speed map', y=1.0, fontsize=17)
ax.set_xlabel(r'Time ($secs$)', fontsize=16)
ax.set_ylabel(r'Space ($x10$ $m$)', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)

ax = fig_a.add_subplot(gs1[1, 0])
plt.imshow(ngsim_avg_pred.reshape(h_act, w_mod), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
ax.set_title('(b) Estimated speed map', y=1.0, fontsize=17)
ax.set_xlabel(r'Time ($secs$)',fontsize=16)
ax.set_ylabel(r'Space ($x10$ $m$)',fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)

# plt.tight_layout()
plt.savefig('Ngsim_pred-TRB Poster-high res.png', dpi=1200, bbox_inches='tight')


# =============================================================================
# NGSIM: Data requirement analysis
# =============================================================================

# Parameters
c=3; h=80; w=60
h_act=67; w_act=2760
max_speed = 95
x_test = np.array([])
y_test = np.array([])
datasets = ['Ngsim']

# Load data
for df in datasets:
    for i in ['us101_lane2']: 
        output_Y = np.load('./5_Testing data/{}/out_data_{}_{}.npy'.format(df,df,i))
        input_X = np.load('./5_Testing data/{}/inp_data_{}_{}.npy'.format(df,df,i))
        if len(x_test) == 0:
            x_test = input_X[:, :, :w_act, :]
            y_test = output_Y[:, :, :w_act, :]
        else:
            x_test = np.append(x_test, input_X[:, :, :w_act, :], axis=0)
            y_test = np.append(y_test, output_Y[:, :, :w_act, :], axis=0)
x_bin = (x_test.sum(axis=3) != 0)
y_map = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
x_map = x_bin*y_map
data_ngsim = [x_test, y_test, x_map]

# Evaluate model on NGSIM data
for i in [1,2,3,4,5,6,7,8]:
    
    # Load the base trained model
    base_model = load_model('./3_Training experiments/expt-28 (datareq)/Expt28_anisomodel_5p-{}.h5'.format(i),
                            custom_objects={'Tilted_kernel':Tilted_kernel}, 
                            compile=False)
    # NGSIM model (US-101)
    ngsim_model = CNN_model(h=80, w=2760)
    ngsim_model.set_weights(base_model.get_weights())
    # Predict (crop out the zero-padded region)
    w_mod = 2400
    ngsim_pred = ngsim_model.predict(data_ngsim[0])[:,:h_act,:w_mod,:]
    # RMSE error
    ngsim_rmse = []
    true_array = data_ngsim[1][0]
    for s in range(ngsim_pred.shape[0]):
        true_y = true_array[:h_act,:w_mod,:].reshape(-1)
        pred_y1 = ngsim_pred[s,:h_act,:w_mod,:].reshape(-1)
        rmse1 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y1)*max_speed, 2))), 3)
        ngsim_rmse.append(rmse1)
    print(f'\tRMSE is {np.mean(ngsim_rmse):0.03f} +- {np.std(ngsim_rmse):0.03f}')



