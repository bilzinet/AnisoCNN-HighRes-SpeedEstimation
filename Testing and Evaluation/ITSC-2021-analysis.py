# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:32:36 2021
@author: btt1
Traffic State Estimation using Anisotropic Deep CNN Model
For IEEE-ITSC Conference 2021
"""

# =============================================================================
# Import libraries
# =============================================================================

import time
import keras
import numpy as np
import pickle as pkl
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from keras.models import load_model, Model
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
# import ColorMapping as cmg

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
np.set_printoptions(precision=4, suppress=True)


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
# Load data (5 percentage probe)
# =============================================================================

# Load simulation data
c=3; h=80; h_act=80; w=60
max_speed=100; train_per = 0.95 #0.95 or 0.9
x_test = np.array([])
y_test = np.array([])
data_sep = []
traffic_type = np.array([])
datasets = ['Wide_free_sp','Wide_cong_sp','Wide_cong_more_sp']

for k,df in enumerate(datasets):
    for i in [1,2,3,4,5,6]: 
        if (i>3) and (df!=datasets[0]):
            continue
    
        input_X = np.load('./2_Training data/{}/inp_data_{}{}.npy'.format(df,df,i))
        nonempty_indx = np.load('./2_Training data/{}/noempty_indx_{}{}.npy'.format(df,df,i))
        output = np.load('./2_Training data/{}/out_data_{}.npy'.format(df,df))
        output_Y = output[nonempty_indx.reshape(-1),:,:,:]
        train_nums = int(output_Y.shape[0]*train_per)
        test_nums = int(output_Y.shape[0]-train_nums)
        
        if len(x_test) == 0:
            x_test = input_X[train_nums:, :, :, :3]
            y_test = output_Y[train_nums:, :, :, :]
            traffic_type = np.append(traffic_type, np.repeat(k, test_nums))
        else:
            x_test = np.append(x_test, input_X[train_nums:, :, :, :3], axis=0)
            y_test = np.append(y_test, output_Y[train_nums:, :, :, :], axis=0)
            traffic_type = np.append(traffic_type, np.repeat(k, test_nums))
    
    # separating index between each dataset
    data_sep.append(len(x_test))

# Input speed maps
x_bin = (x_test.sum(axis=3) != 0)
y_map = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
x_map = x_bin*y_map
sim_data = [x_test, y_test, x_bin, x_map]


# =============================================================================
# Load models
# =============================================================================

# Load an anisotropic and isotropic model and predict
fold_loc='./3_Training experiments/expt-24 (iso-ani-compare)/'
ani_model = load_model(fold_loc + './Expt24_anisomodel_5p-1.h5',
                       custom_objects={'Tilted_kernel':Tilted_kernel}, 
                       compile=False)
iso_model = load_model(fold_loc + './Expt24_isomodel_5p-1.h5',
                       custom_objects={'Tilted_kernel':Tilted_kernel}, 
                       compile=False)
ani_predY = ani_model.predict(x_test)
iso_predY = iso_model.predict(x_test)


# Create models for NGSIM and HighD datasets
# Load the trained anisotropic model as the base model
base_model = load_model('./3_Training experiments/expt-24 (iso-ani-compare)/Expt24_anisomodel_5p-1.h5',
                        custom_objects={'Tilted_kernel':Tilted_kernel}, 
                        compile=False)

# NGSIM model (US-101)
# Create a dummy model and transfer weights
ngsim_model = CNN_model(h=80, w=2760) # 2760 for US101 and 1810 for i80
ngsim_model.set_weights(base_model.get_weights())

# HighD model (HW-)
# Create a dummy model and transfer weights
highd_model = CNN_model(h=80, w=1140)  # w=276
highd_model.set_weights(base_model.get_weights())


# =============================================================================
# Problem statement, image representation
# =============================================================================

# ---------- Input-Output representation ------------ #

sample = 2325 # 2350 or 3000
h=80; w=60

a=x_test[sample]
b=x_bin.astype(int)[sample][:,:,None]
c=np.concatenate((a, b), axis=-1)

fig = plt.figure(figsize=(4,2.5))
plt.subplot(121)
im1 = plt.imshow(a, cmap='jet_r', vmin=0, vmax=1, aspect='equal')
plt.xlabel('$Time~[s]$', fontsize=9)
plt.ylabel('$Space~[m]$', fontsize=9)
plt.title('(a) Input', y=-0.4, fontsize=10)
plt.yticks([0,20,40,60,80])
ytick_locs, ytick_labels = plt.yticks()
ytick_labels = ['800','600','400','200','0']
plt.yticks(ytick_locs, ytick_labels, fontsize=7)
plt.xticks([0,20,40,60], fontsize=7)
plt.ylim([80,0])

plt.subplot(122)
im2 = plt.imshow(y_test[sample].reshape(h,w), cmap='jet_r', vmin=0, vmax=1, aspect='equal')
plt.xlabel('$Time~[s]$', fontsize=9)
plt.title('(b) Output', y=-0.4, fontsize=10)
ytick_locs, ytick_labels = plt.yticks()
plt.yticks([0,20,40,60,80])
plt.yticks(ytick_locs, [], fontsize=7)
plt.xticks([0,20,40,60], fontsize=7)
plt.ylim([80,0])

cbar_ax = fig.add_axes([0.93, 0.25, 0.02, 0.45])
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.ax.tick_params(labelsize=7)
cbar.ax.set_ylabel('$V~[kmph]$',fontsize=8)
cbar.set_ticks([0,0.25,0.5,0.75,1.])
cbar.set_ticklabels([0,25,50,75,100])
plt.savefig('inp_out.pdf', dpi=1200, bbox_inches='tight')


# ----------

a=x_test[sample]
b=x_bin.astype(int)[sample][:,:,None]
c=np.concatenate((a, b), axis=-1)

plt.figure(figsize=(4,4))
plt.subplot(131)
im1 = plt.imshow(a[:,:,0], cmap='Reds', vmin=0, vmax=1, aspect='equal')
plt.subplot(132)
im1 = plt.imshow(a[:,:,1], cmap='Greens', vmin=0, vmax=1, aspect='equal')
plt.subplot(133)
im1 = plt.imshow(a[:,:,2], cmap='Blues', vmin=0, vmax=1, aspect='equal')


# =============================================================================
# NGSIM: Load, predict, RMSE, visualize
# =============================================================================

# Parameters
c=3; h=80; w=60
h_act=67; w_act=2760    # 67,2760 for US101 and 53,1810 for I-80
max_speed = 95
x_test = np.array([])
y_test = np.array([])
datasets = ['Ngsim']

# Load data
for df in datasets:
    for i in ['us101_lane2']: # us101_lane2, i80_lane1_2, i80_lane3_2, i80_lane6_2
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
data_ngsim = [x_test, y_test, x_bin, x_map]

# Predict (crop out the zero-padded region)
w_bel = 200     # 200 for US-101 and 100 for I-80
w_mod = 2000    # 2000 for US-101 and 1600 for I-80 
h_mod = 60      # 60 for US-101 and 50 for I-80
h_bel = 10
ngsim_pred = ngsim_model.predict(data_ngsim[0])[:,h_bel:h_mod,w_bel:w_mod,:]

# RMSE error
ngsim_rmse = []
true_array = data_ngsim[1][0]
for s in range(ngsim_pred.shape[0]):
    true_y = true_array[h_bel:h_mod,w_bel:w_mod,:].reshape(-1)
    pred_y1 = ngsim_pred[s,:h_mod,:w_mod,:].reshape(-1)
    rmse1 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y1)*max_speed, 2))), 3)
    ngsim_rmse.append(rmse1)
print('Average RMSE for prediction: ', np.mean(ngsim_rmse))
print('Standard deviation is: ', np.std(ngsim_rmse))

# Visualize estimation results
best_sam = 41 # 41 for US-101, 11 for I-80 lane 3
ngsim_avg_pred = ngsim_pred.mean(axis=0)
ngsim_avg_pred = ngsim_pred[best_sam,:,:,:]  # 43
ngsim_input = data_ngsim[0][best_sam,h_bel:h_mod,w_bel:w_mod,:]
true_speed = data_ngsim[1][0][h_bel:h_mod,w_bel:w_mod,:]
input_probe = data_ngsim[-1][best_sam,h_bel:h_mod,w_bel:w_mod]
input_probe_bin = data_ngsim[-2][best_sam,h_bel:h_mod,w_bel:w_mod]*1.0


fig_a = plt.figure(figsize=(4, 5))
gs1 = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.5)

# display true speed field (true output)
ax = fig_a.add_subplot(gs1[0, 0])
map1 = plt.imshow(true_speed.reshape(h_mod-h_bel, w_mod-w_bel)*max_speed, 
                  cmap='jet_r', vmin=0, vmax=max_speed, aspect='auto')
# map3 = plt.imshow(input_probe_bin, alpha=input_probe_bin,
#                   cmap='Greys', aspect='auto')
x_space = np.nonzero(input_probe)[0]
y_time = np.nonzero(input_probe)[1]
plt.scatter(y_time, x_space, s=3, c='k', marker='.', alpha=0.4)
ax.set_title('(a) True speed map', y=-0.48, fontsize=12)
ax.set_xlabel('$Time~[s]$', fontsize=11)
ax.set_ylabel('$Space~[m]$', fontsize=11)
ax.set_xticks([0, 300, 600, 900, 1200, 1500, 1800]) #[0, 400, 800, 1200, 1600]
ax.set_yticks([0, 25, 50])   # [0, 20, 40] for I-80
ytick_labels = ['500','250','0'] # [400, 200, 0]
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=10)
cbar_ax = fig_a.add_axes([0.93, 0.60, 0.02, 0.25])
cbar = fig_a.colorbar(map1, cax=cbar_ax)
cbar.ax.tick_params(labelsize=9)
cbar.ax.set_ylabel('$V~[kmph]$')

ax = fig_a.add_subplot(gs1[1, 0])
plt.imshow(ngsim_avg_pred.reshape(h_mod-h_bel, w_mod-w_bel)*max_speed, 
           cmap='jet_r', vmin=0, vmax=max_speed, aspect='auto')
ax.set_title('(b) Estimated speed map', y=-0.48, fontsize=12)
ax.set_xlabel('$Time~[s]$',fontsize=11)
ax.set_ylabel('$Space~[m]$',fontsize=11)
ax.set_xticks([0, 300, 600, 900, 1200, 1500, 1800]) # [0, 400, 800, 1200, 1600]
ax.set_yticks([0, 25, 50])   # [0, 20, 40] for I-80
ytick_labels = ['500','250','0']  # [400, 200, 0]
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=10)

plt.savefig('Ngsim_pred-us101.pdf', dpi=900, bbox_inches='tight')

# =============================================================================
# HighD dataset: Load, predict, RMSE, visualize
# =============================================================================

# Parameters
c=3; h=80; w=60
h_act=40; w_act=1140
max_speed = 165  # 75 for hw25 lane 4; 165 for hw44 lane 6
x_test = np.array([])
y_test = np.array([])
datasets = ['HighD']

# Load data
for df in datasets:
    for i in ['track44_lane6-5p']:  # track44_lane6 and track25_lane4
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
data_highd = [x_test, y_test, x_bin, x_map]

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
best_sam=3 # 3 for hw-44 and 10 for hw-25
highd_avg_pred = highd_pred.mean(axis=0)
highd_avg_pred = highd_pred[best_sam,:,:,:]  
highd_input = data_highd[0][best_sam,:h_act,:w_mod,:]
true_speed = data_highd[1][0][:h_act,:w_mod,:]
input_probe = data_highd[-1][best_sam,:h_act,:w_mod]
input_probe_bin = data_highd[-2][best_sam,:h_act,:w_mod]*1.0

fig_a = plt.figure(figsize=(4, 5))
gs1 = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.5)

# display true speed field (true output)
ax = fig_a.add_subplot(gs1[0, 0])
map1 = ax.imshow(true_speed.reshape(h_act, w_mod)*max_speed, 
                 cmap='jet_r', vmin=0, vmax=max_speed, aspect='auto')
map3 = plt.imshow(input_probe_bin, alpha=input_probe_bin,
                  cmap='binary', aspect='auto')
x_space = np.nonzero(input_probe)[0]
y_time = np.nonzero(input_probe)[1]
# plt.scatter(y_time, x_space, s=5, c='k', marker='.', alpha=0.4)
ax.set_title('(a) True speed map', y=-0.48, fontsize=12)
ax.set_xlabel('$Time~[s]$', fontsize=11)
ax.set_ylabel('$Space~[m]$', fontsize=11)
ax.set_yticks([0, 20, 40])
ytick_labels = ['400','200','0']
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=10)
cbar_ax = fig_a.add_axes([0.93, 0.60, 0.02, 0.25])
cbar = fig_a.colorbar(map1, cax=cbar_ax)
cbar.ax.tick_params(labelsize=9)
cbar.ax.set_ylabel('$V~[kmph]$')

# display reconstructed speed field
ax = fig_a.add_subplot(gs1[1, 0])
plt.imshow(highd_avg_pred.reshape(h_act, w_mod), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
ax.set_title('(b) Estimated speed map', y=-0.48, fontsize=12)
ax.set_yticks([0, 20, 40])
ytick_labels = ['400','200','0']
ax.set_yticklabels(ytick_labels)
ax.set_xlabel('$Time~[s]$',fontsize=11)
ax.set_ylabel('$Space~[m]$',fontsize=11)
plt.setp(ax.get_xticklabels(), fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=10) 

plt.savefig('Highd_pred-hw44.pdf', bbox_inches='tight')

# =============================================================================
# T-SNE Mapping
# =============================================================================

# Selecting samples for mapping
samples = x_test.copy()
num_samples = samples.shape[0]
# num_samples = 5000
# rand_index_set = np.random.choice(x_test.shape[0], num_samples, replace=False)
samples = x_test[:,:,:,:]

# Building the encoder model 
layer_name = ani_model.layers[6].name
encoder_model = keras.Model(inputs=ani_model.input,
                            outputs=ani_model.get_layer(layer_name).output)
encoder_output = encoder_model.predict(samples)
tsne_input = encoder_output.copy().reshape(num_samples, -1)

# ------------- t-SNE 2D mapping -------------- #

# Running t-SNE algorithm
start_time = time.time()
tsne2D_model = TSNE(n_components=2, n_iter=1000)
tsne2D_output = tsne2D_model.fit_transform(tsne_input)
end_time = time.time()
total_time = np.round(end_time - start_time, 3)
print('Time for TSNE mapping %0.03f secs' % total_time)

# Aggregating
tsne2D_df = pd.DataFrame(columns=['dim-1','dim-2','label'])
tsne2D_df['dim-1'] = tsne2D_output[:,0]
tsne2D_df['dim-2'] = tsne2D_output[:,1]
tsne2D_df['label'] = traffic_type[:]
traffic_lab = []
for i in tsne2D_df['label']:
    if i == 0:
        traffic_lab.append('Free flow')
    elif i == 1:
        traffic_lab.append('Slow moving')
    else:
        traffic_lab.append('Congested')
tsne2D_df['Traffic regime'] = traffic_lab

# Visualization
sns.set(rc={'figure.facecolor':'white'})
plt.figure(figsize=(3,3))
sns.scatterplot(
    x="dim-1", y="dim-2",
    hue="Traffic regime",
    style="Traffic regime",
    palette=sns.color_palette("hls", 3),
    data=tsne2D_df,
    legend="full",
    s=10,
    alpha=0.3)
plt.xlabel('$Dimension~1$', fontsize=9)
plt.ylabel('$Dimension~2$', fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.legend(fontsize=7)
plt.savefig('tsne2D-5p.pdf', dpi=900, bbox_inches='tight')


# =============================================================================
# Eulerian to lagrangian trajectory analysis
# =============================================================================

def genTraj(d_0, t_0, s_0, pred_map):
    
    tims = []; tims.append(t_0)
    locs = []; locs.append(d_0)
    spds = []; spds.append(s_0)
    
    d_1 = 0 
    while d_1 < (h_mod-h_bel)*10:
        print('yes')
        d_1 = d_0 + s_0*5/18
        t_1 = t_0 + 1
        d_ind = int(pred_map.shape[0]-np.floor(d_1/10)-1)
        t_ind = int(t_1-w_bel)
        try:
            s_1 = pred_map[d_ind, t_ind]
        except:
            break
        locs.append(d_1); d_0 = d_1
        tims.append(t_1); t_0 = t_1
        spds.append(s_1); s_0 = s_1
    
    return [tims, locs, spds]

# ---------------- US 101 ---------------- #

# specific params
w_bel = 200
w_mod = 2000
h_mod = 60
h_bel = 10
max_speed = 95
best_map = 41
ani_map = ngsim_pred[best_map,:,:,0]*max_speed

# Load boundary condition and actual trajectories
with open('US101_ActTraj.pkl', 'rb') as f:
    act_traj = pkl.load(f)
with open('US101_BoundCond.pkl', 'rb') as f:
    bound_cond = pkl.load(f)
x0 = bound_cond['x']
t0 = bound_cond['t']
v0 = bound_cond['v']
bound_veh_cond = [(t0[i], int(x0/10-h_bel), v0[i]) for i in range(len(t0))]

new_bound_cond = []
for i in bound_veh_cond:
    if (i[0] >= 800) and (i[0] < 920):
        new_bound_cond.append(i)


# Predict the output vehicle trajectories (lagrangian coords)
pred_traj_ani = {}; v_id = 0
for v_cond in bound_veh_cond:
    t_0 = v_cond[0]
    d_0 = v_cond[1]
    s_0 = v_cond[2]
    if (t_0 >= w_bel) and (t_0 < w_mod):
        traj_ani = genTraj(d_0, t_0, s_0, ani_map)
        pred_traj_ani[v_id] = traj_ani
        v_id += 1

# Plot predicted
plt.figure()
for v_id in pred_traj_ani.keys():
    t = pred_traj_ani[v_id][0]
    d = pred_traj_ani[v_id][1]
    s = pred_traj_ani[v_id][2]
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=100)
    plt.plot(t, d, lw=0.2, c='k')
plt.colorbar()
plt.grid()
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Space (m)', fontsize=12)
plt.title('Anisotropic trajectories', fontsize=13)

# Plot for Paper actual and predicted trajectories
fig_a = plt.figure(figsize=(4, 6))
gs1 = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.45)

# display actual
ax = fig_a.add_subplot(gs1[0, 0])
for v_id in act_traj.VehNo.unique():
    act_traj_veh = act_traj.query('VehNo == @v_id')
    t = act_traj_veh.SimSec.to_numpy()
    d = act_traj_veh.Pos.to_numpy()
    s = act_traj_veh.Speed.to_numpy()
    plt.scatter(t, d, c=s, s=0.08, cmap='jet_r', vmin=0, vmax=100)
    plt.plot(t, d, lw=0.001, c='k', alpha=0.1)
ax.set_title('(a) True trajectories', y=-0.39, fontsize=11)
ax.set_xlabel('$Time~[s]$', fontsize=10)
ax.set_ylabel('$Space~[m]$', fontsize=10)
ax.set_xlim([800,1000])
ax.set_ylim([100,600])
# ax.set_yticks([100, 350, 600])
ytick_labels = ['0','100','200','300','400','500']
xtick_labels = ['600','650','700','750','800']
ax.set_yticklabels(ytick_labels)
ax.set_xticklabels(xtick_labels)
plt.setp(ax.get_xticklabels(), fontsize=9)
plt.setp(ax.get_yticklabels(), fontsize=9)
# cbar_ax = fig_a.add_axes([0.93, 0.60, 0.02, 0.25])
# cbar = fig_a.colorbar(map1, cax=cbar_ax)
# cbar.ax.tick_params(labelsize=9)
# cbar.ax.set_ylabel('$V~[kmph]$')

# display reconstructed
ax = fig_a.add_subplot(gs1[1, 0])
for v_id in pred_traj_ani.keys():
    t = pred_traj_ani[v_id][0]
    d = pred_traj_ani[v_id][1]
    s = pred_traj_ani[v_id][2]
    plt.scatter(t, d, c=s, s=0.08, cmap='jet_r', vmin=0, vmax=100)
    plt.plot(t, d, lw=0.001, c='k', alpha=0.1)
ax.set_title('(b) Inferred trajectories', y=-0.39, fontsize=11)
ax.set_xlim([800,1000])
ax.set_ylim([0,500])
# ax.set_yticks([0, 20, 40])
# ytick_labels = ['0','100','200','300','400','500']
# ax.set_yticklabels(ytick_labels)
ax.set_xticklabels(xtick_labels)
xtick_labels = ['600','650','700','750','800']
ax.set_xlabel('$Time~[s]$',fontsize=10)
ax.set_ylabel('$Space~[m]$',fontsize=10)
plt.setp(ax.get_xticklabels(), fontsize=9)
plt.setp(ax.get_yticklabels(), fontsize=9)

plt.savefig('US101_traj-1-2.png', bbox_inches='tight', dpi=300)




# ---------------- I-80 ----------------- #

def genTraj(d_0, t_0, s_0, pred_map):
    
    tims = []; tims.append(t_0)
    locs = []; locs.append(d_0)
    spds = []; spds.append(s_0)
    
    d_1 = 0 
    while d_1 < (h_mod-h_bel)*10:
        # print('yes')
        d_1 = d_0 + s_0*5/18
        t_1 = t_0 + 1
        d_ind = int(pred_map.shape[0]-np.floor(d_1/10)-1)
        t_ind = int(t_1-w_bel-1)
        try:
            s_1 = pred_map[d_ind, t_ind]
        except:
            break
        locs.append(d_1); d_0 = d_1
        tims.append(t_1); t_0 = t_1
        spds.append(s_1); s_0 = s_1
    
    return [tims, locs, spds]

# specific params
w_bel = 3750
w_mod = 5250
h_mod = 50
h_bel = 10
max_speed = 95
best_map = 11
ani_map = ngsim_pred[best_map,:,:,0]*max_speed

# Load boundary condition and actual trajectories
with open('I80_lane3_ActTraj.pkl', 'rb') as f:
    act_traj = pkl.load(f)
with open('I80_lane3_BoundCond.pkl', 'rb') as f:
    bound_cond = pkl.load(f)
x0 = bound_cond['x']
t0 = bound_cond['t']
v0 = bound_cond['v']
bound_veh_cond = [(t0[i], int(x0/10-h_bel), v0[i]) for i in range(len(t0))]

new_bound_cond = []
for i in bound_veh_cond:
    if (i[0] >= 4300) and (i[0] < 4600):
        new_bound_cond.append(i)

# Predict the output vehicle trajectories (lagrangian coords)
pred_traj_ani = {}; v_id = 0
for v_cond in bound_veh_cond:
    t_0 = v_cond[0]
    d_0 = v_cond[1]
    s_0 = v_cond[2]
    if (t_0 >= w_bel) and (t_0 < w_mod):
        traj_ani = genTraj(d_0, t_0, s_0, ani_map)
        pred_traj_ani[v_id] = traj_ani
        v_id += 1

# Plot predicted
plt.figure()
for v_id in pred_traj_ani.keys():
    t = pred_traj_ani[v_id][0]
    d = pred_traj_ani[v_id][1]
    s = pred_traj_ani[v_id][2]
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=100)
    plt.plot(t, d, lw=0.2, c='k')
plt.colorbar()
plt.grid()
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Space (m)', fontsize=12)
plt.title('Anisotropic trajectories', fontsize=13)

# Plot for Paper actual and predicted trajectories
fig_a = plt.figure(figsize=(4, 6))
gs1 = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.45)

# display actual
ax = fig_a.add_subplot(gs1[0, 0])
for v_id in act_traj.VehNo.unique():
    act_traj_veh = act_traj.query('VehNo == @v_id')
    t = act_traj_veh.SimSec.to_numpy()
    d = act_traj_veh.Pos.to_numpy()
    s = act_traj_veh.Speed.to_numpy()
    plt.scatter(t, d, c=s, s=0.08, cmap='jet_r', vmin=0, vmax=100)
    plt.plot(t, d, lw=0.001, c='k', alpha=0.1)
ax.set_title('(a) True trajectories', y=-0.39, fontsize=11)
ax.set_xlabel('$Time~[s]$', fontsize=10)
ax.set_ylabel('$Space~[m]$', fontsize=10)
ax.set_xlim([4500, 4700])
ax.set_ylim([100,500])
# ax.set_yticks([100, 350, 600])
ytick_labels = ['0','100','200','300','400']
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=9)
plt.setp(ax.get_yticklabels(), fontsize=9)

# display reconstructed
ax = fig_a.add_subplot(gs1[1, 0])
for v_id in pred_traj_ani.keys():
    t = pred_traj_ani[v_id][0]
    d = pred_traj_ani[v_id][1]
    s = pred_traj_ani[v_id][2]
    plt.scatter(t, d, c=s, s=0.08, cmap='jet_r', vmin=0, vmax=100)
    plt.plot(t, d, lw=0.001, c='k', alpha=0.1)
ax.set_title('(b) Inferred trajectories', y=-0.39, fontsize=11)
ax.set_xlim([4500, 4700])
ax.set_ylim([0,400])
# ax.set_yticks([0, 20, 40])
# ytick_labels = ['0','100','200','300','400','500']
# ax.set_yticklabels(ytick_labels)
ax.set_xlabel('$Time~[s]$',fontsize=10)
ax.set_ylabel('$Space~[m]$',fontsize=10)
plt.setp(ax.get_xticklabels(), fontsize=9)
plt.setp(ax.get_yticklabels(), fontsize=9)

plt.savefig('I80_lane3_traj-1.pdf', bbox_inches='tight')





# ------------- HW 44 -------------- #


def genTraj(d_0, t_0, s_0, pred_map):
    
    tims = []; tims.append(t_0)
    locs = []; locs.append(d_0)
    spds = []; spds.append(s_0)
    
    d_1 = 0 
    while d_1 < (h_mod-h_bel)*10:
        # print('yes')
        d_1 = d_0 + s_0*5/18
        t_1 = t_0 + 1
        d_ind = int(pred_map.shape[0]-np.floor(d_1/10)-1)
        t_ind = int(t_1-w_bel-1)
        try:
            s_1 = pred_map[d_ind, t_ind]
        except:
            break
        locs.append(d_1); d_0 = d_1
        tims.append(t_1); t_0 = t_1
        spds.append(s_1); s_0 = s_1
    
    return [tims, locs, spds]

# specific params
w_bel = 0
w_mod = 1140
h_mod = 40
h_bel = 5
max_speed = 165
best_map = 3
ani_map = highd_pred[best_map,h_bel:h_mod,:,0]*max_speed

# Load boundary condition and actual trajectories
with open('HW44_ActTraj.pkl', 'rb') as f:
    act_traj = pkl.load(f)
with open('HW44_BoundCond.pkl', 'rb') as f:
    bound_cond = pkl.load(f)
x0 = bound_cond['x']
t0 = bound_cond['t']
v0 = bound_cond['v']
bound_veh_cond = [(t0[i], int(x0/10-h_bel), v0[i]) for i in range(len(t0))]

new_bound_cond = []
for i in bound_veh_cond:
    if (i[0] >= 300) and (i[0] < 500):
        new_bound_cond.append(i)

# Predict the output vehicle trajectories (lagrangian coords)
pred_traj_ani = {}; v_id = 0
for v_cond in new_bound_cond:
    t_0 = v_cond[0]
    d_0 = v_cond[1]
    s_0 = v_cond[2]
    if (t_0 >= w_bel) and (t_0 < w_mod):
        traj_ani = genTraj(d_0, t_0, s_0, ani_map)
        pred_traj_ani[v_id] = traj_ani
        v_id += 1

# Plot predicted
plt.figure()
for v_id in pred_traj_ani.keys():
    t = pred_traj_ani[v_id][0]
    d = pred_traj_ani[v_id][1]
    s = pred_traj_ani[v_id][2]
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=max_speed)
    plt.plot(t, d, lw=0.2, c='k')
plt.colorbar()
plt.grid()
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Space (m)', fontsize=12)
plt.title('Anisotropic trajectories', fontsize=13)


df1 = act_traj.query('SimSec >= 300')
df2 = df1.query('SimSec < 500')
plt.figure()
for v_id in df2.VehNo.unique():
    act_traj_veh = df2.query('VehNo == @v_id')
    t = act_traj_veh.SimSec.to_numpy()
    d = act_traj_veh.Pos.to_numpy()
    s = act_traj_veh.Speed.to_numpy()
    plt.scatter(t, d, c=s, s=6, cmap='jet_r', vmin=0, vmax=max_speed)
    plt.plot(t, d, lw=0.4, c='k', alpha=0.4)
plt.colorbar()
plt.grid()
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Space (m)', fontsize=12)
plt.title('Actual trajectories', fontsize=13)

# Plot for Paper actual and predicted trajectories
fig_a = plt.figure(figsize=(4, 6))
gs1 = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.4)

# display actual
ax = fig_a.add_subplot(gs1[0, 0])
for v_id in act_traj.VehNo.unique():
    act_traj_veh = act_traj.query('VehNo == @v_id')
    t = act_traj_veh.SimSec.to_numpy()
    d = act_traj_veh.Pos.to_numpy()
    s = act_traj_veh.Speed.to_numpy()
    plt.scatter(t, d, c=s, s=3, cmap='jet_r', vmin=0, vmax=max_speed)
    plt.plot(t, d, lw=0.4, c='k', alpha=0.4)
# ax.invert_yaxis()
ax.set_title('(a) True trajectories', y=-0.35, fontsize=11)
ax.set_xlabel('$Time~[s]$', fontsize=10)
ax.set_ylabel('$Space~[m]$', fontsize=10)
ax.set_xlim([300,500])
ax.set_ylim([50,400])
ax.set_yticks([50, 150, 250, 350])
# ytick_labels = ['400','300','200','100','0']
# ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_xticklabels(), fontsize=9)
plt.setp(ax.get_yticklabels(), fontsize=9)

# display reconstructed
ax = fig_a.add_subplot(gs1[1, 0])
for v_id in pred_traj_ani.keys():
    t = pred_traj_ani[v_id][0]
    d = pred_traj_ani[v_id][1]
    s = pred_traj_ani[v_id][2]
    plt.scatter(t, d, c=s, s=3, cmap='jet_r', vmin=0, vmax=max_speed)
    plt.plot(t, d, lw=0.4, c='k', alpha=0.4)
ax.set_title('(b) Inferred trajectories', y=-0.35, fontsize=11)
ax.set_xlim([300,500])
ax.set_ylim([0,350])
ax.set_yticks([0, 100, 200, 300])
ytick_labels = ['50','150','250','350']
ax.set_yticklabels(ytick_labels)
ax.set_xlabel('$Time~[s]$',fontsize=10)
ax.set_ylabel('$Space~[m]$',fontsize=10)
plt.setp(ax.get_xticklabels(), fontsize=9)
plt.setp(ax.get_yticklabels(), fontsize=9)

plt.savefig('HW44_lane3_traj.pdf', bbox_inches='tight')