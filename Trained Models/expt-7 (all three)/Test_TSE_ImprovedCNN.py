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
from keras.models import load_model, Model
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import ColorMapping as cmg
import pickle as pkl

import matplotlib
matplotlib.rcParams['text.usetex'] = False
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

def CNN_model(h=80, w=60, n1=5, n2=3):
    
    inp_dep = 3
    input_img = Input(shape=(h, w, inp_dep))
    
    # --------- Encoder model ----------- #
    # Layer 1
    nrows = n1; ncols = n1; inp_dep = 3; out_dep = 8;
    cnn_1 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(input_img)
    mpl_1 = MaxPooling2D((2, 3), padding='same')(cnn_1)
    # Layer 2
    nrows = n1; ncols = n1; inp_dep = 8; out_dep = 32;
    cnn_2 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(mpl_1)
    mpl_2 = MaxPooling2D((2, 2), padding='same')(cnn_2)
    # Layer 3
    nrows = n2; ncols = n2; inp_dep = 32; out_dep = 64;
    cnn_3 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(mpl_2)
    encoder = MaxPooling2D((2, 2), padding='same')(cnn_3)
    # --------- Decoder model --------- #
    # Layer 4
    nrows = n2; ncols = n2; inp_dep = 64; out_dep = 64;
    cnn_4 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(encoder)
    usl_4 = UpSampling2D((2, 2))(cnn_4)
    # Layer 5
    nrows = n1; ncols = n1; inp_dep = 64; out_dep = 32;
    cnn_5 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(usl_4)
    usl_5 = UpSampling2D((2, 2))(cnn_5)
    # Layer 6
    nrows = n1; ncols = n1; inp_dep = 32; out_dep = 8;
    cnn_6 = Conv2D(out_dep, (nrows, ncols), activation='relu', padding='same')(usl_5)
    usl_6 = UpSampling2D((2, 3))(cnn_6)
    # Output layer
    nrows = n1; ncols = n1; inp_dep = 8; out_dep = 1;
    decoder = Conv2D(out_dep, (nrows, ncols), activation='sigmoid', padding='same')(usl_6)

    return Model(input_img, decoder)

def count_neurons(model):
    c = 0
    for l in model.layers:
        if 'conv2d' in l.name:
            c += np.sum(np.sum(np.abs(l.get_weights()[0]), axis=(0,1,2)) > 10**-3)
    return c


# =============================================================================
# Load model and data
# =============================================================================

# Load data
c=3
h=80
h_act=80
w=60
max_speed = 100
# num=int(17940*0.9)
# y_test = np.load('./Training data/Old simulation/out_data_5hr.npy')[num:,:h,:w,:c-2]
# x_test = np.load('./Training data/Old simulation/inp_data_5hr.npy')[num:,:h,:w,:c]

train_per = 0
x_test = np.array([])
y_test = np.array([])
traj_occ = np.array([])
datasets = ['Wide_cong_more_20p']

for df in datasets:
    
    for i in [1,2,3,4,5,6]:
        
        if (i>3) and (df!='Wide_free'):
            continue
    
        output_Y = np.load('./2_Training data/With mask/out_data_{}.npy'.format(df))
        input_X = np.load('./2_Training data/With mask/inp_data_{}.npy'.format(df))
        traj_OCC = np.load('./2_Training data/With mask/traj_occ_{}.npy'.format(df))
        train_nums = int(output_Y.shape[0]*train_per)
        print(train_nums)
        
        if len(x_test) == 0:
            x_test = input_X[train_nums:, :, :, :]
            y_test = output_Y[train_nums:, :, :, :]
            traj_occ = traj_OCC[train_nums:, :, :, :]
        else:
            x_test = np.append(x_test, input_X[train_nums:, :, :, :], axis=0)
            y_test = np.append(y_test, output_Y[train_nums:, :, :, :], axis=0)
            traj_occ = np.append(traj_occ, traj_OCC[train_nums:, :, :, :], axis=0)

# Load models
# tse_cnn1 = load_model("./3_Training experiments/expt-7 (all three)/Expt4_model_nomask-bce.h5",
#                         custom_objects={'Tilted_kernel':Tilted_kernel}, 
#                         compile=False)
# tse_cnn2 = load_model('./3_Training experiments/expt-7 (all three)/Expt5_model_mask1-bce.h5',
#                         custom_objects={'Tilted_kernel':Tilted_kernel}, 
#                         compile=False)
tse_cnn3 = load_model('./3_Training experiments/expt-10 (20percent)/Expt10_model_mask2-bce.h5',
                        custom_objects={'Tilted_kernel':Tilted_kernel}, 
                        compile=False)

# Predict test samples
# iso_predY = tse_cnn1.predict(x_test)
# ani1_predY = tse_cnn2.predict(x_test)
ani2_predY = tse_cnn3.predict(x_test)

# =============================================================================
# Training performance measures
# =============================================================================

loss_fn='bce'
fold_loc='./3_Training experiments/expt-10 (20percent)/'

# Load loss functions
# iso_loss = np.load(fold_loc+'Expt4_mse-{}.npy'.format(loss_fn)) #Loss_values-bce less_complex1/Loss_values
# ani1_loss = np.load(fold_loc+'Expt5_mse-{}.npy'.format(loss_fn))
ani2_loss = np.load(fold_loc+'Expt10_mse-{}.npy'.format(loss_fn))
# iso_valloss = np.load(fold_loc+'Expt4_val_mse-{}.npy'.format(loss_fn))
# ani1_valloss = np.load(fold_loc+'Expt5_val_mse-{}.npy'.format(loss_fn))
ani2_valloss = np.load(fold_loc+'Expt10_val_mse-{}.npy'.format(loss_fn))

# Plot loss functions
plt.figure(figsize=(6,6))
# plt.plot(np.sqrt(iso_loss)*100, label='iso (train)', lw=2)
# plt.plot(np.sqrt(iso_valloss)*100, label='iso (test)', lw=2)
# plt.plot(np.sqrt(ani1_loss)*100, label='ker1 (train)', lw=2)
# plt.plot(np.sqrt(ani1_valloss)*100, label='ker1 (test)', lw=2)
plt.plot(np.sqrt(ani2_loss)*100, label='ker2 (train)', lw=2)
plt.plot(np.sqrt(ani2_valloss)*100, label='ker2 (test)', lw=2)
plt.xlabel('Training epochs', fontsize=14)
plt.ylabel('MSE training loss (kmph)', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig('./Kernel_loss_comparison.png')


# =============================================================================
# Some test reconstruction results
# =============================================================================

h=80; w=60
samples = np.arange(0,len(x_test),1)
n = np.random.choice(samples,4)

plt.figure(figsize=(9,9))
for i, j in enumerate(n):
    # display true speed field (true output)
    ax = plt.subplot(5, len(n), i + 1)
    plt.imshow(y_test[j].reshape(h,w), cmap='jet_r', vmin=0, vmax=1)
    ax.set_title('True', fontsize=12)
    ax.set_axis_off()
    
    # display probe trajectory (input)
    ax = plt.subplot(5, len(n), i  + len(n) + 1)
    plt.imshow(x_test[j].reshape(h,w,3), cmap='jet_r', vmin=0, vmax=1)
    ax.set_title('Probe', fontsize=12)
    ax.set_axis_off()
    
    # display isotropic kernel reconstruction (pred output)
    ax = plt.subplot(5, len(n), i + 2*len(n) + 1)
    plt.imshow(iso_predY[j].reshape(h,w), cmap='jet_r', vmin=0, vmax=1)
    ax.set_title('Iso Pred', fontsize=12)
    ax.set_axis_off()
    
    # display kernel 1 reconstruction (pred output)
    ax = plt.subplot(5, len(n), i + 3*len(n) + 1)
    plt.imshow(ani1_predY[j].reshape(h,w), cmap='jet_r', vmin=0, vmax=1)
    ax.set_title('Ker1 Pred', fontsize=12)
    ax.set_axis_off()
    
    # display kernel 2 reconstruction (pred output)
    ax = plt.subplot(5, len(n), i + 4*len(n) + 1)
    plt.imshow(ani2_predY[j].reshape(h,w), cmap='jet_r', vmin=0, vmax=1)
    ax.set_title('Ker2 Pred', fontsize=12)
    ax.set_axis_off()
plt.tight_layout()
plt.show()

# Speed profile
samples=np.arange(0,len(x_test),1)
j = np.random.choice(samples)
plt.figure()
for k in range(60):
    plt.plot(y_test[j][:,k]*max_speed, label='True')
    plt.plot(iso_predY[j][:,k]*max_speed, label='Iso pred')
    plt.plot(ani1_predY[j][:,k]*max_speed, label='Ker1 pred')
    plt.plot(ani2_predY[j][:,k]*max_speed, label='Ker2 pred')
    plt.xlabel('Section (x10 m)', fontsize=14)
    plt.ylabel('Speed (kmph)', fontsize=14)
    plt.legend()
    plt.ylim([0,110])
    plt.grid()
    plt.pause(0.2)
    plt.clf()


# =============================================================================
# Speed histograms and QQ-plot (More detailed analysis)
# =============================================================================

from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

true_array = y_test
pred_array1 = iso_predY
pred_array2 = ani1_predY
pred_array3 = ani2_predY

# -------------- Error and similarity over test samples ----------------- #

iso_l2 = []; ani1_l2 = []; ani2_l2 = []             # Root mean squared error
iso_cs = []; ani1_cs = []; ani2_cs = []             # Cosine similarity
for s in range(true_array.shape[0]):
    # Predictions and true samples
    true_y = true_array[s,:h_act,:,:].reshape(-1)
    pred1_y = pred_array1[s,:h_act,:,:].reshape(-1)
    pred2_y = pred_array2[s,:h_act,:,:].reshape(-1)
    pred3_y = pred_array3[s,:h_act,:,:].reshape(-1)
    # Root mean squared error
    mse1 = np.round(np.sqrt(np.mean(np.power(true_y-pred1_y, 2)))*max_speed, 3)
    mse2 = np.round(np.sqrt(np.mean(np.power(true_y-pred2_y, 2)))*max_speed, 3)
    mse3 = np.round(np.sqrt(np.mean(np.power(true_y-pred3_y, 2)))*max_speed, 3)
    # Cosine similarity
    cs1 = np.round(cosine_similarity(true_y.reshape(1,-1), pred1_y.reshape(1,-1)), 3)
    cs2 = np.round(cosine_similarity(true_y.reshape(1,-1), pred2_y.reshape(1,-1)), 3)
    cs3 = np.round(cosine_similarity(true_y.reshape(1,-1), pred3_y.reshape(1,-1)), 3)
    # Save result
    iso_l2.append(mse1); iso_cs.append(cs1.item())
    ani1_l2.append(mse2); ani1_cs.append(cs2.item())
    ani2_l2.append(mse3); ani2_cs.append(cs3.item())

# ---------------- Error evaluated on occupied cells ------------------- #

true_array_occ = np.multiply(traj_occ[:,:,:,0], y_test[:,:,:,0])
occ_ind = np.array(traj_occ[:,:,:,0], dtype=np.int)
act_sps = []
for i in range(len(occ_ind.reshape(-1))):
    if occ_ind.reshape(-1)[i] == 1:
        act_sps.append(true_array_occ.reshape(-1)[i])
act_sps = np.array(act_sps)*100

pred_array4 = np.multiply(traj_occ[:,:,:,0], ani2_predY[:,:,:,0])
pred_sps = []
for i in range(len(occ_ind.reshape(-1))):
    if occ_ind.reshape(-1)[i] == 1:
        pred_sps.append(pred_array4.reshape(-1)[i])
pred_sps = np.array(pred_sps)*100

speeds_test = np.array(traj_occ[:,:,:,1], dtype=np.int)
diff = pred_array4*max_speed - speeds_test # difference between actual speeds and predicted
diff_all = (ani2_predY[:,:,:,0] - y_test[:,:,:,0])*max_speed # difference between speed map and predicted for all pixels
root_mse = np.zeros(len(x_test))
root_mse_all = np.zeros(len(x_test))

samples = np.arange(0,len(x_test),1)

for i in samples:
    root_mse[i] = np.sqrt( np.sum(np.square(diff[i])) / (np.count_nonzero(traj_occ[i,:,:,0])-2) )
    root_mse_all[i] = np.sqrt( np.sum(np.square(diff_all[i])) / (h*w-2) )


plt.figure(figsize=(8,6))
plt.hist([act_sps, pred_sps],
          bins=40, rwidth=0.8, alpha=1.0, stacked=True, density=True)
plt.xlabel("Vehicle speed (kmph)", fontsize=12)
plt.ylabel("Counts", fontsize=12)
plt.title("Speed distributions", fontsize=14)
plt.legend(['Actual speed','Predicted speed'], fontsize=10)
plt.grid()

ani2_l2 = []                                        # Root mean squared error
for s in range(true_array.shape[0]):
    
    # Predictions and true speeds
    true_y = true_array[s,:h_act,:,0]
    pred3_y = pred_array3[s,:h_act,:,0]
    
    # Extracting predictions for occupied cells
    occ_ind = np.array(traj_occ[s,:,:,0], dtype=np.int)
    true_yocc = []
    for i in range(occ_ind.shape[0]):
        for j in range(occ_ind.shape[1]):
            if occ_ind[i,j] == 1:
                true_yocc.append(true_y[i,j])
    true_yocc = np.array(true_yocc)
    pred3_yocc = []
    for i in range(occ_ind.shape[0]):
        for j in range(occ_ind.shape[1]):
            if occ_ind[i,j] == 1:
                pred3_yocc.append(pred3_y[i,j])
    pred3_yocc = np.array(pred3_yocc)
    
    # Root mean squared error
    mse3 = np.round(np.sqrt(np.mean(np.power(true_yocc-pred3_yocc, 2)))*max_speed, 3)
    
    # Save result
    ani2_l2.append(mse3)

plt.figure()
plt.hist([ani2_l2], bins=20, rwidth=0.8, stacked=True)
plt.xlabel("Mean squared error (kmph)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend(['Iso','Ker1','Ker2'], fontsize=12)
plt.grid()

# ------------- Error and similarity density function -------------- #

plt.figure()
error_dist = [iso_l2, ani1_l2, ani2_l2]
for ls in error_dist:
    shape, loc, scale = stats.lognorm.fit(ls)
    x = np.arange(np.array(error_dist).min(), np.array(error_dist).max(), 0.01)
    plt.plot(x, stats.lognorm.pdf(x, shape, loc, scale))
plt.xlabel("Root mean squared error (kmph)", fontsize=14)
plt.ylabel("Probability density", fontsize=14)
plt.legend(['Iso','Ker1','Ker2'], fontsize=12)
plt.title('Wide', fontsize=14)
plt.tight_layout()
plt.grid()

plt.figure()
sim_dist = [iso_cs, ani1_cs, ani2_cs]
for ls in sim_dist:
    shape, loc, scale = stats.lognorm.fit(ls)
    x = np.arange(np.array(sim_dist).min(), 1, 0.001)
    plt.plot(x, stats.lognorm.pdf(x, shape, loc, scale))
plt.xlabel("Cosine similarity", fontsize=14)
plt.ylabel("Probability density", fontsize=14)
plt.legend(['Iso','Ker1','Ker2'], fontsize=12)
plt.title('Wide', fontsize=14)
plt.tight_layout()
plt.grid()


plt.figure()
plt.hist(error_dist, bins=20, rwidth=0.8, stacked=True)
plt.xlabel("Mean squared error (kmph)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend(['Iso','Ker1','Ker2'], fontsize=12)
plt.grid()

np.mean(iso_l2)
np.mean(ani1_l2)
np.mean(ani2_l2)

np.std(iso_l2)
np.std(ani1_l2)
np.std(ani2_l2)


# Speed Q-Q plot

num_sam = 200
sam_indx = np.random.randint(1, true_array.reshape(-1).shape[0], num_sam)
true_vals = (true_array.reshape(-1)*max_speed)[sam_indx]
pred1_vals = (pred_array1.reshape(-1)*max_speed)[sam_indx]
pred2_vals = (pred_array2.reshape(-1)*max_speed)[sam_indx]
pred3_vals = (pred_array3.reshape(-1)*max_speed)[sam_indx]

plt.figure(figsize=(6,5))
plt.scatter(true_vals, pred1_vals, marker='o', ec='r', fc='None', label='iso')
plt.scatter(true_vals, pred2_vals, marker='o', ec='b', fc='None', label='ker1')
plt.scatter(true_vals, pred3_vals, marker='o', ec='g', fc='None', label='ker2')
plt.plot(np.arange(0,max_speed,1), np.arange(0,max_speed,1), 'k--', alpha=0.5)
plt.xlabel("True speed values (kmph)", fontsize=14)
plt.ylabel("Predicted speed values (kmph)", fontsize=14)
plt.xlim([0,max_speed+5]); plt.xticks(fontsize=12)
plt.ylim([0,max_speed+5]); plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid()




# =============================================================================
# Eulerian to lagrangian trajectory analysis
# =============================================================================

def genTraj(d_0, t_0, s_0, n, pred_map):
    
    locs = []; locs.append(d_0)
    tims = []; tims.append(t_0)
    spds = []; spds.append(s_0)
    
    for i in range(n):
        
        d_1 = d_0 + s_0*5/18
        t_1 = t_0 + 1
        d_ind = int(pred_map.shape[0]-np.floor(d_1/10))
        t_ind = int(t_1)
        try:
            s_1 = pred_map[d_ind, t_ind]
        except:
            break
        locs.append(d_1); d_0 = d_1
        tims.append(t_1); t_0 = t_1
        spds.append(s_1); s_0 = s_1
    
    return [tims, locs, spds]


# Load trajectories, input and output
dt_name='Wide_cong'
inp_file = open('./2_Training data/Traj_analysis/veh_traj_{}.pkl'.format(dt_name), 'rb')
veh_traj = pkl.load(inp_file)
inp_file.close()
inp_map = np.load('./2_Training data/Traj_analysis/inp_map_{}.npy'.format(dt_name))
out_map = np.load('./2_Training data/Traj_analysis/out_map_{}.npy'.format(dt_name))

# Select vehicles completely traversing whole sec length
sel_vehs = []
for v_id in veh_traj.keys():
    entry_loc = veh_traj[v_id][1][0] <= 50
    exit_loc = veh_traj[v_id][1][-1] >= 750
    if entry_loc and exit_loc:
        sel_vehs.append(v_id)
veh_traj_sel = {}
for v_id in veh_traj.keys():
    if v_id in sel_vehs:
        veh_traj_sel[v_id] = veh_traj[v_id]

# Predict the output speed map (eulerian coords)
tse_cnn4 = CNN_model(h=80, w=360, n1=7, n2=5)
tse_cnn4.set_weights(tse_cnn3.get_weights())
pred_map = tse_cnn4.predict(inp_map.reshape(1,80,360,3)).reshape(80,360)
pred_map *= 100

# Predict the output vehicle trajectories (lagrangian coords)
pred_traj = {}
for v_id in veh_traj_sel.keys():
    
    t_0 = veh_traj_sel[v_id][0][0]
    d_0 = veh_traj_sel[v_id][1][0]
    s_0 = veh_traj_sel[v_id][2][0]
    n = len(veh_traj_sel[v_id][0])
    
    traj = genTraj(d_0, t_0, s_0, n, pred_map)
    pred_traj[v_id] = traj

# Compare predicted and true trajectories
test_vehs = list(veh_traj_sel.keys())

plt.figure()
for v_id in test_vehs[::3]:
    t = veh_traj_sel[v_id][0]
    d = veh_traj_sel[v_id][1]
    s = veh_traj_sel[v_id][2]
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=100)
plt.ylim([0,800])
plt.colorbar()
plt.grid()
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Space (m)', fontsize=12)
plt.title('True trajectories', fontsize=13)

plt.figure()
for v_id in test_vehs[::3]:
    t = pred_traj[v_id][0]
    d = pred_traj[v_id][1]
    s = pred_traj[v_id][2]
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=100)
    plt.plot(t, d, lw=0.2, c='k')
plt.ylim([0,800])
plt.colorbar()
plt.grid()
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Space (m)', fontsize=12)
plt.title('Estimated trajectories', fontsize=13)

# =============================================================================
# Detailed analysis
# =============================================================================

# # Define a mapping function
# colmap = cmg.MplColorMapper(cmap_name='jet_r', min_val=0, 
#                             max_val=60, lookuptable_dim=256)

# # Let's choose a sample
# sam_num = 200

# # Load the actual grid information in Eulerian coordinates for the sample
# pocc_grid = np.load('./test_probe_grid.npy')
# focc_grid = np.load('./test_whole_grid.npy')
# psam_grid = np.flipud(pocc_grid[sam_num:sam_num+60, :, :].T[0,:80,:])
# fsam_grid = np.flipud(focc_grid[sam_num:sam_num+60, :, :].T[0,:80,:])
# fig, ((ax1,ax2)) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7,4))
# ax1.imshow(psam_grid.astype(np.float32), cmap='Blues', vmin=0, vmax=1)
# ax2.imshow(fsam_grid.astype(np.float32), cmap='Blues', vmin=0, vmax=1)
# ax1.set_title('Probe traj occup', fontsize=12)
# ax2.set_title('Full traj occup', fontsize=12)

# # Let's compare the actual prediction and masked prediction
# maskpred_Y = np.multiply(pred_Y[sam_num].reshape(h,w,3), np.repeat(fsam_grid[:80,:].reshape(80,60,1), 3, axis=2))
# fig, ((ax1,ax2,ax3,ax4)) = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(14,4))
# ax1.imshow(prob_X[sam_num].reshape(h,w,3)); ax1.set_title('Probe traj', fontsize=12)
# ax2.imshow(pred_Y[sam_num].reshape(h,w,3)); ax2.set_title('Pred traj', fontsize=12)
# ax3.imshow(true_Y[sam_num].reshape(h,w,3)); ax3.set_title('Actual traj', fontsize=12)
# ax4.imshow(maskpred_Y); ax4.set_title('Masked pred traj', fontsize=12)

# # Let's reconstruct the speed from RGB
# pv_grid = np.flipud(pocc_grid[sam_num:sam_num+60, 1:81, :].T[1,:,:])
# fv_grid = np.flipud(focc_grid[sam_num:sam_num+60, 1:81, :].T[1,:,:])
# fv_pred = np.array(colmap.get_sval_mult(maskpred_Y.reshape(-1,3))).reshape(80,60) # true_Y[sam_num].reshape(h,w,3).reshape(-1,3)

# fig, ((ax1,ax2)) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7,4))
# ax1.imshow(fv_grid.astype(np.float32), cmap='jet_r', vmin=0, vmax=60)
# ax2.imshow(fv_pred.astype(np.float32), cmap='jet_r', vmin=0, vmax=60)
# ax1.set_title('Act traj speed', fontsize=12)
# ax2.set_title('Pred traj speed', fontsize=12)

# fig, ax1 = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(4,4))
# sp = ax1.imshow(np.abs(fv_pred-fv_grid.astype(np.float32)), cmap='Blues')
# fig.colorbar(sp, ax=ax1)
