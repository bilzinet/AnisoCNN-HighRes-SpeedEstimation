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
# import ColorMapping as cmg
import pickle as pkl

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
np.set_printoptions(precision=4, suppress=True)

from matplotlib import rcParams
plt.rcParams["font.family"] = "Times New Roman"
rcParams['mathtext.fontset']='cm'
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Tahoma']

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
# Load data (5 percentage probe)
# =============================================================================

# Load data
c=3; h=80; h_act=80; w=60
max_speed=100; train_per = 0.95
x_test = np.array([]); y_test = np.array([]); data_sep = []
datasets = ['Wide_free_sp','Wide_cong_sp','Wide_cong_more_sp']

for df in datasets:
    for i in [1,2,3,4,5,6]: 
        if (i>3) and (df!=datasets[0]):
            continue
    
        input_X = np.load('./2_Training data/{}/inp_data_{}{}.npy'.format(df,df,i))
        nonempty_indx = np.load('./2_Training data/{}/noempty_indx_{}{}.npy'.format(df,df,i))
        output = np.load('./2_Training data/{}/out_data_{}.npy'.format(df,df))
        output_Y = output[nonempty_indx.reshape(-1),:,:,:]
        train_nums = int(output_Y.shape[0]*train_per)
        
        if len(x_test) == 0:
            x_test = input_X[train_nums:, :, :, :3]
            y_test = output_Y[train_nums:, :, :, :]
        else:
            x_test = np.append(x_test, input_X[train_nums:, :, :, :3], axis=0)
            y_test = np.append(y_test, output_Y[train_nums:, :, :, :], axis=0)
    
    # separating index between each dataset
    data_sep.append(len(x_test))

# Input speed maps
x_bin = (x_test.sum(axis=3) != 0)
y_map = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
x_map = x_bin*y_map
sim_data = [x_test, y_test]

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


# =============================================================================
# Training performance measures
# =============================================================================

fold_loc='./3_Training experiments/expt-24 (iso-ani-compare)/'
num_epochs = 300
iso_rmse_arr = np.zeros((4, num_epochs))
iso_val_rmse_arr = np.zeros((4, num_epochs))
ani_rmse_arr = np.zeros((4, num_epochs))
ani_val_rmse_arr = np.zeros((4, num_epochs))
for i in range(1,5):
    iso_rmse = np.load(fold_loc+'Expt24_isomodel_rmse-{}.npy'.format(i))
    iso_val_rmse = np.load(fold_loc+'Expt24_isomodel_val_rmse-{}.npy'.format(i))
    ani_rmse = np.load(fold_loc+'Expt24_anisomodel_rmse-{}.npy'.format(i))
    ani_val_rmse = np.load(fold_loc+'Expt24_anisomodel_val_rmse-{}.npy'.format(i))
    iso_rmse_arr[i-1,:] = iso_rmse
    iso_val_rmse_arr[i-1,:] = iso_val_rmse
    ani_rmse_arr[i-1,:] = ani_rmse
    ani_val_rmse_arr[i-1,:] = ani_val_rmse
    
plt.figure(figsize=(5,4))  
plt.plot(iso_rmse_arr.mean(axis=0)*100, lw=2, label='Iso train')
plt.plot(iso_val_rmse_arr.mean(axis=0)*100, lw=2, label='Iso test')
plt.plot(ani_rmse_arr.mean(axis=0)*100, lw=2, label='Aniso train')
plt.plot(ani_val_rmse_arr.mean(axis=0)*100, lw=2, label='Aniso test')
plt.xlabel('Training epochs', fontsize=12)
plt.ylabel('RMSE loss (kmph)', fontsize=12)
plt.legend(fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid()
plt.tight_layout()

print('RMSE of isotropic model:')
print(np.round((iso_val_rmse_arr.mean(axis=0)*100)[-1], 3))
print(np.round((iso_val_rmse_arr.std(axis=0)*100)[-1], 3))
print('RMSE of anisotropic model:')
print(np.round((ani_val_rmse_arr.mean(axis=0)*100)[-1], 3))
print(np.round((ani_val_rmse_arr.std(axis=0)*100)[-1], 3))


iso_rmse_arr_mean = iso_rmse_arr.mean(axis=0)
ani_rmse_arr_mean = ani_rmse_arr.mean(axis=0)

iso_first_der = [iso_rmse_arr_mean[i]-iso_rmse_arr_mean[i+1] for i in range(len(iso_rmse_arr_mean)-1)]
iso_second_der = [iso_first_der[i]-iso_first_der[i+1] for i in range(len(iso_first_der)-1)]
ani_first_der = [ani_rmse_arr_mean[i]-ani_rmse_arr_mean[i+1] for i in range(len(ani_rmse_arr_mean)-1)]
ani_second_der = [ani_first_der[i]-ani_first_der[i+1] for i in range(len(ani_first_der)-1)]

plt.plot(iso_second_der)
plt.plot(ani_second_der)

# =============================================================================
# Data requirement analysis
# =============================================================================

def log_fit(X, Y, X_pred):
    opt_vars = np.polyfit(np.log(X), Y, 1)
    Y_pred = opt_vars[0]*np.log(X_pred)+opt_vars[1]
    return Y_pred, opt_vars


print('\n# ------ RMSE v/s Data requirements ------- #')
print('\n\tAnisotropic Model...')
x_aniso = []; y_aniso = []; ofmeas_aniso = []
for m in [1,2,4,5,6,7]:
    x_aniso_i = []; y_aniso_i = []; ofmeas_aniso_i = []
    fold_loc='./3_Training experiments/expt-28 (datareq)/try-{}/'.format(m)
    for i in [1,2,3,4,5,6,7,8,9,10]:
        test_rmse = np.load(fold_loc+'Expt28_anisomodel_testevals-{}.npy'.format(i))
        train_rmse = np.load(fold_loc+'Expt28_anisomodel_trainevals-{}.npy'.format(i))
        print(f'\tRMSE is {test_rmse.mean()*100:0.03f} +- {test_rmse.std()*100:0.03f}; # Samples is {train_rmse.shape[0]}')
        test_rmse = np.sort(test_rmse)
        t = int(len(test_rmse)*0.95)
        t = len(test_rmse)
        of = (test_rmse[:t].mean() - train_rmse[:t].mean())/test_rmse[:t].mean()*100
        ofmeas_aniso_i.append(of)
        x_aniso_i.append(test_rmse[:t].mean()*100)
        y_aniso_i.append(train_rmse.shape[0])
    x_aniso.append(x_aniso_i)
    y_aniso.append(y_aniso_i)
    ofmeas_aniso.append(ofmeas_aniso_i)

print('\n\tIsotropic Model...')
x_iso = []; y_iso = []; ofmeas_iso = []
for m in [1,2,4,5,6,7]:
    x_iso_i = []; y_iso_i = []; ofmeas_iso_i = []
    fold_loc='./3_Training experiments/expt-28 (datareq)/try-{}/'.format(m)
    for i in [1,2,3,4,5,6,7,8,9,10]:
        test_rmse = np.load(fold_loc+'Expt28_isomodel_testevals-{}.npy'.format(i))
        train_rmse = np.load(fold_loc+'Expt28_isomodel_trainevals-{}.npy'.format(i))
        print(f'\tRMSE is {test_rmse.mean()*100:0.03f} +- {test_rmse.std()*100:0.03f}; # Samples is {train_rmse.shape[0]}')
        test_rmse = np.sort(test_rmse)
        t = int(len(test_rmse)*0.95)
        t = len(test_rmse)
        of = (test_rmse[:t].mean() - train_rmse[:t].mean())/test_rmse[:t].mean()*100
        ofmeas_iso_i.append(of)
        x_iso_i.append(test_rmse[:t].mean()*100)
        y_iso_i.append(train_rmse.shape[0])
    x_iso.append(x_iso_i)
    y_iso.append(y_iso_i)
    ofmeas_iso.append(ofmeas_iso_i)

# some useful measures
x_iso = np.array(x_iso)
y_iso = np.array(y_iso)
x_aniso = np.array(x_aniso)
y_aniso = np.array(y_aniso)
ofmeas_iso = np.array(ofmeas_iso)
ofmeas_aniso = np.array(ofmeas_aniso)
ir = np.argsort(y_iso.mean(axis=0))

# visualize data v/s rmse
plt.figure()
plt.plot(y_iso.mean(axis=0)[ir], x_iso.mean(axis=0)[ir], '--bo', label='iso model')
plt.plot(y_aniso.mean(axis=0)[ir], x_aniso.mean(axis=0)[ir], '--r^', label='aniso model')
plt.xlabel('Size of training data', fontsize=12)
plt.ylabel('Test RMSE (km/hr)', fontsize=12)
plt.legend(fontsize=10)
plt.grid()

# visualize data v/s over-fitting measure
plt.figure()
plt.plot(y_iso.mean(axis=0)[ir], ofmeas_iso.mean(axis=0)[ir], '--bo',label='iso model')
plt.plot(y_aniso.mean(axis=0)[ir], ofmeas_aniso.mean(axis=0)[ir], '--r^', label='aniso model')
plt.xlabel('Size of training data', fontsize=12)
plt.ylabel('Over-fitting measure (%)', fontsize=12)
plt.legend(fontsize=10)
plt.grid()

# visualize logarithmic trend
X1 = y_iso.mean(axis=0)[ir]
Y1 = x_iso.mean(axis=0)[ir]
X2 = y_aniso.mean(axis=0)[ir]
Y2 = x_aniso.mean(axis=0)[ir]
X3 = y_iso.mean(axis=0)[ir]
Y3 = ofmeas_iso.mean(axis=0)[ir]
X4 = y_aniso.mean(axis=0)[ir]
Y4 = ofmeas_aniso.mean(axis=0)[ir]
X1_test = np.arange(1000,91000)
    
Y1_test,_ = log_fit(X1, Y1, X1_test)
Y2_test,_ = log_fit(X2, Y2, X1_test)
Y3_test,_ = log_fit(X3, Y3, X1_test)
Y4_test,_ = log_fit(X4, Y4, X1_test)
    
plt.figure()
plt.scatter(X1, Y1, s=25, marker='o', c='r')
plt.scatter(X2, Y2, s=25, marker='^', c='b')
plt.plot(X1_test, Y1_test, 'r', label='iso model')
plt.plot(X1_test, Y2_test, 'b', label='aniso model')
plt.xlabel('Size of training data', fontsize=12)
plt.ylabel('Test RMSE (km/hr)', fontsize=12)
plt.legend(fontsize=10)
plt.grid()

plt.figure()
plt.scatter(X3, Y3, s=25, marker='o', c='r')
plt.scatter(X4, Y4, s=25, marker='^', c='b')
plt.plot(X1_test, Y3_test, 'r', label='iso model')
plt.plot(X1_test, Y4_test, 'b', label='aniso model')
plt.xlabel('Size of training data', fontsize=12)
plt.ylabel('Over-fitting measure (%)', fontsize=12)
plt.legend(fontsize=10)
plt.grid()


# visualize (for papaer)
fig, ax = plt.subplots(figsize=(4,4.5))
ax.scatter(X3/1e5, Y3, c='tab:blue', s=50, marker='o', label='Isotropic model')
ax.scatter(X4/1e5, Y4, c='tab:orange', s=50, marker='^', label='Anisotropic model')
ax.plot(X1_test/1e5, Y3_test, c='tab:blue', ls='-', lw=0.8)
ax.plot(X1_test/1e5, Y4_test, c='tab:orange', ls='-', lw=0.8)
# ax.text(17, 420, '$y=x^2$', c='tab:blue', fontsize=12)
# ax.text(16, 140, '$y=5.72x - 15.42$', c='tab:orange', fontsize=12)
ax.set_xticks([0.2,0.4,0.6,0.8])
ax.set_xlim(-0.02,1)
ax.set_xlabel('Training data (x 1e5)', fontsize=19)
ax.set_ylabel('Over-fitting measure (%)', fontsize=19)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.legend(fontsize=16)
ax.grid()
fig.tight_layout()
fig.savefig('ModelComp_overfit.pdf', bbox_inches='tight')


# =============================================================================
# Model complexity
# =============================================================================


def mask_def1(slope, width, ker_size=7):
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
    elif k1==11:
        ff_mask[4,5]=1; ff_mask[6,5]=1;
    elif k1==13:
        ff_mask[5,6]=1; ff_mask[7,6]=1;
    elif k1==15:
        ff_mask[6,5]=1; ff_mask[8,7]=1;
    elif k1==17:
        ff_mask[7,6]=1; ff_mask[9,8]=1;
    elif k1==19:
        ff_mask[8,7]=1; ff_mask[10,9]=1;
    elif k1==21:
        ff_mask[9,8]=1; ff_mask[11,10]=1;
    
    com_mask = np.zeros((max(k1, k2), max(k1, k2)))
    com_mask = np.maximum(com_mask, ff_mask)
    com_mask = np.maximum(com_mask, cg_mask)
    
    ker_mask = np.zeros((max(k1,k2),max(k1,k2),inp_dep,out_dep), np.int32)
    for i in range(ker_mask.shape[2]):
        for j in range(ker_mask.shape[3]):
            ker_mask[:,:,i,j] = com_mask
    
    return ker_mask

# ---------- Kernel Size v/s Model Params ----------- #

# calculate number of parameters
iso_params = []
aniso_params = []
ker_size = [3,5,7,9,11,13,15,17,19,21]
max_v = 30; max_w = -5.3; min_v = 9; w_w = 1.2
for k in ker_size:
    a = create_mask(max_v,min_v,max_w,w_w,k,k,1,1,mask_def2).sum()
    b = k**2
    aniso_params.append(a)
    iso_params.append(b)

# visualize
# xlabs = ['3x3','5x5','7x7','9x9','11x11','13x13','15x15','17x17','19x19','21x21'] 
xlabs = [3,5,7,9,11,13,15,17,19,21]
x_range = np.arange(3,24)
iso_y1 = np.power(x_range,2)
aniso_y2 = 5.7515*x_range - 15.42
fig, ax = plt.subplots(figsize=(4,3.8))
ax.scatter(xlabs, iso_params, c='tab:blue', s=50, marker='o', label='Isotropic kernel')
ax.scatter(xlabs, aniso_params, c='tab:orange', s=50, marker='^', label='Anisotropic kernel')
# ax.set_xticklabels(xlabs)
ax.plot(x_range, iso_y1, c='tab:blue', ls='-', lw=0.7)
ax.plot(x_range, aniso_y2, c='tab:orange', ls='-', lw=0.7)
ax.text(11, 350, '$p$ $\sim$ $o(w^2)$', c='tab:blue', fontsize=13)
ax.text(17, 150, '$p$ $\sim$ $o(w)$', c='tab:orange', fontsize=13)
ax.set_xticks([0,8,16,24])
ax.set_xlabel('Widths of CNN kernel $w$', fontsize=16)
ax.set_ylabel('No. of parameters $p$', fontsize=16)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.legend(fontsize=13)
ax.grid()
fig.tight_layout()
fig.savefig('ModelComp_Kersize.pdf', bbox_inches='tight')


# ------------ Problem size effect: Road length ----------- #

# values
road_lens = np.array([200, 300, 400, 500, 600, 700, 800])
modelparams = np.array([[114897,133737,146657,170121,241633,314153,507513],
                        [59281,69737,86881,98761,138913,164217,232185]])
modelparams = modelparams/1e5

from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(road_lens.reshape(-1,1))
poly.fit(X_poly, modelparams[0,:])
lin2 = LinearRegression()
lin2.fit(X_poly, modelparams[0,:])
y = lin2.predict(poly.fit_transform(x_range.reshape(-1,1)))

# trends
x_range = np.arange(200,850)
# iso_y1 = 2E-05*np.power(x_range,2) - 0.0092*x_range + 2.58
aniso_y2 = 0.0027*x_range - 0.1424
iso_y1 = lin2.predict(poly.fit_transform(x_range.reshape(-1,1)))

# visualize
fig, ax = plt.subplots(figsize=(4,4.5))
ax.scatter(road_lens, modelparams[0], c='tab:blue', marker='o', s=50, label='Isotropic model')
ax.scatter(road_lens, modelparams[1], c='tab:orange', marker='^', s=50, label='Anisotropic model')
ax.plot(x_range, iso_y1, c='tab:blue', ls='-', lw=0.8)
ax.plot(x_range, aniso_y2, c='tab:orange', ls='-', lw=0.8)
ax.text(480, 3.5, '$p$ $\sim$ $o(X^2)$', c='tab:blue', fontsize=16)
ax.text(680, 1.2, '$p$ $\sim$ $o(X)$', c='tab:orange', fontsize=16)
# ax.set_xticks([0,200,400,30])
ax.set_xlabel('Road length $X$ (m)', fontsize=19)
ax.set_ylabel('No.of parameters $p$ (x 1e5)', fontsize=19)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.legend(fontsize=16)
ax.grid()
fig.tight_layout()
fig.savefig('ModelComp_roadlen.pdf', bbox_inches='tight')


# =============================================================================
# Some test reconstruction results
# =============================================================================

'''
Selected samples:
    Congested: 2379, 2269, 2855, 2152, 2935, 2588, 3142
    Slow-down: 2046, 1568, 1955, 
    Free-flow: 709, 871, 204, 710
'''
    
h=80; w=60; n=5
samples = np.arange(0, len(x_test), 1)
random_samples = np.random.choice(samples, n)
random_samples = np.array([2152, 2935, 3142, 1568, 204])

plt.figure(figsize=(7,6))
for i, j in enumerate(random_samples):
    # display true speed field (true output)
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(y_test[j].reshape(h,w), cmap='jet_r', vmin=0, vmax=1)
    if i == 2:
        ax.set_title('True speeds', fontsize=12)
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_xticks([]); ax.set_yticks([])
    
    # display probe trajectory (input)
    ax = plt.subplot(3, n, i  + n + 1)
    plt.imshow(x_test[j].reshape(h,w,3), cmap='jet_r', vmin=0, vmax=1)
    if i == 2:
        ax.set_title('Input trajectory', fontsize=12)
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_xticks([]); ax.set_yticks([])
    
    # display anisotropic kernel reconstruction
    ax = plt.subplot(3, n, i + 2*n + 1)
    plt.imshow(ani_predY[j].reshape(h,w), cmap='jet_r', vmin=0, vmax=1)
    if i == 2:
        ax.set_title('Predicted speeds', fontsize=12)
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_xticks([]); ax.set_yticks([])

# plt.tight_layout()
plt.show()

# Speed profile
samples=np.arange(0,len(x_test),1)
j = np.random.choice(samples)
j = 710
plt.figure()
for k in range(60):
    plt.plot(y_test[j][:,k]*max_speed, label='True')
    plt.plot(iso_predY[j][:,k]*max_speed, label='Iso pred')
    plt.plot(ani_predY[j][:,k]*max_speed, label='Ani pred')
    inp_speed_prof = x_map[j][:,k]*max_speed
    inp_speed_loc = np.nonzero(inp_speed_prof)[0]
    plt.vlines(inp_speed_loc, np.zeros(len(inp_speed_loc)), 
               inp_speed_prof[inp_speed_loc], 
               label='input')
    plt.xlabel('Section (x10 m)', fontsize=14)
    plt.ylabel('Speed (kmph)', fontsize=14)
    plt.legend()
    plt.ylim([0,110])
    plt.grid()
    plt.pause(0.2)
    plt.clf()


# =============================================================================
# Reconstruction result with speed profile
# =============================================================================

h=80; w=60; n=5
random_samples = np.array([2152, 2935, 3142, 1568, 204])

for i, sam in enumerate(random_samples):

    fig_a = plt.figure(figsize=(8.2, 2.9))
    gs1 = gridspec.GridSpec(1, 4, wspace=0.3, hspace=0.1)
    
    # display true speed field (true output)
    ax = fig_a.add_subplot(gs1[0, 0])
    ax.imshow(y_test[sam].reshape(h,w), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
    if i == 0:
        ax.set_title('True', fontsize=15)
    ax.set_ylabel('Space (m)', fontsize=14)
    ytick_labels = ['1000','800','600','400','200','0']
    ax.set_yticklabels(ytick_labels, fontsize=13)
    # if i == 4:
    ax.set_xticks([0, 30, 60])
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.tick_params(labelsize=13)
    # else:
        # ax.set_xticks([])
        # ax.set_xlabel('Time', fontsize=14);
    
    # display probe trajectory (input)
    ax = fig_a.add_subplot(gs1[0, 1])
    ax.imshow(x_test[sam].reshape(h,w,3), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
    if i == 0:
        ax.set_title('Input', fontsize=15)
    ax.set_xlabel('Time (s)', fontsize=14); ax.set_ylabel('Space', fontsize=14)
    ax.set_yticks([])
    # for tim_k in [10, 30, 50]:
    #     ax.axvline(x=tim_k, ymin=-0.1, ymax=1.1, label='line at x')
    # if i == 4:
    ax.set_xticks([0, 30, 60])
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.tick_params(labelsize=13)
    # else:
        # ax.set_xticks([])
        # ax.set_xlabel('Time', fontsize=14);
            
    # display anisotropic kernel reconstruction
    ax = fig_a.add_subplot(gs1[0, 2])
    ax.imshow(ani_predY[sam].reshape(h,w), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
    if i == 0:
        ax.set_title('Predicted', fontsize=15)
    ax.set_xlabel('Time (s)', fontsize=14); ax.set_ylabel('Space', fontsize=14)
    ax.set_yticks([])
    # if i == 4:
    ax.set_xticks([0, 30, 60])
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.tick_params(labelsize=13)
    # else:
        # ax.set_xticks([])
        # ax.set_xlabel('Time', fontsize=14);
    
    # add separating line
    # ax.plot([-1, -1], [0, 1], color='black', lw=1, transform=ax.transAxes, clip_on=False)
    
    # display speed profiles
    gs2 = gridspec.GridSpecFromSubplotSpec(3, 1,
            subplot_spec=gs1[0, 3], wspace=0.15, hspace=0.3)
    for k, tim_k in enumerate([10, 30, 50]):
        
        ax = fig_a.add_subplot(gs2[k])
        if (i == 0) and (k == 0):
            ax.set_title('Speed profile', fontsize=15)
        # Plot speed profile
        ax.plot(y_test[sam][:,tim_k]*max_speed, label='True')
        ax.plot(ani_predY[sam][:,tim_k]*max_speed, label='Ani pred')
        # Plot input probe
        inp_speed_prof = x_map[sam][:,tim_k]*max_speed
        inp_speed_loc = np.nonzero(inp_speed_prof)[0]
        plt.vlines(inp_speed_loc, np.zeros(len(inp_speed_loc)), 
                   inp_speed_prof[inp_speed_loc], linestyles='dashed', label='t={}s'.format(tim_k))
        # Format axes
        ax.set_ylim([0, 100])
        ax.set_xticks([])
        # ax.set_xlabel('$X$')
        ax.invert_xaxis()
        if k == 2:
            ax.set_xticks([0, 40, 80])
            ax.set_xlabel('Space (m)', fontsize=14)
            ax.set_xticklabels(['800','400','0'])
        ax.yaxis.set_tick_params(labelsize=11)
        ax.xaxis.set_tick_params(labelsize=13)
        ax.grid(True)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        # if k == 0:
        ytick_labels = [item.get_text() for item in ax.get_yticklabels()]
        ytick_labels = ['0', '50', '100 kmph']
        ax.set_yticklabels(ytick_labels, fontsize=13)
        if i == 0:
            txt_y_pos = 0.94
            txt_y_loc = 'top'
            ax.text(0.55, txt_y_pos, 't={}s'.format(tim_k), 
                    horizontalalignment='center', transform=ax.transAxes,
                    verticalalignment=txt_y_loc,fontsize=11)
        
    plt.savefig('Aniso_recons-{}.pdf'.format(i+1), bbox_inches='tight')

# =============================================================================
# Blah blah
# =============================================================================

sam = 3142  #3142

fig_a = plt.figure(figsize=(7, 5))
gs1 = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.1)

# display true speed field (true output)
ax = fig_a.add_subplot(gs1[0, 0])
ax.imshow(y_test[sam].reshape(h,w)*100, cmap='jet_r', vmin=0, vmax=80, aspect='auto')
ax.set_title('(a) True', fontsize=17)
ax.set_ylabel('Space (m)', fontsize=15)
ax.set_yticks([0, 20, 40, 60, 80])
ytick_labels = ['800','600','400','200','0']
ax.set_yticklabels(ytick_labels, fontsize=14)
ax.set_xticks([0, 30, 60])
ax.set_xlabel('Time (s)', fontsize=15)
ax.tick_params(labelsize=15)

# display anisotropic kernel reconstruction
ax = fig_a.add_subplot(gs1[0, 1])
map1 = ax.imshow(ani_predY[sam].reshape(h,w)*100, cmap='jet_r', vmin=0, vmax=80, aspect='auto')
ax.set_title('(b) Estimated', fontsize=17)
# ax.set_ylabel('Space', fontsize=14)
ax.set_yticks([])
ax.set_xticks([0, 30, 60])
ax.set_xlabel('Time (s)', fontsize=15)
ax.tick_params(labelsize=14)
cbar_ax = fig_a.add_axes([0.93, 0.50, 0.02, 0.35])
cbar = fig_a.colorbar(map1, cax=cbar_ax)
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_ylabel('Speed (kmph)', fontsize=16)

plt.savefig('transition.png', bbox_inches='tight')



y1 = np.flip(y_test[sam][:,30,0]*100, -1)
y2 = np.flip(ani_predY[sam][:,30,0]*100, -1)

fig_a = plt.figure(figsize=(4, 3))
gs1 = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.1)

ax = fig_a.add_subplot(gs1[0, 0])
ax.set_title('Speed profile at t = 30 s', fontsize=17)
ax.plot(np.arange(0,80), y1, label='True')
ax.plot(np.arange(0,80), y2, label='Estimate')
ax.set_xticks([0, 20, 40, 60, 80])
xtick_labels = ['0','200','400','600','800']
ax.set_xticklabels(xtick_labels, fontsize=15)
ax.set_ylim([0, 80])
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_ylabel('Speed (kmph)', fontsize=16)
ax.set_xlabel('Space (m)', fontsize=16)

ax.tick_params(labelsize=15)
ax.legend(fontsize=14)

plt.savefig('speedprof.png', bbox_inches='tight')




# =============================================================================
# Error analysis
# =============================================================================

max_speed=100; h_act=80
iso_rmse = []; ani_rmse = []
for s in range(iso_predY.shape[0]):
    true_y = y_test[s,:h_act,:,:].reshape(-1)
    pred_y1 = iso_predY[s,:h_act,:,:].reshape(-1)
    pred_y2 = ani_predY[s,:h_act,:,:].reshape(-1)
    rmse1 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y1)*max_speed, 2))), 3)
    rmse2 = np.round(np.sqrt(np.mean(np.power((true_y-pred_y2)*max_speed, 2))), 3)
    iso_rmse.append(rmse1)
    ani_rmse.append(rmse2)

print('RMSE of isotropic model:')
print('\t', np.mean(iso_rmse))
print('\t', np.std(iso_rmse))
print('RMSE of anisotropic model:')
print('\t', np.mean(ani_rmse))
print('\t', np.std(ani_rmse))

print('RMSE of isotropic model')
print('\t Free-flow: %0.3f (+/-%0.3f) ' % (np.mean(iso_rmse[:data_sep[0]]), np.std(iso_rmse[:data_sep[0]])))
print('\t Slowed-down: %0.3f (+/-%0.3f) ' % (np.mean(iso_rmse[data_sep[0]:data_sep[1]]), np.std(iso_rmse[data_sep[0]:data_sep[1]])))
print('\t Congested: %0.3f (+/-%0.3f) ' % (np.mean(iso_rmse[data_sep[1]:]), np.std(iso_rmse[data_sep[1]:])))

print('RMSE of anisotropic model')
print('\t Free-flow: %0.3f (+/-%0.3f) ' % (np.mean(ani_rmse[:data_sep[0]]), np.std(ani_rmse[:data_sep[0]])))
print('\t Slowed-down: %0.3f (+/-%0.3f) ' % (np.mean(ani_rmse[data_sep[0]:data_sep[1]]), np.std(ani_rmse[data_sep[0]:data_sep[1]])))
print('\t Congested: %0.3f (+/-%0.3f) ' % (np.mean(ani_rmse[data_sep[1]:]), np.std(ani_rmse[data_sep[1]:])))


# =============================================================================
# Anisotropic v/s isotropic reconstruction samples
# =============================================================================

'''
Congested: 2993, 2794, 2961, 2193
Slowed-down: 
Free-flow: 1077

'''

h=80; w=60; n=5
samples = np.arange(0, len(x_test), 1)
# random_samples = np.random.choice(samples, n)
random_samples = np.array([2993, 2794, 2961, 2193, 1077])

plt.figure(figsize=(7,6))
for i, j in enumerate(random_samples):
    # display true speed field (true output)
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(y_test[j].reshape(h,w), cmap='jet_r', vmin=0, vmax=1)
    if i == 2:
        ax.set_title('True', fontsize=12)
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_xticks([]); ax.set_yticks([])
    
    # display probe trajectory (input)
    ax = plt.subplot(4, n, i  + n + 1)
    plt.imshow(x_test[j].reshape(h,w,3), cmap='jet_r', vmin=0, vmax=1)
    if i == 2:
        ax.set_title('Input', fontsize=12)
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_xticks([]); ax.set_yticks([])
    
    # display anisotropic kernel reconstruction
    ax = plt.subplot(4, n, i + 2*n + 1)
    plt.imshow(iso_predY[j].reshape(h,w), cmap='jet_r', vmin=0, vmax=1)
    if i == 2:
        ax.set_title('Isotropic', fontsize=12)
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_xticks([]); ax.set_yticks([])
    
    # display anisotropic kernel reconstruction
    ax = plt.subplot(4, n, i + 3*n + 1)
    plt.imshow(ani_predY[j].reshape(h,w), cmap='jet_r', vmin=0, vmax=1)
    if i == 2:
        ax.set_title('Anisotropic', fontsize=12)
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_xticks([]); ax.set_yticks([])
plt.show()

# =============================================================================
# Iso v/s Aniso comparison for paper
# =============================================================================

h=80; w=60
random_samples = np.array([2993, 2193, 1077])

for i, sam in enumerate(random_samples):

    fig_a = plt.figure(figsize=(5, 2.4))
    gs1 = gridspec.GridSpec(1, 3, wspace=0.3, hspace=0.1)
    
    # display true speed field (true output)
    ax = fig_a.add_subplot(gs1[0, 0])
    ax.imshow(y_test[sam].reshape(h,w), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
    x_space = np.nonzero(x_map[sam,:,:])[0]
    y_time = np.nonzero(x_map[sam,:,:])[1]
    if i == 2:
        ax.scatter(y_time, x_space, s=4, c='k', marker='o', alpha=0.8)
    else:
        ax.scatter(y_time, x_space, s=0.5, c='k', marker='o', alpha=0.8)
    ax.set_ylabel('Space (m)', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=14)
    ytick_labels = ['1000','800','600','400','200','0']
    ax.set_yticklabels(ytick_labels)
    if i == 0:
        ax.set_title('(i) True', fontsize=14)
    # if i == 2:
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel('Time (s)', fontsize=14)
    # else:
        # ax.set_xticks([])
        # ax.set_xlabel('Time', fontsize=12)
    
    # display isotropic kernel reconstruction
    ax = fig_a.add_subplot(gs1[0, 1])
    ax.imshow(iso_predY[sam].reshape(h,w), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
    ax.set_yticks([])
    ax.set_xlabel('Time (s)', fontsize=14)
    if i == 0:
        ax.set_title('(ii) Isotropic', fontsize=14)
        ax.set_xlim([0, 60])
    # if i == 2:
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel('Time (s)', fontsize=14)
    # else:
        # ax.set_xticks([])
        # ax.set_xlabel('$Time$', fontsize=14)

    # display anisotropic kernel reconstruction
    ax = fig_a.add_subplot(gs1[0, 2])
    ax.imshow(ani_predY[sam].reshape(h,w), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
    ax.set_yticks([])
    ax.set_xlabel('Time (s)', fontsize=14)
    if i == 0:
        ax.set_title('(iii) Anisotropic', fontsize=14)
    # if i == 2:
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel('Time (s)', fontsize=14)
    # else:
        # ax.set_xticks([])
        # ax.set_xlabel('$Time$', fontsize=14)
        
    # plt.tight_layout()
    plt.savefig('Aniso_iso_comp-{}.pdf'.format(i+1), bbox_inches='tight')


# =============================================================================
# Eulerian to lagrangian trajectory analysis
# =============================================================================

def genTraj(d_0, t_0, s_0, pred_map):
    
    tims = []; tims.append(t_0)
    locs = []; locs.append(d_0)
    spds = []; spds.append(s_0)
    
    d_1 = 0
    while d_1 < 800:
        
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
sam_num = 2193
true_map = y_test[sam_num].reshape(80,60)*100
iso_map = iso_predY[sam_num].reshape(80,60)*100
ani_map = ani_predY[sam_num].reshape(80,60)*100
init_cond = true_map[-2, :]
bound_cond = true_map[:, 1]
init_veh_cond = [(3*(i), 15, init_cond[::3][i]) for i in range(init_cond[::3].shape[0])]
bound_veh_cond = [(1, 20*i+5, init_cond[::2][i]) for i in range(init_cond[::2].shape[0])]

# Predict the output vehicle trajectories (lagrangian coords)
pred_iso_traj = {}
pred_ani_traj = {}; v_id = 0
for v_cond in init_veh_cond:
    t_0 = v_cond[0]
    d_0 = v_cond[1]
    s_0 = v_cond[2]
    iso_traj = genTraj(d_0, t_0, s_0, iso_map)
    ani_traj = genTraj(d_0, t_0, s_0, ani_map)
    pred_iso_traj[v_id] = iso_traj
    pred_ani_traj[v_id] = ani_traj
    v_id += 1
for v_cond in bound_veh_cond:
    t_0 = v_cond[0]
    d_0 = v_cond[1]
    s_0 = v_cond[2]
    iso_traj = genTraj(d_0, t_0, s_0, iso_map)
    ani_traj = genTraj(d_0, t_0, s_0, ani_map)
    pred_iso_traj[v_id] = iso_traj
    pred_ani_traj[v_id] = ani_traj
    v_id += 1


# Compare predicted and true trajectories
plt.figure()
for v_id in pred_iso_traj.keys():
    t = pred_iso_traj[v_id][0]
    d = pred_iso_traj[v_id][1]
    s = pred_iso_traj[v_id][2]
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=100)
plt.ylim([0,800])
plt.colorbar()
plt.grid()
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Space (m)', fontsize=12)
plt.title('Isotropic trajectories', fontsize=13)

plt.figure()
for v_id in pred_ani_traj.keys():
    t = pred_ani_traj[v_id][0]
    d = pred_ani_traj[v_id][1]
    s = pred_ani_traj[v_id][2]
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=100)
    plt.plot(t, d, lw=0.2, c='k')
plt.ylim([0,800])
plt.colorbar()
plt.grid()
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Space (m)', fontsize=12)
plt.title('Anisotropic trajectories', fontsize=13)


plt.figure()
plt.imshow(true_map, cmap='jet_r', vmin=0, vmax=100)

# =============================================================================
# For TRB Poster
# =============================================================================

sam=2193
i=0

fig_a = plt.figure(figsize=(7.5, 3.5))
gs1 = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.1)

# display true speed field (true output)
ax = fig_a.add_subplot(gs1[0, 0])
ax.imshow(y_test[sam].reshape(h,w), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
x_space = np.nonzero(x_map[sam,:,:])[0]
y_time = np.nonzero(x_map[sam,:,:])[1]
ax.scatter(y_time, x_space, s=3, c='k', marker='x', alpha=0.5)
ax.set_ylabel('Space ($x 10$ $m$)', fontsize=16)
ax.set_xlabel('$t$ ($secs$)', fontsize=16)
ax.set_title('True', fontsize=17)
ax.set_xticks([0, 20, 40, 60])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# display isotropic kernel reconstruction
ax = fig_a.add_subplot(gs1[0, 1])
ax.imshow(iso_predY[sam].reshape(h,w), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
ax.set_yticks([])
ax.set_xlabel('$t$ ($secs$)', fontsize=16)
ax.set_title('Isotropic', fontsize=17)
ax.set_xlim([0, 60])
ax.set_xticks([0, 20, 40, 60])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# display anisotropic kernel reconstruction
ax = fig_a.add_subplot(gs1[0, 2])
ax.imshow(ani_predY[sam].reshape(h,w), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
ax.set_yticks([])
ax.set_xlabel('$t$ ($secs$)', fontsize=16)
ax.set_title('Anisotropic', fontsize=17)
ax.set_xticks([0, 20, 40, 60])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
    
plt.savefig('Aniso_iso_comp-TRB-high res.png', dpi=1200, bbox_inches='tight')

