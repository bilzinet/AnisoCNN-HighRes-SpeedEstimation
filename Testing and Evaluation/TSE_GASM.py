"""
Created on Tue Apr 16 2019
@author: btt1

Traffic State Estimation Comparison method using GASM anisotropic filter
GASM: Generalized adaptive smooting filter for traffic state estimation [Trieber et al. 2009]
https://arxiv.org/pdf/0909.4467.pdf

"""

# load packages
import numpy as np
import matplotlib.pyplot as plt

# some settings
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
np.set_printoptions(precision=4, suppress=True)


# =============================================================================
# GASM functions
# =============================================================================

def exponential_kernel(x, t, lam, tau):
#    z = np.abs(x)/lam + np.abs(t)/tou
    z = np.power(x,2)/(2*lam**2) + np.power(t,2)/(2*tau**2)
    phi = np.exp(-z)
    return phi

def phi_kernel(x_j, x_i, t_j, t_i, c, lam, tau):
    x = x_j-x_i
    t = (t_j-t_i) - (x_j-x_i)/c
    phi_i = exponential_kernel(x, t, lam, tau)
    return phi_i

def filter_weights(v_free, v_cong, V_c, delta_v):
    v_star = min(v_cong, v_free)
    z = (V_c - v_star)/delta_v
    weight_w = 0.5*( 1 + np.tanh(z))
    return weight_w

def nonlinear_filter_phi_score(x_j, x_i, t_j, t_i, v_i, c_free, c_cong, V_c, delta_v, lam, tau):
    z_free = phi_kernel(x_j, x_i, t_j, t_i, c_free, lam, tau)
    z_cong = phi_kernel(x_j, x_i, t_j, t_i, c_cong, lam, tau)
    w_cf = filter_weights(z_free, z_cong, V_c, delta_v)
    z = w_cf*z_cong + (1-w_cf)*z_free
    return z

def GASM(x_j, t_j, probe_coords, c_free, c_cong, V_c, delta_v, lam, tau):
    N = 0; V = 0
    for i in range(probe_coords.shape[0]):
        t_i = probe_coords[i, 0]
        x_i = probe_coords[i, 1]
        v_i = probe_coords[i, 2]
        phi = nonlinear_filter_phi_score(x_j, x_i, t_j, t_i, v_i, c_free, c_cong, V_c, delta_v, lam, tau)
        N += phi
        V += phi*v_i
    V /= N
    return np.around(V,2)

def V_estm(X_probes, D):

    V_est = np.zeros((D.shape[0], D.shape[1]))
    for m in range(D.shape[0]):
        print(m)
        for n in range(D.shape[1]):
            x = D[m, n, 0]
            t = D[m, n, 1]
            v = GASM(x, t, X_probes, c_free, c_cong, V_c, delta_v, lam, tau)
            V_est[m,n] = v

    return V_est

def V_estm_vectorized(X_probes, D, N, M):
    
    # space-time coordinates and probe coordinates
    T_i = D[:,:,0].reshape(-1,1)
    X_i = D[:,:,1].reshape(-1,1)
    T_p = X_probes[:,0]
    X_p = X_probes[:,1]
    V_p = X_probes[:,2]
    
    # calculate free-flow condition
    X_tilt_cong = (X_i - X_p)
    T_tilt_cong = (T_i - T_p)
    # Z_cong = np.power(X_tilt_cong, 2)/(2*lam**2) + np.power(T_tilt_cong, 2)/(2*tau**2)
    Z_cong = np.abs(X_tilt_cong)/lam + np.abs(T_tilt_cong)/tau
    phi_cong = np.exp(-Z_cong)
    V_cong = np.divide(np.dot(phi_cong, V_p), phi_cong.sum(axis=1))
    
    # calculate congested condition
    X_tilt_free = X_i - X_p
    T_tilt_free = (T_i - T_p)-(X_i - X_p)/c_free
    Z_free = np.power(X_tilt_free, 2)/(2*lam**2) + np.power(T_tilt_free, 2)/(2*tau**2)
    phi_free = np.exp(-Z_free)
    V_free = np.divide(np.dot(phi_free, V_p), phi_free.sum(axis=1))
    
    # weigh free-flow and congestion
    V_thr = np.repeat(V_c, T_i.shape[0]).reshape(-1,1)
    V_star_cand = np.concatenate((V_cong.reshape(-1,1),(V_free.reshape(-1,1))), axis=1)
    V_star = np.min(V_star_cand, axis=1).reshape(-1,1)
    Z = np.divide(V_thr - V_star, delta_v)
    W = 0.5*(1 + np.tanh(Z))
    
    V_est = W.reshape(-1)*V_cong.reshape(-1) + (1-W).reshape(-1)*V_free.reshape(-1)
    V_est = V_est.reshape(M, N)
    
    return V_est


# =============================================================================
# Configuration
# =============================================================================

# road params
v_max = 95
k_max = 150

# discretization params
x_len = 670
t_start = 600
t_end = 900
t_len = t_end - t_start
delx = 10 / 1000
delt = 1 / 3600
cfl_limit = v_max*(delt/delx)
M = np.floor((x_len/1000) / delx).astype(np.int)        # number of space discretizations
N = np.floor((t_len/3600) / delt).astype(np.int)        # number of time discretizations

# GASM params
c_free = 95                     # free-flow speed in km/hr
c_cong = -18.5                    # backward wave speed in km/hr
V_c = 40                        # congestion threshold speed in km/hr
delta_v = 5                     # threshold width in km/hr
lam = np.round(20/1000,3)      # deviation in x in km (20)
tau = np.round(5/3600,3)       # deviation in t in hrs (5)


# =============================================================================
# Sample predictions
# =============================================================================

def load_data(df, hw1, hw2, c=3, h=80, w=60, h_act=67, w_act=1800, max_speed = 95):

    output_Y = np.load('./5_Testing data/{}/out_data_{}_{}.npy'.format(df,df,hw1))
    input_X = np.load('./5_Testing data/{}/inp_data_{}_{}.npy'.format(df,df,hw2))
    x_test = input_X[:, :h_act, :w_act, :]
    y_test = output_Y[:, :h_act, :w_act, :]
    x_bin = (x_test.sum(axis=3) != 0)
    y_map = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
    x_map = x_bin * y_map
    data = [x_test, y_test*max_speed, x_map*max_speed, x_bin.astype(np.int32)]
    
    return data

def gen_random_input(V_true, per):

    V_meas = V_true.copy()
    tot_cells = V_meas.reshape(-1).shape[0]
    sam_cells = int(tot_cells - tot_cells*per/100) 
    r = np.random.choice((np.arange(0,tot_cells)), size=sam_cells, replace=False)
    V_meas.reshape(-1)[r] = -1
    V_loc = (V_meas != -1).astype(np.int)
    
    return V_meas, V_loc


# # load data
# df = 'Ngsim'
# hw1 = 'us101_lane2'
# hw2 = 'us101_lane2_5p'
# max_speed = v_max
# data_ngsim = load_data(df, hw1, hw2)

# # true and input speed matrix
# sam = 43
# v_true = data_ngsim[1][0].squeeze()
# v_meas = data_ngsim[2][sam]
# v_measloc = data_ngsim[3][sam]
# # V_true = (v_true[0::3,:][:M,:]+v_true[1::3,:]+v_true[2::3,:])/3
# # V_loc = (v_measloc[0::3,:][:M,:]+v_measloc[1::3,:]+v_measloc[2::3,:])/3
# # V_loc = (V_loc != 0).astype(np.int32)
# # V_meas = np.multiply(V_true, V_loc)
# V_true = v_true
# V_meas = v_meas
# V_loc = v_measloc

# # random input matrix
# per = 5
# V_meas, V_loc = gen_random_input(V_true, per)

# # generate space-time coordinates
# t_space = np.arange(0, int(delt*3600)*N, int(delt*3600))
# x_space = np.arange(0, int(delx*1000)*M, int(delx*1000)) + (int(delx*1000)/2)
# T, X = np.meshgrid(t_space/3600, x_space/1000)
# X = np.expand_dims(X, -1)
# T = np.expand_dims(T, -1)
# D = np.concatenate((T,X), axis=-1)

# # generate input coordinates vector for GASM
# X_full = np.concatenate((D, np.expand_dims(V_meas, -1)), axis=-1)
# X_meas = np.multiply(X_full, V_loc.reshape(M, N, 1))
# X_ind = np.transpose(np.nonzero(X_meas.sum(axis=2)))
# X_probes = X_meas[X_ind[:,0], X_ind[:,1], :]
# print(X_probes.shape)

# # predict using GASM (single sample)
# V_est = V_estm_vectorized(X_probes, D, N, M)
# rmse = np.sqrt(np.nanmean(np.power(V_est-V_true, 2)))
# print(f'RMSE: {rmse:0.03f} km/hr')

# plt.figure()
# plt.imshow(V_true, cmap='jet_r', aspect='auto')
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(V_est, cmap='jet_r', aspect='auto')
# plt.colorbar()
# plt.show()



# =============================================================================
# Prediction on NGSIM data
# =============================================================================

def rmse_calc(data):
    
    errors = []
    est = []
    for sam in range(sams):
        # print('\tSample: ', sam)
        # load sample
        v_true = data[1][0].squeeze()
        v_meas = data[2][sam]
        v_measloc = data[3][sam]
        V_true = v_true.copy()
        V_meas = v_meas.copy()
        V_loc = v_measloc.copy()
        
        # generate space-time coordinates
        t_space = np.arange(0, int(delt*3600)*N, int(delt*3600))
        x_space = np.arange(0, int(delx*1000)*M, int(delx*1000)) + (int(delx*1000)/2)
        T, X = np.meshgrid(t_space/3600, x_space/1000)
        X = np.expand_dims(X, -1)
        T = np.expand_dims(T, -1)
        D = np.concatenate((T,X), axis=-1)
        
        # generate input coordinates vector for GASM
        X_full = np.concatenate((D, np.expand_dims(V_meas, -1)), axis=-1)
        X_meas = np.multiply(X_full, V_loc.reshape(M, N, 1))
        X_ind = np.transpose(np.nonzero(X_meas.sum(axis=2)))
        X_probes = X_meas[X_ind[:,0], X_ind[:,1], :]
        
        # estimate using GASM and predict error
        V_est = V_estm_vectorized(X_probes, D, N, M)
        rmse = np.sqrt(np.nanmean(np.power(V_est-V_true, 2)))
        errors.append(rmse)
        est.append(V_est)

    return errors, est


# parameters
h_act=67
w_sta=600
w_act=900
max_speed=100

# output data
df = 'Ngsim'
out_i = 'us101_lane2'
output_Y = np.load('./5_Testing data/{}/out_data_{}_{}.npy'.format(df,df,out_i))
y_test = output_Y[:, :h_act, w_sta:w_act, :]
y_map = y_test.copy().squeeze(axis=-1)

# input data
data_ngsim = []
for i in ['us101_lane2_3p','us101_lane2_5p','us101_lane2_10p']:
    input_X = np.load('./5_Testing data/{}/inp_data_{}_{}.npy'.format(df,df,i))
    x_test = input_X[:, :h_act, w_sta:w_act, :]
    x_bin = (x_test.sum(axis=3) != 0)
    x_map = x_bin * y_map
    data_ngsim.append([x_test, y_test*max_speed, 
                       x_map*max_speed, x_bin.astype(np.int32)])

# predict and calculate error
print('\nGASM Prediction Results...')
sams = 25
rmse_p = []
ests_p = []
for i in range(len(data_ngsim)):
    print('\n\tData: ', i)
    r_p, e_p = rmse_calc(data_ngsim[i])
    rmse_p.append((np.nanmean(r_p), np.nanstd(r_p)))
    ests_p.append(e_p)
    print(f'\tMean rmse: {np.nanmean(r_p):0.03f} +- {np.nanstd(r_p):0.03f}') 

# save results
np.save('./5_Testing data/Ngsim/GASM_preds.npy', np.array(ests_p))
np.save('./5_Testing data/Ngsim/GASM_preds_rmse.npy', np.array(rmse_p))

# x = data_ngsim[1][0]
# plt.figure()
# plt.imshow(x[0], cmap='jet_r', aspect='auto')
# plt.colorbar()
# plt.show()

# y = data_ngsim[1][1]
# plt.figure()
# plt.imshow(y[0].squeeze(), cmap='jet_r', aspect='auto', vmin=0, vmax=100)
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(ests_p[1][0], cmap='jet_r', aspect='auto')
# plt.colorbar()
# plt.show()

# =============================================================================
# Offline calibration of GASM Parameters
# =============================================================================

from scipy.optimize import curve_fit

def V_estm_vectorized_calbr(X_probes, c_free=95, c_cong=-18, lam=0.02, tau=0.001389):
    
    # params
    # c_free=95
    # c_cong=-18
    
    # space-time coordinates and probe coordinates
    T_i = D[:,:,0].reshape(-1,1)
    X_i = D[:,:,1].reshape(-1,1)
    T_p = X_probes[:,0]
    X_p = X_probes[:,1]
    V_p = X_probes[:,2]
    
    # calculate free-flow condition
    X_tilt_cong = (X_i - X_p)
    T_tilt_cong = (T_i - T_p)
    # Z_cong = np.power(X_tilt_cong, 2)/(2*lam**2) + np.power(T_tilt_cong, 2)/(2*tau**2)
    Z_cong = np.abs(X_tilt_cong)/lam + np.abs(T_tilt_cong)/tau
    phi_cong = np.exp(-Z_cong)
    V_cong = np.divide(np.dot(phi_cong, V_p), phi_cong.sum(axis=1))
    
    # calculate congested condition
    X_tilt_free = X_i - X_p
    T_tilt_free = (T_i - T_p)-(X_i - X_p)/c_free
    Z_free = np.power(X_tilt_free, 2)/(2*lam**2) + np.power(T_tilt_free, 2)/(2*tau**2)
    phi_free = np.exp(-Z_free)
    V_free = np.divide(np.dot(phi_free, V_p), phi_free.sum(axis=1))
    
    # weigh free-flow and congestion
    V_thr = np.repeat(V_c, T_i.shape[0]).reshape(-1,1)
    V_star_cand = np.concatenate((V_cong.reshape(-1,1),(V_free.reshape(-1,1))), axis=1)
    V_star = np.min(V_star_cand, axis=1).reshape(-1,1)
    Z = np.divide(V_thr - V_star, delta_v)
    W = 0.5*(1 + np.tanh(Z))
    
    V_est = W.reshape(-1)*V_cong.reshape(-1) + (1-W).reshape(-1)*V_free.reshape(-1)
    V_est = V_est.reshape(M, N)
    
    return V_est

def build_data(data, sam):
    
    # load sample
    v_true = data[1][0].squeeze()
    v_meas = data[2][sam]
    v_measloc = data[3][sam]
    V_true = v_true.copy()
    V_meas = v_meas.copy()
    V_loc = v_measloc.copy()
    
    # generate space-time coordinates
    t_space = np.arange(0, int(delt*3600)*N, int(delt*3600))
    x_space = np.arange(0, int(delx*1000)*M, int(delx*1000)) + (int(delx*1000)/2)
    T, X = np.meshgrid(t_space/3600, x_space/1000)
    X = np.expand_dims(X, -1)
    T = np.expand_dims(T, -1)
    D = np.concatenate((T,X), axis=-1)
    
    # generate input coordinates vector for GASM
    X_full = np.concatenate((D, np.expand_dims(V_meas, -1)), axis=-1)
    X_meas = np.multiply(X_full, V_loc.reshape(M, N, 1))
    X_ind = np.transpose(np.nonzero(X_meas.sum(axis=2)))
    X_probes = X_meas[X_ind[:,0], X_ind[:,1], :]

    return X_probes, D, N, M, V_true.reshape(-1)

def GASM_pred(X_probes, c_free, c_cong, lam, tau):
    V_est = V_estm_vectorized_calbr(X_probes, c_free, c_cong, lam, tau)
    return V_est.reshape(-1)


# optimize lam and tau variables
X_probes, D, N, M, V_true = build_data(data_ngsim[1], 0)
popt, pcov = curve_fit(GASM_pred, X_probes, V_true, bounds=[(60,-25,0,0),(100,-10,0.2,0.067)])
V_est = V_estm_vectorized_calbr(X_probes, lam=popt[2], tau=popt[3])
print(popt)

# save data
np.save('./5_Testing data/Ngsim/GASM_pred0.npy', V_est)

# plot optimized
plt.figure()
plt.imshow(V_est, cmap='jet_r', aspect='auto')
plt.colorbar()
plt.show()
