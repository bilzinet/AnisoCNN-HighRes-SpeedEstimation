"""
Created on Sun Aug 22 2021
@author: btt1

Traffic State Estimation Comparison Method using LWR-v PDE [Daniel et al. 2008, IEEE CDC Proceedings]
Numerical method - Godunov scheme
Assimilation technqiue - Ensemble Kalman Filter

"""

# import packages
import numpy as np
import matplotlib.pyplot as plt

# some settings
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
np.set_printoptions(precision=4, suppress=True)


# =============================================================================
# Some useful functions
# =============================================================================

def Rv(v):
    
    return np.round(np.power(v,2) - v*v_max, 4)

def model_update(n_prev, x_prev_k, V_a, V_b):
    '''
    Update velocity dyanmics following LWR-v PDE model
    '''
    x_hat_k = np.zeros(x_prev_k.shape)
    for m in range(x_hat_k.shape[0]):
        
        # upstream boundary cell
        if (m == 0):
            v_m_n = x_prev_k[m]
            v_nextm_n = x_prev_k[m+1]
            v_prevm_n = (V_a[n_prev] + V_a[n_prev+1])/2
        # downstream boundary cell
        elif (m==M-1):
            v_m_n = x_prev_k[m]
            v_nextm_n = (V_b[n_prev] + V_b[n_prev+1])/2
            v_prevm_n = x_prev_k[m-1]
        # other cells
        else:
            v_m_n = x_prev_k[m]
            v_prevm_n = x_prev_k[m-1]
            v_nextm_n = x_prev_k[m+1]
            
        g_out = bound_flows(v_m_n, v_nextm_n, v_c)
        g_in = bound_flows(v_prevm_n, v_m_n, v_c)
        v_m_nextn = v_m_n - (delt/delx)*(g_out - g_in)
        
        x_hat_k[m] = v_m_nextn
            
    return x_hat_k

def bound_flows(v_1, v_2, v_c):
    
    if (v_1 <= v_2) and (v_2 <= v_c):
        g = Rv(v_2)
    elif (v_1 <= v_c) and (v_c <= v_2):
        g = Rv(v_c)
    elif (v_c <= v_1) and (v_1 <= v_2):
        g = Rv(v_1)
    elif (v_1 >= v_2):
        g = max( Rv(v_1), Rv(v_2) )
    
    return g


# =============================================================================
# Configuration
# =============================================================================

# road params
v_max = 60
k_max = 140

# discretization params
x_len = 670
t_len = 2760
t_sta = 600
t_end = 900
t_len = t_end - t_sta
delx = 30 / 1000
delt = 1 / 3600
cfl_limit = v_max*(delt/delx)
M = np.floor((x_len/1000) / delx).astype(np.int)        # number of space discretizations
N = np.floor((t_len/3600) / delt).astype(np.int)        # number of time discretizations
K = 100                                                 # number of samples (ensembles)

print(f'\n\tCFL limit: {cfl_limit:0.05f}' )
if cfl_limit <= 1.0:
    print('\tCFL condition satisfied !')
else:
    print('\tCFL condition not satisfied !')
print(f'\tLength of road section: {x_len} m')
print(f'\tTime period of simulation: {t_len} sec')
print(f'\tDiscretization size: {delx*1000} m x {delt*3600} sec')
print(f'\tDiscretization shape: ({M} x {N})')

# model params
v_c = v_max/2


# =============================================================================
# Build data matrix and Implement EnKF
# =============================================================================

def load_data(df, hw1, hw2, c=3, h=80, w=60, h_act=67, w_sta=600, w_act=900, max_speed = 95):

    output_Y = np.load('./5_Testing data/{}/out_data_{}_{}.npy'.format(df,df,hw1))
    input_X = np.load('./5_Testing data/{}/inp_data_{}_{}.npy'.format(df,df,hw2))
    x_test = input_X[:, :h_act, w_sta:w_act, :]
    y_test = output_Y[:, :h_act, w_sta:w_act, :]
    x_bin = (x_test.sum(axis=3) != 0)
    y_map = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
    x_map = x_bin * y_map
    data = [x_test, y_test*max_speed, x_map*max_speed, x_bin.astype(np.int32)]
    
    return data

def msmrt_matrix(n, V_loc, V_meas):
    
    y_nloc = V_loc[:,n]
    y_nmeas = V_meas[:,n]
    pn = np.count_nonzero(y_nloc)
    if pn == 0:
        H_n = np.zeros((1,y_nloc.shape[0]))
        y_n = np.zeros((1,1))
    else:
        vn = np.nonzero(y_nloc)
        H_n = np.zeros((pn, y_nloc.shape[0]))
        for i in range(H_n.shape[0]):
            H_n[i, vn[0][i]] = 1
        y_n = y_nmeas[vn[0]].reshape(-1, 1)
        
    return y_n, H_n

def R_matrix(n, V_meas):
    
    y_nmeas = V_meas[:,n]
    vn = np.nonzero(y_nmeas)
    y_n = y_nmeas[vn[0]]
    R = np.zeros(y_n.shape*2)
    for i in range(R.shape[0]):
        if i%2 == 0:
            R[i,i] = 0.00390625 * np.power(y_n[i], 2)             # 0.015625
        elif i%2 == 1:
            R[i,i] = 0.00390625 * np.power(y_n[i], 2)
    
    return R

def V_noisyprior(M, K, V_true, n=0, x_std=1.5, max_speed=95):
    
    x_0 = V_true[:,n].reshape(-1, 1)
    init_noise = np.random.normal(0, x_std, size=(M, K))
    x_0_noise = x_0 + init_noise
    x_0_noise[x_0_noise > max_speed] = max_speed
    x_0_noise[x_0_noise < 0] = 0
    
    return x_0_noise
    

# # Load data
# df = 'Ngsim'
# hw1 = 'us101_lane2'
# hw2 = 'us101_lane2_10p'
# max_speed = v_max
# data_ngsim = load_data(df, hw1, hw2)

# # True and measurement speed matrix - particular sample
# sam = 10
# v_true = data_ngsim[1][0].squeeze()
# v_meas = data_ngsim[2][sam]
# v_measloc = data_ngsim[3][sam]

# # True and measurement matrix for test resolution
# V_true = (v_true[0::3,:][:M,:]+v_true[1::3,:]+v_true[2::3,:])/3
# V_loc = (v_measloc[0::3,:][:M,:]+v_measloc[1::3,:]+v_measloc[2::3,:])/3
# V_loc = (V_loc != 0).astype(np.int32)
# V_meas = np.multiply(V_true, V_loc)

# # plt.figure()
# # plt.imshow(V_meas, origin='upper', cmap='gray_r', aspect='auto')
# # plt.xlabel('Time (sec)', fontsize=12)
# # plt.ylabel('Space (x30 m)', fontsize=12)

# # Step 1: sample initial states (smooth prior)
# n_0 = 60
# x_0 = V_noisyprior(M, K, V_true, n_0, x_std=1.5, max_speed=95)
# V_a = V_true[-1, :]
# V_b = V_true[0, :]

# # initialize
# V_est = np.zeros((M, N))
# x_prev = x_0.copy()
# V_est[:,n_0] = x_prev.mean(axis=1)

# for n in range(n_0+1, N):
    
#     print('Time step: ', n)
#     y_n, H_n = msmrt_matrix(n, V_loc, V_meas)
#     R_n = R_matrix(n, V_meas)
    
#     # Step 2: Predict velocity dynamics
#     x_hat_n = np.zeros(x_prev.shape)
#     for k in range(x_hat_n.shape[1]):
#         x_hat_nk = model_update(n-1, x_prev[:,k], V_a, V_b)
#         x_hat_n[:,k] = x_hat_nk
    
#     # Step 3: Calculate mean and covariance of ensemble prediction
#     v_n = x_hat_n.mean(axis=1).reshape(-1,1)
#     E_n = x_hat_n - v_n
#     P_n = (1/(K-1))*np.dot(E_n, E_n.T)
    
#     # Step 4: Compute Kalman gain and Update predicted velocities
#     if sum(y_n == 0) == 1:
#         # if no measurements available, goes with what model predicts
#         x_n = x_hat_n
#     else:
#         A_n = (H_n.dot(P_n)).dot(H_n.T) + R_n
#         B_n = np.linalg.inv(A_n)
#         G_n = P_n.dot(H_n.T).dot(B_n)
        
#         C_n = y_n - H_n.dot(x_hat_n)
#         x_n = x_hat_n + G_n.dot(C_n)
    
#     # store results and forward to next time step
#     V_est[:,n] = x_n.mean(axis=1)


# plt.figure()
# plt.imshow(V_true[:,60:120], origin='upper', cmap='jet_r', aspect='auto')

# plt.figure()
# plt.imshow(V_est[:,60:120], origin='upper', cmap='jet_r', aspect='auto')


# =============================================================================
# NGSIM Prediction: Ensemble Kalman Filter implementation
# =============================================================================

# True and measurement speed matrix - particular sample
def get_datamatrix(data):
    
    sams = 5
    data_matrix = []
    for sam in range(sams):
        v_true = data[1][0].squeeze()
        v_meas = data[2][sam]
        v_measloc = data[3][sam]
        V_true = (v_true[0::3,:][:M,:]+v_true[1::3,:]+v_true[2::3,:])/3
        V_loc = (v_measloc[0::3,:][:M,:]+v_measloc[1::3,:]+v_measloc[2::3,:])/3
        V_loc = (V_loc != 0).astype(np.int32)
        V_meas = np.multiply(V_true, V_loc)
        data_matrix.append([V_loc, V_meas, V_true])
    
    return data_matrix

def EnKF_SingleRun(V_loc, V_meas, V_true, K=50):
    
    # Step 1: sample initial states (smooth prior)
    n_0 = 0
    x_0 = V_noisyprior(M, K, V_true, n_0, x_std=2.5, max_speed=95)
    V_a = V_true[0, :]
    V_b = V_true[-1, :]
    
    # initialize
    V_est = np.zeros((M, N))
    x_prev = x_0.copy()
    V_est[:,n_0] = x_prev.mean(axis=1)
    for n in range(n_0+1, N):
        
        # print('Time step: ', n)
        y_n, H_n = msmrt_matrix(n, V_loc, V_meas)
        R_n = R_matrix(n, V_meas)
        # Step 2: Predict velocity dynamics
        x_hat_n = np.zeros(x_prev.shape)
        for k in range(x_hat_n.shape[1]):
            x_hat_nk = model_update(n-1, x_prev[:,k], V_a, V_b)
            x_hat_n[:,k] = x_hat_nk
        # Step 3: Calculate mean and covariance of ensemble prediction
        v_n = x_hat_n.mean(axis=1).reshape(-1,1)
        E_n = x_hat_n - v_n
        P_n = (1/(K-1))*np.dot(E_n, E_n.T)
        # Step 4: Compute Kalman gain and Update predicted velocities
        if sum(y_n == 0) == 1:
            # if no measurements available, goes with what model predicts
            x_n = x_hat_n
        else:
            A_n = (H_n.dot(P_n)).dot(H_n.T) + R_n
            B_n = np.linalg.inv(A_n)
            G_n = P_n.dot(H_n.T).dot(B_n)
            
            C_n = y_n - H_n.dot(x_hat_n)
            x_n = x_hat_n + G_n.dot(C_n)
        # store results and forward to next time step
        V_est[:,n] = x_n.mean(axis=1)
        
    return V_est

def EnFK_error(data_p):
    
    ests = []
    errors = []
    for i in range(len(data_p)):
        # print('\tSample: ',i)
        V_loc, V_meas, V_true = data_p[0]
        V_est = EnKF_SingleRun(V_loc, V_meas, V_true)
        rmse = np.sqrt(np.nanmean(np.power(V_est-V_true, 2)))
        ests.append(V_est)
        errors.append(rmse)
    
    return errors, ests


# Load data
df = 'Ngsim'
data_ngsim = []
for i in ['us101_lane2_3p','us101_lane2_5p','us101_lane2_10p','us101_lane2_20p']:
    data = load_data(df, 'us101_lane2', i)
    data_matrix = get_datamatrix(data)
    data_ngsim.append(data_matrix)

# Predict samples and compute error
ngsim_preds = []
ngsim_rmse = []
print('\nEnsemble Kalman Filter prediction results...')
for j in range(3):
    print('\nData: ', j)
    r_p, e_p = EnFK_error(data_ngsim[j])
    ngsim_preds.append(e_p)
    ngsim_rmse.append((np.nanmean(r_p), np.nanstd(r_p)))
    print(f'RMSE: {np.nanmean(r_p):0.03f} +- {np.nanstd(r_p):0.03f}')
    
# save results
np.save('./5_Testing data/Ngsim/EnKF_preds.npy', np.array(ngsim_preds))
np.save('./5_Testing data/Ngsim/EnKF_preds_rmse.npy', np.array(ngsim_rmse))



# r_p, e_p = EnFK_error(data_ngsim[-1])
# print(f'RMSE: {np.nanmean(r_p):0.03f} +- {np.nanstd(r_p):0.03f}')

# plt.figure()
# plt.imshow(e_p[0], cmap='jet_r', aspect='auto')
# plt.colorbar()

# plt.figure()
# plt.imshow(V_true, cmap='jet_r', aspect='auto')
# plt.colorbar()


