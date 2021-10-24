# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:31:27 2020

@author: btt1

Project: Traffic state estimation using deep convolutional neural networks

Code: Generate input-output data for training and testing CNNs

Data format: 

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tms = False
import matplotlib
matplotlib.rcParams['text.usetex'] = tms
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
np.set_printoptions(precision=4, suppress=True)

from matplotlib import rcParams
plt.rcParams["font.family"] = "Times New Roman"
rcParams['mathtext.fontset']='cm'

# =============================================================================
# Some helper functions
# =============================================================================


def readData(vr_filename, link_id, lane_id, params, vr_folder="1_Simulations and raw data\\Simulated"):
    '''
    Read the probe vehicle trajectory dataset
    '''
    # Unpack some parameters
    time_start = params['time_start']
    time_end = params['time_end']
    sec_len = int(params['sec_end']-params['sec_start'])
    # Read trajectory data
    vr_column_names = ['SimSec','TimeInNet','VehNo','LinkNo','LaneInd','Pos','LatPos','CoordFront','CoordRear','Length','Speed','Acc','TotDistTrav','FollowDist','Hdwy']
    traj_df = pd.read_csv(os.path.join(os.getcwd(), vr_folder, vr_filename), 
                          skiprows=30, sep=';', names=vr_column_names)
    # Preprocess and filter
    traj_df.SimSec = traj_df.SimSec.apply(lambda x: np.around(x))
    traj_data = traj_df[traj_df['SimSec'] >= time_start]
    traj_data = traj_data[traj_data['SimSec'] < time_end]
    traj_data = traj_data[traj_data['LinkNo'] == link_id]
    traj_data = traj_data[traj_data['LaneInd'] == lane_id]
    traj_data = traj_data[traj_data['Pos'] < sec_len]
    
    return traj_data



# =============================================================================
# Analyse simulated dataset
# =============================================================================

# Parameters used
t_int = 1; x_int = 10
l_dn = 80; l_up = 40
link_id = 4; lane_id = 1
sec_start = 0; sec_end = 800
min_speed = 0; max_speed = 100
min_density = 0; max_density = 155
time_start = 600; time_end =  7800
time_len = int(time_end - time_start); sec_len = int(sec_end - sec_start)
t_num = int((time_end-time_start)/t_int); x_num = int(sec_len/x_int)
params = { 't_int':t_int, 'x_int':x_int, 'sec_start':sec_start, 'sec_end':sec_end, 
           'time_start':time_start, 'time_end':time_end, 't_num':t_num, 'x_num':x_num }

# Load simulated data
dfs = []
for df_name in ['cong more', 'cong', 'free']:
    df = readData("Abudhabi_Alain_Road_Wide {}.fzp".format(df_name), link_id, lane_id, params)
    veh_headway = df['Hdwy'].to_numpy().copy()
    loc_density = np.round(1e6/veh_headway, 4)
    # loc_density[loc_density > max_density] = max_density
    loc_density[loc_density < min_density] = min_density
    df['Density'] = loc_density
    df['Flow'] = df['Density']*df['Speed']
    dfs.append(df)
df_cong_more = dfs[0]
df_cong = dfs[1]
df_free = dfs[2]

# Flow - density scatter plot (separate plots for different datasets)
labels = ['Slow moving data', 'Free flowing data', 'Congested data']
colors = ['tab:blue','tab:green','tab:orange']
for i, df in enumerate([df_cong, df_free, df_cong_more]):
    plt.figure(figsize=(8,6))
    num_samples = 500
    q = df.Flow.to_numpy()
    k = df.Density.to_numpy()
    rand_indx = np.random.randint(0, df.Speed.shape[0], num_samples)
    x = k[rand_indx]
    y = q[rand_indx]
    y_filter = y[y <= 3350]
    x_filter = x[y <= 3350]
    plt.scatter(x_filter, y_filter, s=15, ec=colors[i], fc='none', lw=1, label=labels[i], alpha=1.0)
    plt.xlabel('Density (vehs/km)', fontsize=13)
    plt.ylabel('Flow (vehs/hr)', fontsize=13)
    plt.legend(fontsize=13)
    plt.title('Flow-density scatter plot', fontsize=14)
    plt.xlim([0, 200])
    plt.ylim([0, 4000])
    plt.grid()

# Flow - density scatter plot
labels = ['Congested data', 'Slow-moving data', 'Free-flowing data']
colors = ['tab:orange','tab:green','tab:blue']
markers = ['*', '+', 'o']
marker_size = [60, 60, 24]
q_sim = np.empty(0); k_sim = np.empty(0)
plt.figure(figsize=(5,4))
for i, df in enumerate([df_cong_more, df_cong, df_free]):
    num_samples = 500
    q = df.Flow.to_numpy()
    k = df.Density.to_numpy()
    rand_indx = np.random.randint(0, df.Speed.shape[0], num_samples)
    x = k[rand_indx]
    y = q[rand_indx]
    y_filter = y[y <= 3350]; q_sim = np.append(q_sim, y_filter)
    x_filter = x[y <= 3350]; k_sim = np.append(k_sim, x_filter)
    plt.scatter(x_filter, y_filter, s=marker_size[i], marker=markers[i],
                ec=colors[i], fc='white', lw=1,
                label=labels[i], alpha=1.0)
plt.xlabel('Density (vehs/km)', fontsize=15)
plt.ylabel('Flow (vehs/hr)', fontsize=15)
plt.xticks(fontsize=13); plt.yticks(fontsize=13)
plt.legend(fontsize=13)
# plt.title('Flow-density scatter plot', fontsize=14)
plt.xlim([-0, 160])
plt.ylim([0, 4000])
plt.grid()
plt.tight_layout()
plt.savefig('FDscatter_Traindata.pdf', bbox_inches='tight')


# Speed - density relationship
labels = ['Slow moving data', 'Free flowing data', 'Congested data']
speeds = [df_cong.Speed.to_list(), df_free.Speed.to_list(), df_cong_more.Speed.to_list()]
plt.figure(figsize=(8,6))
plt.hist(speeds, bins=30, rwidth=0.8, alpha=0.8, stacked=True, density=True)
plt.xlabel('Speed (kmp/hr)', fontsize=13); plt.xticks(fontsize=11)
plt.ylabel('Frequency', fontsize=13); plt.yticks(fontsize=11)
plt.legend(labels, loc='upper left', fontsize=11)
# plt.title('Speed distributions', fontsize=13)
plt.grid()


# Space-time-speed contour plot
labels = ['Congested', 'Slow-moving', 'Free-flowing']
fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(5,3.7))
for i, df in enumerate([df_cong_more, df_cong, df_free]):
    time_period = 300 # Seconds
    start_time = 5100
    end_time = int(start_time+time_period)
    df1 = df.query('SimSec >= @start_time')
    df2 = df1.query('SimSec < @end_time')
    t = df2.SimSec.to_numpy() - start_time
    x = df2.Pos.to_list()
    v = df2.Speed.to_list()
    img = axs[i].scatter(t, x, c=v, s=5, cmap='jet_r', vmin=0, vmax=100)
    axs[i].set_title(labels[i], fontsize=15)
    plt.setp(axs[i].get_xticklabels(), fontsize=13)

axs[0].set_ylabel('Space (m)', fontsize=15)
axs[1].set_xlabel('Time (s)', fontsize=15)
plt.setp(axs[0].get_yticklabels(), fontsize=13)
plt.xlim([0, time_period])
plt.ylim([0, 800])
cbar_ax = fig.add_axes([0.93, 0.20, 0.02, 0.60])
cbar = fig.colorbar(img, cax=cbar_ax)
cbar.ax.tick_params(labelsize=13)
cbar.ax.set_ylabel('Vehicle Speed $V$ (km/hr)', fontsize=14)
# plt.tight_layout()
plt.savefig('SpaceTime_Traindata.pdf', bbox_inches='tight')



# =============================================================================
# Comparison with NGSIM, HIGHD datasets
# =============================================================================


# Flow - density relationship (comparison with HighD)

# Flow density of HighD data (normalized)
df_fd_lane = pd.read_csv('df_highd_hw44_lane6.csv')  # df_highd_hw25_lane4, df_highd_hw44_lane6
q_highd = np.empty(0); k_highd = np.empty(0)
num_samples = 500
q = df_fd_lane.Flow.to_numpy()
k = df_fd_lane.Density.to_numpy()
rand_indx = np.random.randint(0, df_fd_lane.Speed.shape[0], num_samples)
x = k[rand_indx]
y = q[rand_indx]
x_filter = x[y <= 4500]
y_filter = y[y <= 4500]
q_highd = np.append(q_highd, y_filter)
k_highd = np.append(k_highd, x_filter)

plt.figure(figsize=(3.5,3.5))
plt.scatter(k_sim, q_sim, s=15, ec='tab:blue', fc='white', marker='o',
            lw=1, label='Simulated data', alpha=1.0)
plt.scatter(k_highd, q_highd, s=35, ec='tab:orange', fc='white', marker='*',
            lw=1, label='HighD data (HW-25)', alpha=1.0)
plt.xlabel('Density (vehs/km)', fontsize=15)
plt.ylabel('Flow (vehs/hr)', fontsize=15)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=12)
# plt.title('Flow-density scatter plot', fontsize=14)
plt.xlim([0, 160])
plt.ylim([0, 5100])
plt.grid()
plt.tight_layout()
plt.savefig('FD_sim-highd-hw44.pdf', bbox_inches='tight')


# Flow - density relationship (comparison with NGSIM)

# Flow density of HighD data (normalized)
df_ng_lane = pd.read_csv('df_ngsim_us101_lane2.csv')
q_ngsim = np.empty(0); k_ngsim = np.empty(0)
num_samples = 500
q = df_ng_lane.Flow.to_numpy()
k = df_ng_lane.Density.to_numpy()
rand_indx = np.random.randint(0, df_ng_lane.Speed.shape[0], num_samples)
x = k[rand_indx]
y = q[rand_indx]
x_filter = x[y <= 4000]
y_filter = y[y <= 4000]
q_ngsim = np.append(q_ngsim, y_filter)
k_ngsim = np.append(k_ngsim, x_filter)

plt.figure(figsize=(3.5,3.5))
plt.scatter(k_sim, q_sim, s=15, ec='tab:blue', fc='white', marker='o',
            lw=1, label='Simulated data', alpha=0.8)
plt.scatter(k_ngsim, q_ngsim, s=35, ec='tab:orange', fc='white', marker='*',
            lw=1, label='NGSIM data (US101)', alpha=0.8)
plt.xlabel('Density (vehs/km)', fontsize=16)
plt.ylabel('Flow (vehs/hr)', fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=12)
# plt.title('Flow-density scatter plot', fontsize=14)
plt.xlim([0, 160])
plt.ylim([0, 5100])
plt.grid()
plt.tight_layout()
plt.savefig('FD_sim-ngsimUS101.pdf', bbox_inches='tight')


# =============================================================================
# For TRB poster
# =============================================================================

labels = ['Congested data', 'Slow moving data', 'Free flowing data']
colors = ['tab:orange','tab:green','tab:blue']
markers = ['o', 's', 'p']
q_sim = np.empty(0); k_sim = np.empty(0)
plt.figure(figsize=(5,4))
for i, df in enumerate([df_cong_more, df_cong, df_free]):
    num_samples = 500
    q = df.Flow.to_numpy()
    k = df.Density.to_numpy()
    rand_indx = np.random.randint(0, df.Speed.shape[0], num_samples)
    x = k[rand_indx]
    y = q[rand_indx]
    y_filter = y[y <= 3350]; q_sim = np.append(q_sim, y_filter)
    x_filter = x[y <= 3350]; k_sim = np.append(k_sim, x_filter)
    plt.scatter(x_filter, y_filter/1000, s=18, marker=markers[i],
                ec=colors[i], fc='white', lw=1,
                label=labels[i], alpha=1.0)
plt.xlabel(r'Density ($vehs/km$)', fontsize=16)
plt.ylabel(r'Flow ($x10^3$ $vehs/hr$)', fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=13)
# plt.title('Flow-density scatter plot', fontsize=14)
plt.xlim([-0, 160])
plt.ylim([0, 4])
plt.grid()
plt.tight_layout()
plt.savefig('FDscatter_Traindata-high res.png', dpi=1200, bbox_inches='tight')
