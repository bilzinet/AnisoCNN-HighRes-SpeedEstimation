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
import pickle as pkl
import matplotlib.pyplot as plt
import ColorMapping as cmg

np.set_printoptions(precision=4, suppress=True)

# =============================================================================
# Some helper functions
# =============================================================================

def genCoordinates(time_start, time_end, sec_start, sec_end, t_int, x_int, t_num, x_num):
    '''
    Generate space-time grid coordinates.
    '''
    coords = np.empty((t_num, x_num, 2), dtype=np.float32)
    for t_id, t in enumerate(range(time_start, time_end, t_int)):
        for x_id, x in enumerate(range(sec_start, sec_end, x_int)):
            coords[t_id, x_id, 0] = t
            coords[t_id, x_id, 1] = x+x_int/2
    
    return coords

def readData(vr_filename, link_id, lane_id, params, vr_folder="1_Simulations and raw data\\Ngsim"):
    '''
    Read the probe vehicle trajectory dataset
    '''
    # Unpack some parameters
    time_start = params['time_start']
    time_end = params['time_end']
    sec_len = int(params['sec_end']-params['sec_start'])
    
    # Read trajectory data
    traj_df = pd.read_csv(os.path.join(os.getcwd(), vr_folder, vr_filename))
    
    # Preprocess and filter
    df1 = traj_df.query('Lane_ID==@lane_id')
    df2 = df1.query('Location==@link_id')
    
    df3 = pd.DataFrame()
    df3['VehNo'] = df2['Vehicle_ID']
    df3['SimSec'] = df2.Global_Time/1000 # Convert to sec
    df3['SimSec'] = df3['SimSec'] - np.min(df3['SimSec'].to_list()) # scale
    df3['Pos'] = df2['Local_Y']*0.3048  # Convert to metre
    df3['Speed'] = df2['v_Vel']*1.09728 # Convert to km/hr
    df3['Density'] = 1000/(df2['Space_Headway'].to_numpy()*0.3048) # Convert to veh/km
    df3['Density'] = df3['Density'].apply(lambda x: min(x, 155))
    df3['Flow'] = 3600/(df2['Time_Headway'].to_numpy())
    df3['Flow'] = df3['Flow'].apply(lambda x: min(x, 5000))
    
    df4 = df3[df3['SimSec'].map(lambda x: x%1==0.0)] # Keep one sec cadence data
    df4 = df4[df4['SimSec'] >= time_start]
    df4 = df4[df4['SimSec'] < time_end]
    df4 = df4[df4['Pos'] < sec_len]
    
    return df4

def sampleProbes(traj_data, probe_per, load=False):
    '''
    Sample vehicles from given vehicle ID list.
    '''
    if load:
        sampled_vehs = np.load('./SampleVehIDs_{}per.npy'.format(probe_per))
    else:
        vehIDs = traj_data.VehNo.unique()
        num_probes = int(len(vehIDs)*probe_per/100)
        sampled_vehs = np.random.choice(vehIDs, num_probes)
        np.save('./SampledVehIDs_{}per.npy'.format(probe_per), sampled_vehs)
    
    return sampled_vehs

def getProbeCoords(traj_data, probe_vehs, t_int):
    '''
    Get space-time-speed coordinates of probe vehicles.
    '''
    probe_coords = []
    
    for sim_sec in traj_data.SimSec.unique():
        
        if sim_sec % t_int == 0:
            df1 = traj_data.query("SimSec == @sim_sec")
            
            if np.sum(np.isin(probe_vehs, df1.VehNo.unique())) > 0:
                for probe_veh in probe_vehs[np.isin(probe_vehs, df1.VehNo.unique())]:
                    
                    veh_pos = df1.loc[df1['VehNo'] == probe_veh, 'Pos'].to_list()[0]
                    veh_speed = df1.loc[df1['VehNo'] == probe_veh, 'Speed'].to_list()[0]
                    probe_coords.append((sim_sec, veh_pos, np.around(veh_speed, 2)))
    
    return np.array(probe_coords)

def cellSpeedCalc(cell_ix, p_speed, p_occup, occupiedCells):
    '''
    Calculate the speed in each cell based on probe vehicle.
    '''
    if cell_ix not in occupiedCells:
        cell_speed = p_speed
        occupiedCells[cell_ix] = [[p_speed, p_occup]]
    else:
        num = 0
        den = 0
        for k in occupiedCells[cell_ix]:
            num += k[0]*k[1]
            den += k[1]
        cell_speed = num/den
        occupiedCells[cell_ix].append([p_speed, p_occup])
    
    return cell_speed, occupiedCells

def genProbeTimeSpace(probe_coords, params, veh_len=4):
    '''
    Fill probe coordinates on a space-time grid.
    '''
    # Unpack some parameters
    time_start = params['time_start']
    time_end = params['time_end']
    sec_start = params['sec_start']
    sec_end = params['sec_end']
    t_int = params['t_int']
    x_int = params['x_int']
    t_num = params['t_num']
    x_num = params['x_num']
    
    # Generate the space-time coordinates
    grid_coords = genCoordinates(time_start, time_end, 
                                 sec_start, sec_end, 
                                 t_int, x_int, 
                                 t_num, x_num)
    
    # Section lattice; each cell is represented by it's center long. coord.
    sec_lattice = np.flip(grid_coords[0, :, 1])
    probe_grid = np.zeros((x_num, t_num, 3), dtype=np.float32)
    aux_grid = np.zeros((x_num, t_num, 2), dtype=np.float16)
    probe_coords = pd.DataFrame(probe_coords, columns=['time','sec','speed'])
    
    # Loop through each time step of the data
    for t_id in range(probe_grid.shape[1]):
        
        # print("\tTimestep: ", t_id)
        
        # Filter probe data corresponding to time step t
        t =  grid_coords[t_id,:,0][0]
        df = probe_coords.query("time==@t")
        df = df.sort_values('sec', ascending=False, inplace=False)
        occupiedCells = {}
        
        # Update cells having a probe vehicle. Update is carried out 
        # from downstream to upstream cells to account for multiple 
        # observations in each cell - in this case we take weighted average
        # with weights as vehicle space occupancy in each cell.
        for i, rec in df.iterrows():
            
            p_s = rec['speed']                      # Speed of probe veh
            p_x = rec['sec']                        # Coordinate of probe veh
            p_x_f = min(p_x+veh_len/2, sec_len)     # Front coordinate of veh
            p_x_b = max(p_x-veh_len/2, 0)           # Back coordinate of veh
            
            try:
                d_ix = np.squeeze(np.where(sec_lattice >= p_x))[-1]
            except IndexError:
                d_ix = np.squeeze(np.where(sec_lattice < p_x))[0]
            try:
                u_ix = np.squeeze(np.where(sec_lattice < p_x))[0]
            except IndexError:
                u_ix = np.squeeze(np.where(sec_lattice >= p_x))[-1]
            
            d_x = sec_lattice[d_ix]
            u_x = sec_lattice[u_ix]
            
            if p_x_f <= u_x+x_int/2:
                # update upstream cell if vehicle only occupy upstream cell
                p_occup = round(veh_len/x_int, 3)
                c_speed, occupiedCells = cellSpeedCalc(u_ix,  p_s, p_occup, occupiedCells)
                probe_grid[u_ix, t_id, :] = colmap.get_rgb(c_speed)[:3]
                aux_grid[u_ix, t_id, :] = [1, c_speed]
            
            elif p_x_b >= d_x-x_int/2:
                # update downstream cell if vehicle only occupy downstream cell
                p_occup = round(veh_len/x_int, 3)
                c_speed, occupiedCells = cellSpeedCalc(d_ix, p_s, p_occup, occupiedCells)
                probe_grid[d_ix, t_id, :] = colmap.get_rgb(c_speed)[:3]
                aux_grid[d_ix, t_id, :] = [1, c_speed]
            
            else:
                # update downstream and upstream cell if vehicle occupy both cells
                p_occup = round((u_x+x_int/2-p_x_b)/x_int, 3)
                c_speed, occupiedCells = cellSpeedCalc(u_ix,  p_s, p_occup, occupiedCells)
                probe_grid[u_ix, t_id, :] = colmap.get_rgb(c_speed)[:3]
                aux_grid[u_ix, t_id, :] = [1, c_speed]
                
                p_occup = round((p_x_f-d_x+x_int/2)/x_int, 3)
                c_speed, occupiedCells = cellSpeedCalc(d_ix, p_s, p_occup, occupiedCells)
                probe_grid[d_ix, t_id, :] = colmap.get_rgb(c_speed)[:3]
                aux_grid[d_ix, t_id, :] = [1, c_speed]
    
    return probe_grid, aux_grid

def localSpeedfield(full_coords, params):
    '''
    Obtain the local speed field from trajectories using linear interpolation.
    '''
    # Unpack some parameters
    time_start = params['time_start']
    time_end = params['time_end']
    sec_start = params['sec_start']
    sec_end = params['sec_end']
    t_int = params['t_int']
    x_int = params['x_int']
    t_num = params['t_num']
    x_num = params['x_num']
    
    # Generate the space-time coordinates
    grid_coords = genCoordinates(time_start, time_end, sec_start, sec_end, 
                                 t_int, x_int, t_num, x_num)
    
    # Section lattice; each cell is represented by it's center long. coord.
    sec_lattice = np.flip(grid_coords[0, :, 1])
    speed_field = np.zeros((x_num, t_num), dtype=np.float32)
    full_coords = pd.DataFrame(full_coords, columns=['time','sec','speed'])
    
    # Loop through each time step of the data
    for t_id in range(speed_field.shape[1]):
        
        # print("\tTimestep: ", t_id)
        # Filter probe data corresponding to time step t
        t =  grid_coords[t_id,:,0][0]
        df = full_coords.query("time==@t")
        df = df.sort_values('sec', ascending=False, inplace=False)
        
        for x_id in range(sec_lattice.shape[0]):
            
            x = sec_lattice[x_id]
            df_up = df[df['sec'] < x]
            df_dn = df[df['sec'] >= x]
            
            # Interpolate speed (definition from pg.167 Trieber and Kesting 2013)
            if (df_up.empty) and (df_dn.empty):
                v_cell = max_speed
            elif df_up.empty:
                x_dn = df_dn['sec'].to_list()[-1]
                v_dn = df_dn['speed'].to_list()[-1]
                wt_dn = min((x_dn-x)/l_dn, 1)
                v_cell = np.round((1-wt_dn)*v_dn + wt_dn*max_speed, 3)
            elif df_dn.empty:
                x_up = df_up['sec'].to_list()[0]
                v_up = df_up['speed'].to_list()[0] 
                wt_up = min((x-x_up)/l_up, 1)
                v_cell = np.round((1-wt_up)*v_up + wt_up*max_speed, 3)
            else:
                x_up = df_up['sec'].to_list()[0]
                v_up = df_up['speed'].to_list()[0]
                x_dn = df_dn['sec'].to_list()[-1]
                v_dn = df_dn['speed'].to_list()[-1]
                wt_dn = min((x_dn-x)/l_dn, 1)
                wt_up = min((x-x_up)/l_up, 1)
                if (wt_dn<1) and (wt_up<1):
                    v_cell = np.round(v_dn*(x-x_up)/(x_dn-x_up) + v_up*(x_dn-x)/(x_dn-x_up),3)
                elif (wt_dn<=1) and (wt_up==1):
                    v_cell = np.round((1-wt_dn)*v_dn + wt_dn*max_speed,3)
                elif (wt_dn==1) and (wt_up<=1):
                    v_cell = np.round((1-wt_up)*v_up + wt_up*max_speed,3)
                else:
                    v_cell = max_speed
#            speeds.append(v_cell)
            speed_field[x_id, t_id] = v_cell
    
    return speed_field

def genTrainingData(probe_grid, full_grid, method):
    '''
    Generate input-ouput training data from probe and full vehicle speeds.
    '''
    t_num = 60
    x_num = 80
    x_num_act = params['x_num']
    
    inp_spacetime = np.zeros((probe_grid.shape[1]-t_num, x_num, t_num, 3))
    for t in range(inp_spacetime.shape[0]):
        inp_spacetime[t,:x_num_act,:,:] = probe_grid[:,t:t+t_num, :]
    
    if method=='A':
        out_spacetime = np.zeros((probe_grid.shape[1]-t_num, x_num, t_num, 3))
        for t in range(inp_spacetime.shape[0]):
            out_spacetime[t,:x_num_act,:,0] = np.flipud(full_grid[t:t+t_num, :, :].T[0,:,:])
            out_spacetime[t,:x_num_act,:,1] = np.flipud(full_grid[t:t+t_num, :, :].T[1,:,:])
            out_spacetime[t,:x_num_act,:,2] = np.flipud(full_grid[t:t+t_num, :, :].T[2,:,:])
    elif method=='B':
        out_spacetime = np.zeros((probe_grid.shape[1]-t_num, x_num, t_num, 1))
        for t in range(inp_spacetime.shape[0]):
            out_spacetime[t,:x_num_act,:,0] = np.round(full_grid[:,t:t+t_num]/max_speed, 4)
    
    return inp_spacetime, out_spacetime

# =============================================================================
# Read and process the data
# =============================================================================

'''
Meta information about NGSIM dataset:
    
    Location = ["i-80", "us-101", "peachtree","lankershim"]
    
    i-80 is 530m long section and contains trajectory information from
        0-960sec and 3650-5460secs. There are six lanes, and first lane 
        is HOV lane.
    
    us-101 is 670m long section anc contain trajectory data from 0-2760 secs.
        There are six lanes and uni-directional movements. Prefer: lane 2

'''

# Parameters used
link_id = 'us-101'
lane_id = 2
min_speed = 0
max_speed = 100
t_int = 1
x_int = 10
sec_start = 0
sec_end = 670
time_start = 0
time_end = 2760
time_len = int(time_end - time_start)
sec_len = int(sec_end - sec_start)
t_num = int((time_end-time_start)/t_int)
x_num = int(sec_len/x_int)
params = { 't_int':t_int, 'x_int':x_int, 'sec_start':sec_start, 'sec_end':sec_end, 
           'time_start':time_start, 'time_end':time_end, 't_num':t_num, 'x_num':x_num }
vr_filename = "NGSIM_Vehicle_Trajectories_Data.csv"
df = readData(vr_filename, link_id, lane_id, params)
max_speed = 95 #df.Speed.max()

# Speed distributions
plt.figure(figsize=(7,6))
plt.hist(df.Speed, bins=50, rwidth=0.8, alpha=0.7, density=True)
plt.xlabel("Vehicle speed (kmph)", fontsize=12)
plt.ylabel("Density distribution", fontsize=12)
plt.title("Speed distributions", fontsize=14)
plt.legend(['One class'])
plt.grid()
plt.show()

# Trajectory
plt.figure()
plt.scatter(df.SimSec, df.Pos, c=df.Speed, s=5, cmap='jet_r', vmin=0, vmax=100)
plt.colorbar()

# =============================================================================
# For US-101 (extracting boundary conditions)
# =============================================================================

# 
plt.figure()
for v_id in df.VehNo.unique():
    # if v_id % 1 != 0:
        # continue
    df_veh = df.query('VehNo == @v_id')
    t = df_veh.SimSec.to_numpy()
    d = df_veh.Pos.to_numpy()
    s = df_veh.Speed.to_numpy()
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=100)
    # plt.plot(t, d, lw=0.2, c='k')
plt.xlim([200,2000])
plt.ylim([100,600])
plt.colorbar()
plt.grid()
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Space (m)', fontsize=12)
plt.title('Anisotropic trajectories', fontsize=13)


# Zoom trajectory to smaller sections
df1 = df.query('SimSec >= 800')
df2 = df1.query('SimSec < 1000')
with open('US101_ActTraj.pkl', 'wb') as handle:
    pkl.dump(df2, handle, protocol=pkl.HIGHEST_PROTOCOL)
plt.figure()
for v_id in df2.VehNo.unique():
    df_veh = df2.query('VehNo == @v_id')
    t = df_veh.SimSec.to_numpy()
    d = df_veh.Pos.to_numpy()
    s = df_veh.Speed.to_numpy()
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=100)
    plt.plot(t, d, lw=0.2, c='k')
plt.xlim([800,1000])
plt.ylim([100,600])


# Getting boundary condition (vehicle crossing times and speeds)
sec = 100
cross_times = []
speed_x0 = []
for v_id in df.VehNo.unique():
    df_veh = df.query('VehNo == @v_id')
    df_veh_sort = df_veh.sort_values('Pos')
    t = df_veh_sort.SimSec.to_numpy()
    d = df_veh_sort.Pos.to_numpy()
    s = df_veh_sort.Speed.to_numpy()
    
    try:
        d_bef = d[d < sec][-1]
        d_aft = d[d >= sec][0]
    except IndexError:
        continue
    ind_bef = np.argwhere(d == d_bef)[0].item()
    ind_aft = np.argwhere(d == d_aft)[0].item()
    t_bef = t[ind_bef]
    t_aft = t[ind_aft]
    s_bef = s[ind_bef]
    s_aft = s[ind_aft]
    
    t_crs = t_bef + (sec-d_bef)*(t_aft-t_bef)/(d_aft-d_bef)
    s_crs = s_bef + (sec-d_bef)*(s_aft-s_bef)/(d_aft-d_bef)
    cross_times.append(t_crs)
    speed_x0.append(s_crs)

US101_BoundCond = {'x':sec,
                   't':cross_times,
                   'v':speed_x0}
with open('US101_BoundCond.pkl', 'wb') as handle:
    pkl.dump(US101_BoundCond, handle, protocol=pkl.HIGHEST_PROTOCOL)

# =============================================================================
# For I-80 highway (extracting boundary conditions)
# =============================================================================

plt.figure()
for v_id in df.VehNo.unique():
    df_veh = df.query('VehNo == @v_id')
    t = df_veh.SimSec.to_numpy()
    d = df_veh.Pos.to_numpy()
    s = df_veh.Speed.to_numpy()
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=100)
plt.xlim([3750,5250])
plt.ylim([100,500])
plt.colorbar()
plt.grid()
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Space (m)', fontsize=12)
plt.title('Vehicle trajectories', fontsize=13)


# Zoom trajectory to smaller sections
df1 = df.query('SimSec >= 4300')
df2 = df1.query('SimSec < 4600')
with open('I80_lane3_ActTraj.pkl', 'wb') as handle:
    pkl.dump(df2, handle, protocol=pkl.HIGHEST_PROTOCOL)
plt.figure()
for v_id in df2.VehNo.unique():
    df_veh = df2.query('VehNo == @v_id')
    t = df_veh.SimSec.to_numpy()
    d = df_veh.Pos.to_numpy()
    s = df_veh.Speed.to_numpy()
    plt.scatter(t, d, c=s, s=7, cmap='jet_r', vmin=0, vmax=100)
    plt.plot(t, d, lw=0.2, c='k')
plt.xlim([5050, 5250])
plt.ylim([100,500])


# Getting boundary condition (vehicle crossing times and speeds)
sec = 100
cross_times = []
speed_x0 = []
for v_id in df.VehNo.unique():
    df_veh = df.query('VehNo == @v_id')
    df_veh_sort = df_veh.sort_values('Pos')
    t = df_veh_sort.SimSec.to_numpy()
    d = df_veh_sort.Pos.to_numpy()
    s = df_veh_sort.Speed.to_numpy()
    
    try:
        d_bef = d[d < sec][-1]
        d_aft = d[d >= sec][0]
    except IndexError:
        continue
    ind_bef = np.argwhere(d == d_bef)[0].item()
    ind_aft = np.argwhere(d == d_aft)[0].item()
    t_bef = t[ind_bef]
    t_aft = t[ind_aft]
    s_bef = s[ind_bef]
    s_aft = s[ind_aft]
    
    t_crs = t_bef + (sec-d_bef)*(t_aft-t_bef)/(d_aft-d_bef)
    s_crs = s_bef + (sec-d_bef)*(s_aft-s_bef)/(d_aft-d_bef)
    cross_times.append(t_crs)
    speed_x0.append(s_crs)

I80_BoundCond = {'x':sec,
                   't':cross_times,
                   'v':speed_x0}
with open('I80_lane3_BoundCond.pkl', 'wb') as handle:
    pkl.dump(I80_BoundCond, handle, protocol=pkl.HIGHEST_PROTOCOL)



# =============================================================================
# Analyze dataset - Fundamental relations
# =============================================================================

from scipy.optimize import curve_fit

def plot_fd(x, y):
    
    fig = plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=15)
    plt.xlabel('Density (vehs/km)', fontsize=13)
    plt.ylabel('Flow (vehs/hr)', fontsize=13)
    plt.title('Flow density scatter diagram', fontsize=14)
    plt.xlim([0, x.max()])
    plt.ylim([50, y.max()])
    plt.grid()
    
    return fig


def GreenFD(v, k_jam, v_free):
    q = k_jam * (1 - v/v_free) * v
    return q

def NewFrankFD(v, k_jam, v_free, v_back):
    den = 1 - (v_free/v_back)*np.log(1 - v/v_free)
    q = v * k_jam / den
    return q

def TraingFD(k, k_jam, v_free, k_cr):
    
    # np.min([95*k, 95*np.repeat(50,k.shape[0])], axis=0)
    # np.min([95*np.repeat(50,k.shape[0]), (150-k)*(50*100)/(150-50)], axis=0)
    w = (k_cr*v_free)/(k_jam-k_cr)
    q = np.min([k*v_free, w*(k_jam-k)], axis=0)
    return q


# -------- Fitting fundamental relations ---------- # 

# get speed density data
v_act = df.Speed.to_numpy()
k_act = df.Density.to_numpy()

# sample data
num_samples = 5000
ind = np.random.choice(np.arange(1, v_act.shape[0]), num_samples, replace=False)
v_sam = v_act[ind]
k_sam = k_act[ind]
q_sam = k_sam * v_sam
errfree_ind = np.argwhere(q_sam <= 4500).squeeze()
v_sam_mod = v_sam[errfree_ind]
k_sam_mod = k_sam[errfree_ind]
q_sam_mod = q_sam[errfree_ind]


# Greenshield's fundamental relation
popt, pcov = curve_fit(GreenFD, v_sam_mod, q_sam_mod, bounds=([0, 0], [155, 95]))
print('\nGreenshields fundamental relation:')
print(f'\tJam density: {popt[0]:0.03f} vehs/km;\n\tFree-flow speed: {popt[1]:0.03f} km/hr')
fig = plot_fd(k_sam_mod, q_sam_mod)
v_est = np.arange(0.1,popt[1])
q_est = GreenFD(v_est, popt[0], popt[1])
k_est = q_est / v_est
plt.scatter(k_est, q_est)

v_est1 = np.arange(0.1, 60)
q_est1 = GreenFD(v_est1, 140, 60)
k_est1 = q_est1 / v_est1
plt.scatter(k_est1, q_est1)


# Newell-Franklin fundamental relation
popt, pcov = curve_fit(NewFrankFD, v_sam_mod, q_sam_mod, p0=[85,95,18], bounds=([10, 10, 5], [155, 95, 25]))
print('\nNewell_frnklin fundamental relation:')
print(f'\tJam density: {popt[0]:0.03f} vehs/km;\n\tFree-flow speed: {popt[1]:0.03f} km/hr')
fig = plot_fd(k_sam_mod, q_sam_mod)
v_est = np.arange(0.1,popt[1])
q_est = NewFrankFD(v_est, popt[0], popt[1], popt[2])
k_est = q_est / v_est
plt.scatter(k_est, q_est)


# Triangular fundamental relation
popt, pcov = curve_fit(TraingFD, k_sam_mod, q_sam_mod, p0=[85,95,18], bounds=([10, 10, 5], [155, 95, 100]))
print('\nTriangular fundamental relation:')
print(f'\tJam density: {popt[0]:0.03f} vehs/km;\n\tFree-flow speed: {popt[1]:0.03f} km/hr')
fig = plot_fd(k_sam_mod, q_sam_mod)
k_est = np.arange(0.1, popt[0])
q_est = TraingFD(k_est, popt[0], popt[1], popt[2])
v_est = q_est / k_est
plt.scatter(k_est, q_est)

q_est1 = TraingFD(k_est, 155, 50, 55)
plt.scatter(k_est, q_est1)


# Fundamental diagram relation
df = df.query('SimSec <= 600')
num_samples = 1000
rand_indx = np.random.randint(1, df.shape[0], num_samples)
q = df.Flow.to_numpy()
k = df.Density.to_numpy()
v = df.Speed.to_numpy()
x = k[rand_indx]
y = q[rand_indx]
z = v[rand_indx]

plt.figure(figsize=(8,6))
plt.scatter(x, y, s=15, c=z, cmap='jet_r', vmin=0, vmax=100, lw=1, alpha=1.0)
plt.xlabel('Density (vehs/km)', fontsize=13)
plt.ylabel('Flow (vehs/hr)', fontsize=13)
plt.title('Flow density scatter plot (I80 lane 1)', fontsize=14)
plt.xlim([0, 150])
plt.ylim([50, 4500])
plt.colorbar()
plt.grid()

plt.figure(figsize=(8,6))
plt.scatter(k, v, s=15, ec='k', fc='none', lw=1, alpha=1.0)
plt.xlabel('Density (vehs/km)', fontsize=13)
plt.ylabel('Flow (vehs/hr)', fontsize=13)
plt.title('Flow density scatter plot (US 101 Lane 2)', fontsize=14)
plt.xlim([0, 150])
plt.ylim([50, 4500])
plt.grid()


# =============================================================================
# Generate dataset for whole time period
# =============================================================================

# more parameters
l_dn = 80
l_up = 40

# Trajectory dataset
num_samples = 25
datasets = [df]
datasets_name = ["Ngsim_us101_lane2_20p"]

# Define a mapping function
colmap = cmg.MplColorMapper(cmap_name='jet_r', min_val=min_speed, 
                            max_val=max_speed, lookuptable_dim=256)

for i in range(len(datasets)):
    
    traj_data = datasets[i]
    df_name = datasets_name[i]

    # Saver
    x_num = 80
    t_num = 2760 #960 #2760 #1810
    x_num_act = params['x_num']
    inp_spacetime = np.zeros((num_samples, x_num, t_num, 3))
    out_spacetime = np.zeros((1, x_num, t_num, 1))
    
    # Full vehicle trajectories
    # full_vehs = traj_data.VehNo.unique()
    # full_coords = getProbeCoords(traj_data, full_vehs, t_int=1)
    # localSpeed = localSpeedfield(full_coords, params)
    # out_spacetime[0,:x_num_act,:,0] = np.round(localSpeed/max_speed, 4)

    # Sample *num_samples* probe trajectories from whole dataset
    for j in range(num_samples):
        
        print('Sample number: ', j)
        probe_vehs = sampleProbes(traj_data, probe_per=20, load=False)
        probe_coords = getProbeCoords(traj_data, probe_vehs, t_int=1)
        probe_grid, occ_grid = genProbeTimeSpace(probe_coords, params, veh_len=5)
        inp_spacetime[j,:x_num_act,:,:] = probe_grid
        
    # Generate training data
    noempty_indx = np.argwhere(inp_spacetime.reshape(inp_spacetime.shape[0],-1).sum(axis=1) != 0)
    np.save('./inp_data_{}.npy'.format(df_name), inp_spacetime[noempty_indx.reshape(-1),:,:,:])
    # np.save('./out_data_{}.npy'.format(df_name), out_spacetime[:,:,:,:])

t=np.random.choice(noempty_indx.reshape(-1))
# plt.figure()
# plt.imshow(out_spacetime[0,:,:].reshape(80,t_num), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
plt.figure()
plt.imshow(inp_spacetime[t,:,:,:], cmap='jet_r', vmin=0, vmax=1, aspect='auto')
