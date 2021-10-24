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

def readData(vr_filename, link_id, lane_id, params, vr_folder="1_Simulations and raw data\\HighD dataset\\data"):
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
    df3['Speed'] = df2['v_Vel']*1.09728 # Convert to metre/sec
    
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

def genProbeTimeSpace(probe_coords, params, colmap, veh_len=4):
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
    probe_coords = pd.DataFrame(probe_coords[:,:3], columns=['time','sec','speed'])
    
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

def localSpeedfield(full_coords, params, max_speed):
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
    x_num = params['x_num']
    
    inp_spacetime = np.zeros((probe_grid.shape[1]-t_num, x_num, t_num, 3))
    for t in range(inp_spacetime.shape[0]):
        inp_spacetime[t,:,:,:] = probe_grid[:,t:t+t_num, :]
    
    if method=='A':
        out_spacetime = np.zeros((probe_grid.shape[1]-t_num, x_num, t_num, 3))
        for t in range(inp_spacetime.shape[0]):
            out_spacetime[t,:,:,0] = np.flipud(full_grid[t:t+t_num, :, :].T[0,:,:])
            out_spacetime[t,:,:,1] = np.flipud(full_grid[t:t+t_num, :, :].T[1,:,:])
            out_spacetime[t,:,:,2] = np.flipud(full_grid[t:t+t_num, :, :].T[2,:,:])
    elif method=='B':
        out_spacetime = np.zeros((probe_grid.shape[1]-t_num, x_num, t_num, 1))
        for t in range(inp_spacetime.shape[0]):
            out_spacetime[t,:,:,0] = np.round(full_grid[:,t:t+t_num]/max_speed, 4)
    
    return inp_spacetime, out_spacetime

# =============================================================================
# Read and process the data
# =============================================================================

'''
Meta information about HighD dataset:
    
for i in range(1,61,1):
    rec_filename = "{}_recordingMeta.csv".format((str(0)+str(i))[-2:])
    if i == 1:
        df = pd.read_csv(os.path.join(os.getcwd(), vr_folder, rec_filename))
    else:
        df_temp = pd.read_csv(os.path.join(os.getcwd(), vr_folder, rec_filename))
        df = pd.concat([df, df_temp])
        
'''

# File name and folder
track_no = 44
highway_no = '0'+str(track_no) if len(str(track_no)) !=2 else str(track_no)
vr_folder="1_Simulations and raw data\\HighD dataset\\data"
vr_filename = "{}_tracks.csv".format(highway_no)
meta_filename = "{}_tracksMeta.csv".format(highway_no)

# Loading meta data
df = pd.read_csv(os.path.join(os.getcwd(), vr_folder, meta_filename))
vehs_left = df.query('drivingDirection==1')['id'].to_list()
vehs_right = df.query('drivingDirection==2')['id'].to_list()

# Loading trajectory data
lane_id = '6'
traj_df = pd.read_csv(os.path.join(os.getcwd(), vr_folder, vr_filename))
df_right = traj_df[traj_df['id'].map(lambda x: x in vehs_right)]
df_left = traj_df[traj_df['id'].map(lambda x: x in vehs_left)]
df_lane = df_right.query('laneId==@lane_id')

# Speed distributions
speeds = np.round(np.abs(df_lane.xVelocity)*18/5, 3)
plt.figure(figsize=(7,6))
plt.hist(speeds, bins=30, rwidth=0.8, alpha=0.7, density=True)
plt.xlabel("Vehicle speed (kmph)", fontsize=12)
plt.ylabel("Density distribution", fontsize=12)
plt.title("Speed distributions", fontsize=14)
plt.legend(['One class'])
plt.grid()
plt.show()

plt.figure(figsize=(8,6))
for veh_id in df_lane.id.unique():
    df2 = df_lane.query('id==@veh_id')
    df3 = df2[df2['frame'].map(lambda x: x%25==0)]
    veh_time = df3.frame
    veh_pos = df3.x
    veh_speed = np.round(np.abs(df3.xVelocity)*18/5, 3)
    plt.scatter(veh_time, veh_pos, s=15, c=veh_speed, 
                cmap='jet_r', vmin=0, vmax=max(speeds))
plt.colorbar()


# =============================================================================
# Generate testing dataset (Whole plane)
# =============================================================================

# Forming the dataset in the required form
frame_rate = 25
df_processed = pd.DataFrame(columns=['SimSec','VehNo','Pos','Speed','Density'])
df_lane_persec = df_lane[df_lane['frame'].map(lambda x: x%frame_rate==0)]
df_processed['SimSec'] = np.array(df_lane_persec['frame'].to_numpy()/frame_rate, dtype=np.int32)
df_processed['VehNo'] = df_lane_persec['id'].to_numpy()
# df_processed['Pos'] = df_lane_persec['x'].max() - df_lane_persec['x'].to_numpy() # for df_left
df_processed['Pos'] = df_lane_persec['x'].to_numpy() # for df_right
df_processed['Speed'] = np.round(np.abs(df_lane_persec['xVelocity'].to_numpy())*18/5, 3)
df_processed['Density'] = df_lane_persec['dhw'].to_numpy()


# Visualize vehicle trajectories with speeds
plt.figure(figsize=(8,6))
for veh_id in df_processed.VehNo.unique():
    df2 = df_processed.query('VehNo==@veh_id')
    veh_time = df2.SimSec
    veh_pos = df2.Pos
    veh_speed = df2.Speed
    plt.scatter(veh_time, veh_pos, s=15, c=veh_speed, 
                cmap='jet_r', vmin=0, vmax=max(df_processed.Speed))
plt.colorbar()

# Parameters used
speeds_sorted = sorted(df_processed.Speed.to_list())
speed_95per = speeds_sorted[np.int(0.95*len(speeds_sorted))]
max_speed = speeds_sorted[-1]
min_speed = 0
l_dn = 80; l_up = 40

t_int = 1 ; x_int = 10
sec_start = 0 ; sec_end = 400 ; time_start = 0 
time_end = int(np.floor(df_processed['SimSec'].max() / 60)) * 60
time_len = int(time_end - time_start)
sec_len = int(sec_end - sec_start)
t_num = int((time_end-time_start)/t_int)
x_num = int(sec_len/x_int)
params = { 't_int':t_int, 'x_int':x_int, 'sec_start':sec_start, 'sec_end':sec_end, 
           'time_start':time_start, 'time_end':time_end, 't_num':t_num, 'x_num':x_num }

# Define a mapping function
colmap = cmg.MplColorMapper(cmap_name='jet_r', min_val=min_speed, 
                            max_val=max_speed, lookuptable_dim=256)


# Generate dataset
num_samples = 100
datasets = [df_processed]
datasets_name = ["highD_Track{}_Lane{}".format(highway_no, lane_id)]

for i in range(len(datasets)):
    
    traj_data = datasets[i]
    df_name = datasets_name[i]

    # Saver
    x_num = 80
    t_num = 1140
    x_num_act = params['x_num']
    inp_spacetime = np.zeros((num_samples, x_num, t_num, 3))
    out_spacetime = np.zeros((1, x_num, t_num, 1))
    
    # Full vehicle trajectories
    full_vehs = traj_data.VehNo.unique()
    full_coords = getProbeCoords(traj_data, full_vehs, t_int=1)
    localSpeed = localSpeedfield(full_coords, params, speed_95per)
    out_spacetime[0,:x_num_act,:,0] = np.round(localSpeed/max_speed, 4)

    # Sample *num_samples* probe trajectories from whole dataset
    for j in range(num_samples):
        
        print('Sample number: ', j)
        # Sample probe trajectories from whole dataset
        probe_vehs = sampleProbes(traj_data, probe_per=5, load=False)
        probe_coords = getProbeCoords(traj_data, probe_vehs, t_int=1)
        probe_grid, occ_grid = genProbeTimeSpace(probe_coords, params, colmap, veh_len=5)
        inp_spacetime[j,:x_num_act,:,:] = probe_grid
        
    # Generate training data
    noempty_indx = np.argwhere(inp_spacetime.reshape(inp_spacetime.shape[0],-1).sum(axis=1) != 0)
    np.save('./inp_data_HighD_track{}_lane{}-5p.npy'.format(highway_no,lane_id), inp_spacetime[noempty_indx.reshape(-1),:,:,:])
    np.save('./out_data_HighD_track{}_lane{}-5p.npy'.format(highway_no,lane_id), out_spacetime[:,:,:,:])
    

# Visualize Eulerian grid
plt.figure(figsize=(12,6))
plt.imshow(probe_grid, cmap='jet_r', vmin=0, vmax=1, aspect='auto')
plt.ylabel('Sec (m)', fontsize=13)
plt.xlabel('Time (sec)', fontsize=13)
plt.title('Probe grid // Highway No {} // Lane {}'.format(highway_no, lane_id),fontsize=13)
plt.colorbar()
plt.savefig('PG_Hw{}_Lane{}_5p'.format(highway_no, lane_id), dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(12,6))
plt.imshow(full_grid, cmap='jet_r', vmin=0, vmax=1, aspect='auto')
plt.ylabel('Sec (m)', fontsize=13)
plt.xlabel('Time (sec)', fontsize=13)
plt.title('Probe grid // Highway No {} // Lane {}'.format(highway_no, lane_id),fontsize=13)
plt.colorbar()
plt.savefig('FG_Hw{}_Lane{}'.format(highway_no, lane_id), dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(12,6))
plt.imshow(np.round(localSpeed/max_speed, 3), cmap='jet_r', vmin=0, vmax=1, aspect='auto')
plt.ylabel('Sec (m)', fontsize=13)
plt.xlabel('Time (sec)', fontsize=13)
plt.title('Probe grid // Highway No {} // Lane {}'.format(highway_no, lane_id),fontsize=13)
plt.colorbar()
plt.savefig('LS_Hw{}_Lane{}'.format(highway_no, lane_id), dpi=300, bbox_inches = 'tight')

# =============================================================================
# Extract speed distributions and trajectories of all highway segments
# =============================================================================

import matplotlib
matplotlib.use('Agg')

for sec_id in range(1, 61):
    
    print('Highway no:', sec_id)
    
    # Load the highway data
    highway_no = '0'+str(sec_id) if len(str(sec_id)) !=2 else str(sec_id)
    vr_folder="1_Simulations and raw data\\HighD dataset\\data"
    vr_filename = "{}_tracks.csv".format(highway_no)
    meta_filename = "{}_tracksMeta.csv".format(highway_no)
    
    # Load meta data (details of vehicles recorded)
    df = pd.read_csv(os.path.join(os.getcwd(), vr_folder, meta_filename))
    vehs_left = df.query('drivingDirection==1')['id'].to_list()
    vehs_right = df.query('drivingDirection==2')['id'].to_list()
    
    # Load trajectory
    traj_df = pd.read_csv(os.path.join(os.getcwd(), vr_folder, vr_filename))
    df_right = traj_df[traj_df['id'].map(lambda x: x in vehs_right)]
    df_left = traj_df[traj_df['id'].map(lambda x: x in vehs_left)]
    
    # For each lane
    highway_speeds = []
    for lane_id in df_left.laneId.unique():
        df_lane = df_left.query('laneId==@lane_id')
        speeds = np.round(np.abs(df_lane.xVelocity)*18/5, 3)
        highway_speeds.append(speeds)
            
        # Plot speed trajectories
        plt.figure(figsize=(7,6))
        for veh_id in df_lane.id.unique():
            df2 = df_lane.query('id==@veh_id')
            df3 = df2[df2['frame'].map(lambda x: x%25==0)]
            veh_time = df3.frame
            veh_pos = df3.x
            veh_speed = np.round(np.abs(df3.xVelocity)*18/5, 3)
            plt.scatter(veh_time, veh_pos, c=veh_speed, cmap='jet_r', 
                        s=15, vmin=0, vmax=max(speeds))
        plt.xlabel('Time (fps)', fontsize=12)
        plt.ylabel('Position (m)', fontsize=12)
        plt.title('Highway: {0} // Right Direction // Lane: {1}'.format(highway_no, lane_id), fontsize=13)
        plt.colorbar()
        plt.tight_layout()
        
        file_name = 'speed trajectories\\Track_{} Lane_{}'.format(highway_no, lane_id)
        file_folder = os.path.join(os.getcwd(), vr_folder, file_name)
        plt.savefig(file_folder)
        plt.close()
        
    # Plot speed distributions
    plt.figure(figsize=(7,6))
    plt.hist(highway_speeds, bins=30, rwidth=0.8, alpha=0.7, density=True, stacked=True)
    plt.xlabel("Vehicle speed (kmph)", fontsize=12)
    plt.ylabel("Density distribution", fontsize=12)
    plt.title("Speed distributions", fontsize=14)
    plt.legend(df_right.laneId.unique().astype(str))
    plt.grid()
    
    file_name = 'speed distributions\\Track_{}'.format(highway_no)
    file_folder = os.path.join(os.getcwd(), vr_folder, file_name)
    plt.savefig(file_folder)
    plt.close()

# =============================================================================
# Analyse dataset
# =============================================================================

def process_data(df_lane):
    
    df_fd = pd.DataFrame(columns=['Time','VehNo','Pos','Speed','Time_headway','Dist_headway','Flow','Density'])
    time = []; vehno = []; pos = []; speed = []; thdwy = []; dhdwy = []; flow = []; density = []
    for veh_id in df_lane.id.unique():
        df_lane_1 = df_lane.query('id==@veh_id')
        df_lane_2 = df_lane_1[df_lane_1['frame'].map(lambda x: x%25==0)]
        time = time + df_lane_2.frame.to_list()
        vehno = vehno + list(np.repeat(veh_id, df_lane_2.shape[0]))
        pos = pos + df_lane_2.x.to_list()
        speed = speed + list(np.round(np.abs(df_lane_2.xVelocity.to_numpy())*18/5, 3))
        thdwy = thdwy + df_lane_2.thw.to_list()
        dhdwy = dhdwy + df_lane_2.dhw.to_list()
        flow = flow + list(np.round(3600/df_lane_2.thw.to_numpy(), 4))
        density = density + list(np.round(1e3/df_lane_2.dhw.to_numpy(), 4))
    df_fd['Time'] = time
    df_fd['VehNo'] = vehno
    df_fd['Pos'] = pos
    df_fd['Speed'] = speed
    df_fd['Time_headway'] = thdwy
    df_fd['Dist_headway'] = dhdwy
    df_fd['Flow'] = flow
    df_fd['Density'] = density
    
    return df_fd

# Processing dataset for each lanes
df_fds = []
lane_ids = [4]
for lane_id in lane_ids:
    df_lane = df_left.query('laneId==@lane_id')
    df_fd_lane = process_data(df_lane)
    df_fds.append(df_fd_lane)
df_fd_lane.to_csv('df_hw{}_lane{}.csv'.format(highway_no, lane_id), index=False)

# Empirical Flow-density-speed
num_samples = 1000
plt.figure(figsize=(8,6))
colors = ['tab:blue','tab:green','tab:orange','tab:red']
for df_i, df_fd in enumerate(df_fds):
    q = df_fd.Flow.to_numpy()
    k = df_fd.Density.to_numpy()
    rand_indx = np.random.randint(0, df_fd.Speed.shape[0], num_samples)
    x = k[rand_indx]
    y = q[rand_indx]
    y_filter = y[y <= 4300]
    x_filter = x[y <= 4300]
    plt.scatter(x_filter, y_filter, s=15, ec=colors[df_i], label=lane_ids[df_i], fc='none', lw=1, alpha=1.0)
plt.xlabel('Density (vehs/km)', fontsize=13)
plt.ylabel('Flow (vehs/hr)', fontsize=13)
plt.legend(fontsize=13)
plt.title('Flow density scatter plot (Highway {})'.format(highway_no), fontsize=14)
plt.xlim([0, 200])
# plt.ylim([0, 4500])
plt.grid()


# =============================================================================
# Extracting boundary conditions
# =============================================================================

# Plot trajectories
df1 = df_processed.query('SimSec >= 300')
df2 = df1.query('SimSec < 500')
plt.figure()
for v_id in df2.VehNo.unique():
    act_traj_veh = df2.query('VehNo == @v_id')
    t = act_traj_veh.SimSec.to_numpy()
    d = act_traj_veh.Pos.to_numpy()
    s = act_traj_veh.Speed.to_numpy()
    plt.scatter(t, d, c=s, s=6, cmap='jet_r', vmin=0, vmax=165)
    plt.plot(t, d, lw=0.3, c='k', alpha=0.4)
plt.colorbar()
plt.grid()

# Getting boundary condition (vehicle crossing times and speeds)
sec = 50
cross_times = []
speed_x0 = []
for v_id in df_processed.VehNo.unique():
    df_veh = df_processed.query('VehNo == @v_id')
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

HW44_BoundCond = {'x':sec,
                  't':cross_times,
                  'v':speed_x0}
with open('HW44_BoundCond.pkl', 'wb') as handle:
    pkl.dump(HW44_BoundCond, handle, protocol=pkl.HIGHEST_PROTOCOL)

df1 = df_processed.query('SimSec >= 0')
df2 = df1.query('SimSec < 1140')
with open('HW44_ActTraj.pkl', 'wb') as handle:
    pkl.dump(df2, handle, protocol=pkl.HIGHEST_PROTOCOL)
