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
import ColorMapping as cmg
import pickle as pkl

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

def readData(vr_filename, link_ids, lane_ids, params, vr_folder="1_Simulations and raw data\\Simulated"):
    '''
    Read the probe vehicle trajectory dataset
    '''
    # Unpack some parameters
    time_start = params['time_start']
    time_end = params['time_end']
    # sec_len = int(params['sec_end']-params['sec_start'])
    # Read trajectory data
    vr_column_names = ['SimSec','TimeInNet','VehNo','LinkNo','LaneInd','Pos','LatPos','CoordFront','CoordRear','Length','Speed','Acc','TotDistTrav','FollowDist','Hdwy']
    traj_df = pd.read_csv(os.path.join(os.getcwd(), vr_folder, vr_filename), 
                          skiprows=30, sep=';', names=vr_column_names)
    # Preprocess and filter
    traj_df.SimSec = traj_df.SimSec.apply(lambda x: np.around(x))
    traj_data = traj_df[traj_df['SimSec'] >= time_start]
    traj_data = traj_data[traj_data['SimSec'] < time_end]
    traj_data = traj_data[traj_data['LinkNo'].isin(link_ids)]
    traj_data = traj_data[traj_data['LaneInd'].isin(lane_ids)]
    for i, l_id in enumerate(link_ids[1:]):
        traj_data.loc[traj_data['LinkNo']==l_id, 'Pos'] += sum(sec_lens[:i])
    # traj_data = traj_data[traj_data['Pos'] < sec_len]
    
    return traj_data

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

def getVehTrajectories(traj_data, veh_ids, params):
    '''
    '''
    veh_trajs = {}
    for v_id in veh_ids:
        df1 = traj_data.query('VehNo==@v_id')
        time_stmp = df1.SimSec.to_numpy()
        time_stmp = time_stmp - params['time_start']
        location = df1.Pos.to_numpy()
        speed = df1.Speed.to_numpy()
        veh_trajs[v_id] = [time_stmp, location, speed]
        
    return veh_trajs

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
                    
                    veh_pos = df1.loc[df1['VehNo'] == probe_veh, 'Pos'].item()
                    veh_speed = df1.loc[df1['VehNo'] == probe_veh, 'Speed'].item()
                    veh_density = df1.loc[df1['VehNo'] == probe_veh, 'Density'].item()
                    probe_coords.append((sim_sec, veh_pos, 
                                         np.around(veh_speed, 2), 
                                         np.around(veh_density, 2) ))
    
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

def genProbeDensityTimeSpace(probe_coords, params, colmap, veh_len=4):
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
    probe_coords = pd.DataFrame(probe_coords[:,[0,1,3]], columns=['time','sec','density'])
    
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
            
            p_s = rec['density']                    # Local density of probe veh
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
                c_density, occupiedCells = cellSpeedCalc(u_ix,  p_s, p_occup, occupiedCells)
                probe_grid[u_ix, t_id, :] = colmap.get_rgb(c_density)[:3]
                aux_grid[u_ix, t_id, :] = [1, c_density]
            
            elif p_x_b >= d_x-x_int/2:
                # update downstream cell if vehicle only occupy downstream cell
                p_occup = round(veh_len/x_int, 3)
                c_density, occupiedCells = cellSpeedCalc(d_ix, p_s, p_occup, occupiedCells)
                probe_grid[d_ix, t_id, :] = colmap.get_rgb(c_density)[:3]
                aux_grid[d_ix, t_id, :] = [1, c_density]
            
            else:
                # update downstream and upstream cell if vehicle occupy both cells
                p_occup = round((u_x+x_int/2-p_x_b)/x_int, 3)
                c_density, occupiedCells = cellSpeedCalc(u_ix,  p_s, p_occup, occupiedCells)
                probe_grid[u_ix, t_id, :] = colmap.get_rgb(c_density)[:3]
                aux_grid[u_ix, t_id, :] = [1, c_density]
                
                p_occup = round((p_x_f-d_x+x_int/2)/x_int, 3)
                c_density, occupiedCells = cellSpeedCalc(d_ix, p_s, p_occup, occupiedCells)
                probe_grid[d_ix, t_id, :] = colmap.get_rgb(c_density)[:3]
                aux_grid[d_ix, t_id, :] = [1, c_density]
    
    return probe_grid, aux_grid

def getVehicleCrossingTimes(traj_data, params):
    '''
    Extract cross times of vehicle trajectories at the section mid-points
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
    
    # Generate space-time grid coordinates
    grid_coords = genCoordinates(time_start, time_end, sec_start, sec_end, 
                                 t_int, x_int, t_num, x_num)
    
    # Store vehicle passing times at each section using a dictonary
    crossTimes = {}
    sec_lattice = grid_coords[0, :, 1]
    print('Retrieving the passing times...')
    
    # Loop through each cell section
    for x_i in range(sec_lattice.shape[0]):
    
        # Cell section mid-point
        print('\tSection: ', x_i)
        x = sec_lattice[x_i]
        cross_times = []
    
        # Consider trajectories locally around the section
        x_u = x+2*x_int
        x_d = x-2*x_int
        cell_traj = traj_data.query('Pos<@x_u')
        cell_traj = cell_traj.query('Pos>=@x_d')
        VehIDs = sorted(cell_traj.VehNo.unique())
        
        # Loop through each vehicles
        for veh_i in VehIDs:
            
            # Vehicle i trajectory record
            df = cell_traj.query('VehNo==@veh_i')
            
            # Time and location just before the section
            try:
                t_bef = df[df['Pos'] <= x]['SimSec'].to_list()[-1]
                x_bef = df[df['Pos'] <= x]['Pos'].to_list()[-1]
            except:
                # For the first cell
                if x_i == sec_lattice[0]:
                    continue
                # For the last cell
                elif x_i == sec_lattice[-1]:
                    continue
                # For other cells (possibly due to lane change)
                else:
                    continue
        
            # Time and location just after the section
            try:
                t_aft = df[df['Pos'] > x]['SimSec'].to_list()[0]
                x_aft = df[df['Pos'] > x]['Pos'].to_list()[0]
            except:
                # For the first cell
                if x_i == sec_lattice[0]:
                    continue
                # For the last cell
                elif x_i == sec_lattice[-1]:
                    continue
                # For other cells (possibly due to lane change)
                else:
                    continue
            
            # Interpolate the crossing time of vehicle i (for later: weight by corresponding speed)
            t_x = np.round(t_bef + (t_aft-t_bef)*(x-x_bef)/(x_aft-x_bef), 4)
            cross_times.append(t_x)
        crossTimes[x_i] = cross_times
    
    return crossTimes, grid_coords

def getVehicleSpacings(traj_data, params):
    '''
    Extract vehicle spacing (distance headway) at each time step from the
    trajectory data.
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
    
    # Generate space-time grid
    grid_coords = genCoordinates(time_start, time_end, sec_start, sec_end, 
                                 t_int, x_int, t_num, x_num)
    
    # Vehicle positions at each time step is recorded in a dictionary
    vehSpacings = {}
    time_lattice = grid_coords[:, 0, 0]
    print('Retrieving vehicle positions at each time step...')
    
    # For each time interval
    for t_i in range(len(time_lattice)):
        
        # Filter and sort vehicle positions
        t = time_lattice[t_i]
        sec_traj = traj_data.query('SimSec==@t')
        veh_pos = sorted(sec_traj['Pos'].to_list())
        
        # Save it
        vehSpacings[t_i] = veh_pos
        
    return vehSpacings, grid_coords

def getGridFlow(crossTimes, grid_coords, params):
    '''
    Calculate flow over the space-time grid (Euler representation)
    '''
    # Unpack some parameters
    time_start = params['time_start']
    time_end = params['time_end']
    t_int = params['t_int']
    t_num = params['t_num']
    x_num = params['x_num']
    
    # Initialize
    time_lattice = grid_coords[:, 0, 0]
    flow_grid = np.zeros((x_num, t_num))
    time_headway_grid = np.zeros((x_num, t_num))
    print('Retrieving time headway and flow during each time interval...')
    
    # For each section cell
    for x_i in crossTimes.keys():
        
        # Get the crossing times; add start and end time
        cross_times = [time_start-5*t_int] + sorted(crossTimes[x_i]) + [time_end+5*t_int]
        
        # Get the time_headway
        flow = []
        time_headway = []
        for t_i in range(len(cross_times)-1):
            h = cross_times[t_i+1] - cross_times[t_i]
            time_headway.append(round(h, 4))
            flow.append(round((1/h)*3600, 4))
        
        # For each time interval
        for t_i in range(len(time_lattice)):
            
            # Get the time headway and flow during this time interval
            t = time_lattice[t_i]
            t_h = time_headway[np.argwhere(t<=cross_times)[0].item()-1]
            t_f = flow[np.argwhere(t<=cross_times)[0].item()-1]
            
            # Save it
            time_headway_grid[x_i, t_i] = t_h
            flow_grid[x_i, t_i] = t_f
    
    return time_headway_grid, flow_grid

def getGridDensity(vehSpacings, grid_coords, params):
    '''
    Calculate density over the space-time grid (Euler representation)
    '''
    # Unpack some parameters
    sec_start = params['sec_start']
    sec_end = params['sec_end']
    x_int = params['x_int']
    t_num = params['t_num']
    x_num = params['x_num']
    
    # Initialize
    sec_lattice = grid_coords[0, :, 1]
    density_grid = np.zeros((x_num, t_num))
    space_headway_grid = np.zeros((x_num, t_num))
    
    # For each time step
    for t_i in vehSpacings.keys():
        
        # Get vehicle spacings; add start and end section
        spacings = [sec_start-x_int] + vehSpacings[t_i] + [sec_end+x_int]
        
        # Get the space headways at time t
        density = []; space_headway = []
        for x_i in range(len(spacings)-1):
            s = spacings[x_i+1] - spacings[x_i]
            space_headway.append(round(s, 4))
            density.append(round((1/s)*1000, 4))
            
        # For each cell section
        for x_i in range(len(sec_lattice)):
            
            # Get the space headway and density during this time interval
            x = sec_lattice[x_i]
            x_s = space_headway[np.argwhere(x<=spacings)[0].item()-1]
            x_d = density[np.argwhere(x<=spacings)[0].item()-1]
            
            # Save it
            space_headway_grid[x_i, t_i] = x_s
            density_grid[x_i, t_i] = x_d
        
    return space_headway_grid, density_grid

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
    
    inp_spacetime = np.zeros((probe_grid.shape[1]-t_num, x_num, t_num, probe_grid.shape[-1]))
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

# Parameters used
link_ids = [4,10007,10,10006,12,10009,13]
sec_lens=[811.136,5.331,93.552,3.158,136.941,3.712,2027.766]
lane_ids = [3,3,5,4,4,3,3]
min_speed = 0
max_speed = 100
t_int = 1
x_int = 10
sec_start = 0
sec_end = 800
time_start = 600 #72000 #600
time_end =  7800 #86400 #7800
time_len = int(time_end - time_start)
sec_len = int(sec_end - sec_start)
t_num = int((time_end-time_start)/t_int)
x_num = int(sec_len/x_int)
params = { 't_int':t_int, 'x_int':x_int, 'sec_start':sec_start, 'sec_end':sec_end, 
           'time_start':time_start, 'time_end':time_end, 't_num':t_num, 'x_num':x_num }

df_wide = readData("Abudhabi_Alain_Road_Wide cong.fzp", link_ids, lane_ids, params)
df = df_wide

# adding density
l_dn = 80 #60 # 80
l_up = 40 #30 # 40
min_speed = 0
max_speed = 100
min_density = 0
max_density = 155

veh_headway = df['Hdwy'].to_numpy().copy()
loc_density = np.round(1e6/veh_headway, 4)
loc_density[loc_density > max_density] = max_density
loc_density[loc_density < min_density] = min_density
df['Density'] = loc_density

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
# Generate training dataset
# =============================================================================

# more parameters
l_dn = 80 # Car-following influence limit downstream of vehicle
l_up = 40 # Car-following influence limit upstream of vehicle
max_speed = 100

# Trajectory dataset
datasets = [df]
datasets_name = ["Wide_cong_more"]

# Define a mapping function
colmap = cmg.MplColorMapper(cmap_name='jet_r', min_val=min_speed, 
                            max_val=max_speed, lookuptable_dim=256)

# Generate dataset
for i in range(len(datasets)):
    
    traj_data = datasets[i]
    df_name = datasets_name[i]

    # Sample probe trajectories from whole dataset
    probe_vehs = sampleProbes(traj_data, probe_per=5, load=False)
    probe_coords = getProbeCoords(traj_data, probe_vehs, t_int=1)
    probe_grid, occ_grid = genProbeTimeSpace(probe_coords, params, colmap, veh_len=5)
    
    # Full vehicle trajectories
    full_vehs = traj_data.VehNo.unique()
    full_coords = getProbeCoords(traj_data, full_vehs, t_int=1)
    full_grid, full_occ_grid = genProbeTimeSpace(full_coords, params, colmap, veh_len=5)
    localSpeed = localSpeedfield(full_coords[:,:3], params, max_speed)
    
    # Spatio-temporal traffic flow and density
    # crossTimes, grid_coords = getVehicleCrossingTimes(traj_data, params)
    # th_grid, fl_grid = getGridFlow(crossTimes, grid_coords, params)
    # vehSpacings, grid_coords = getVehicleSpacings(traj_data, params)
    # dh_grid, de_grid = getGridDensity(vehSpacings, grid_coords, params)
    # sp_grid = np.divide(fl_grid, de_grid)
    # sp_grid = np.clip(sp_grid, 0, max_speed)
    
    # Generate training data
    inp_spacetime, out_spacetime = genTrainingData(probe_grid, localSpeed, 'B')
    noempty_indx = np.argwhere(inp_spacetime.reshape(inp_spacetime.shape[0],-1).sum(axis=1) != 0)
    np.save('./inp_data_{}.npy'.format(df_name), inp_spacetime[noempty_indx.reshape(-1),:,:,:])
    np.save('./out_data_{}.npy'.format(df_name), out_spacetime[noempty_indx.reshape(-1),:,:,:])
    
    # Save occupancy data
    # inp_occ, _ = genTrainingData(occ_grid, localSpeed, 'B')
    # np.save('./probe_occ_{}.npy'.format(df_name), inp_occ[noempty_indx.reshape(-1),:,:,:])
    out_occ, _ = genTrainingData(full_occ_grid, localSpeed, 'B')
    np.save('./traj_occ_{}.npy'.format(df_name), out_occ[noempty_indx.reshape(-1),:,:,:])


# Some analysis on generated dataset
# Visualize
t=np.random.choice(noempty_indx.reshape(-1))
plt.figure()
plt.imshow(out_spacetime[t,:,:].reshape(80,60), cmap='jet_r', vmin=0, vmax=1)
plt.figure()
plt.imshow(inp_spacetime[t,:,:,:], cmap='jet_r', vmin=0, vmax=1)
plt.figure()
plt.imshow(full_grid[:,t:t+60,:], cmap='jet_r', vmin=0, vmax=100)

# Visualize the occupancy
occ_grid = np.array(full_occ_grid[:,:,0], dtype=np.int)
plt.figure(figsize=(7,6))
plt.imshow(occ_grid, cmap='Blues')
plt.xlabel("Time (sec)", fontsize=12)
plt.ylabel("Space (m)", fontsize=12)
plt.title("Occupied cells", fontsize=14)
plt.show()

# Speed distribution in occupied cells, trajectory and all cells
occ_speed_grid = np.multiply(localSpeed, occ_grid)
occ_speed = []
for i in range(len(occ_speed_grid.reshape(-1))):
    if occ_grid.reshape(-1)[i] == 1:
        occ_speed.append(occ_speed_grid.reshape(-1)[i])

plt.figure(figsize=(8,6))
plt.hist([localSpeed.reshape(-1), occ_speed, full_coords[:,2]],
          bins=30, rwidth=0.8, alpha=1.0, stacked=True, density=False)
plt.xlabel("Vehicle speed (kmph)", fontsize=12)
plt.ylabel("Counts", fontsize=12)
plt.title("Speed distributions", fontsize=14)
plt.legend(['Full cells','Occupied cells','Trajectory'], fontsize=10)
plt.grid()


# =============================================================================
# Generate test dataset
# =============================================================================

# test data parameters
test_time_start = 90000
test_time_end = 93600
test_params = params
test_params['time_start'] = test_time_start
test_params['time_end'] = test_time_end

# Filter data
# test_traj_data = traj_df[traj_df['SimSec'] >= test_time_start]
# test_traj_data = test_traj_data[test_traj_data['SimSec'] < test_time_end]
# test_traj_data = test_traj_data[test_traj_data['LinkNo'] == link_id]
# test_traj_data = test_traj_data[test_traj_data['LaneInd'] == lane_id]
# test_traj_data = test_traj_data[test_traj_data['Pos'] < sec_len]
df_test_old = readData("Abudhabi_Alain_Road_Vehicle Record (20-30).fzp", link_id, lane_id, test_params)


# Sample probe trajectories from whole dataset
probe_vehs = sampleProbes(df_test_old, probe_per=5, load=False)
probe_coords = getProbeCoords(df_test_old, probe_vehs, t_int=1)
probe_grid, occ_grid = genProbeTimeSpace(probe_coords, test_params, veh_len=5)

# Full vehicle trajectories
full_vehs = df_test_old.VehNo.unique()
full_coords = getProbeCoords(df_test_old, full_vehs, t_int=1)
# full_grid, full_occ_grid = genProbeTimeSpace(full_coords, test_params, veh_len=5)

# Generate training data
inp_spacetime, out_spacetime = genTrainingData(probe_grid, full_grid)
np.save('./test_inp_data.npy', inp_spacetime)
np.save('./test_out_data.npy', out_spacetime)
np.save('./test_probe_grid.npy', occ_grid)
np.save('./test_whole_grid.npy', full_occ_grid)


# =============================================================================
# Generate data with vehicle spacing (Density)
# =============================================================================

plt.figure()
plt.hist(df.FollowDist.to_numpy()/1000, bins=25, rwidth=0.8, density=True)
plt.xlabel('Spacing (m)', fontsize=12)
plt.title('Congested data', fontsize=13)
plt.grid()

plt.figure()
plt.hist(df.FollowDist.to_numpy()/1000, bins=25, rwidth=0.8, density=True)
plt.xlabel('Spacing (m)', fontsize=12)
plt.title('Free flow data', fontsize=13)
plt.grid()

plt.figure()
plt.scatter(df.SimSec, df.Pos, c=df.Density, s=3, cmap='jet', vmin=min_density, vmax=max_density)
plt.colorbar()

# Params
max_speed = 100
speed_95per = 100

# Trajectory dataset
datasets = [df, df, df, df, df, df]
datasets_name = ["Wide_free_20p1","Wide_free_20p2","Wide_free_20p3",
                 "Wide_free_20p4","Wide_free_20p5","Wide_free_20p6"]

# Define a mapping function
sp_colmap = cmg.MplColorMapper(cmap_name='jet_r', min_val=min_speed, 
                               max_val=max_speed, lookuptable_dim=256)
dn_colmap = cmg.MplColorMapper(cmap_name='jet', min_val=min_density, 
                               max_val=max_density, lookuptable_dim=256)

for i in range(len(datasets)):
    
    traj_data = datasets[i]
    df_name = datasets_name[i]

    # Sample probe trajectories from whole dataset
    probe_vehs = sampleProbes(traj_data, probe_per=20, load=False)
    probe_coords = getProbeCoords(traj_data, probe_vehs, t_int=1)
    sp_probe_grid, occ_grid = genProbeTimeSpace(probe_coords, params, sp_colmap, veh_len=5)
    dn_probe_grid, occ_grid = genProbeDensityTimeSpace(probe_coords, params, dn_colmap, veh_len=5)
    probe_grid = np.concatenate((sp_probe_grid, dn_probe_grid), axis=2)
    
    # Full vehicle trajectories
    if i == 0:
        full_vehs = traj_data.VehNo.unique()
        full_coords = getProbeCoords(traj_data, full_vehs, t_int=1)
        # full_grid, full_occ_grid = genProbeTimeSpace(full_coords, params, sp_colmap, veh_len=5)
        localSpeed = localSpeedfield(full_coords[:,:3], params, speed_95per)
    
    # Generate training data
    inp_spacetime, out_spacetime = genTrainingData(probe_grid, localSpeed, 'B')
    noempty_indx = np.argwhere(inp_spacetime.reshape(inp_spacetime.shape[0],-1).sum(axis=1) != 0)
    np.save('./inp_data_{}.npy'.format(df_name), inp_spacetime[noempty_indx.reshape(-1),:,:,:])
    np.save('./noempty_indx_{}.npy'.format(df_name), noempty_indx)
    if i == 0:
        np.save('./out_data_{}.npy'.format(df_name[:-1]), out_spacetime)


t=np.random.choice(noempty_indx.reshape(-1))
plt.figure()
plt.imshow(out_spacetime[t,:,:].reshape(80,60), cmap='jet_r')
plt.colorbar()
plt.figure()
plt.imshow(inp_spacetime[t,:,:,:3], cmap='jet_r', vmin=0, vmax=1)
plt.colorbar()
plt.figure()
plt.imshow(inp_spacetime[t,:,:,3:], cmap='jet', vmin=0, vmax=1)
plt.colorbar()
plt.figure()
plt.imshow(full_grid[:,t:t+60,:], cmap='jet_r', vmin=0, vmax=100)


plt.figure()
plt.imshow(probe_grid[:,:,:3], cmap='jet_r', vmin=0, vmax=1)
plt.colorbar()
plt.figure()
plt.imshow(probe_grid[:,:,3:], cmap='jet', vmin=0, vmax=1, aspect='auto')
plt.colorbar()
plt.figure()
plt.imshow(localSpeed, cmap='jet_r', vmin=0, vmax=100)
plt.colorbar()

# =============================================================================
# Eulerian coords to Lagrangian trajectory analysis
# =============================================================================

# Parameters
l_dn = 80
l_up = 40
max_speed = 100
dt_name = 'Wide_free'
traj_data = df

# Vehicle trajectory (lagrangian coords)
full_vehs = traj_data.VehNo.unique()
veh_trajs = getVehTrajectories(traj_data, full_vehs, params)
out_file = open('veh_traj_{}.pkl'.format(dt_name), 'wb')
pkl.dump(veh_trajs, out_file)
out_file.close()

# Input vehicle trajectories (eulerian coords)
colmap = cmg.MplColorMapper(cmap_name='jet_r', min_val=min_speed, 
                            max_val=max_speed, lookuptable_dim=256)
probe_vehs = sampleProbes(traj_data, probe_per=5, load=False)
probe_coords = getProbeCoords(traj_data, probe_vehs, t_int=1)
probe_grid, occ_grid = genProbeTimeSpace(probe_coords, params, colmap, veh_len=5)
np.save('./inp_map_{}.npy'.format(dt_name), probe_grid)

# Output speed map (eulerian coords)
full_vehs = traj_data.VehNo.unique()
full_coords = getProbeCoords(traj_data, full_vehs, t_int=1)
localSpeed = localSpeedfield(full_coords[:,:3], params)
np.save('./out_map_{}.npy'.format(dt_name), localSpeed)
