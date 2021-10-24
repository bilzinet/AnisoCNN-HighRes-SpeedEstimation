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
    traj_data = traj_data[traj_data['LaneInd'].isin(lane_id)]
    traj_data = traj_data[traj_data['Pos'] < sec_len]
    
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
        sampled_vehs = np.random.choice(vehIDs, num_probes, replace=False)
        # np.save('./SampledVehIDs_{}per.npy'.format(probe_per), sampled_vehs)
    
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
    
    if method =='A':
        out_spacetime = np.zeros((probe_grid.shape[1]-t_num, x_num, t_num, 3))
        for t in range(inp_spacetime.shape[0]):
            out_spacetime[t,:,:,0] = np.flipud(full_grid[t:t+t_num, :, :].T[0,:,:])
            out_spacetime[t,:,:,1] = np.flipud(full_grid[t:t+t_num, :, :].T[1,:,:])
            out_spacetime[t,:,:,2] = np.flipud(full_grid[t:t+t_num, :, :].T[2,:,:])
    
    elif method =='B':
        out_spacetime = np.zeros((probe_grid.shape[1]-t_num, x_num, t_num, 1))
        for t in range(inp_spacetime.shape[0]):
            out_spacetime[t,:,:,0] = np.round(full_grid[:,t:t+t_num]/max_speed, 4)
    
    elif method == 'C':
        out_spacetime = np.zeros((probe_grid.shape[1]-t_num, x_num, t_num, full_grid.shape[-1]))
        for t in range(inp_spacetime.shape[0]):
            out_spacetime[t,:,:,:] = np.round(full_grid[:,t:t+t_num,:]/max_speed, 4)
    
    return inp_spacetime, out_spacetime

# =============================================================================
# Read and process the data
# =============================================================================

# Parameters used
link_id = 4
lane_id = [1,2]
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

df_wide = readData("Abudhabi_Alain_Road_Wide cong more.fzp", link_id, lane_id, params)

# adding density
l_dn = 80 #60 # 80
l_up = 40 #30 # 40
min_speed = 0
max_speed = 100
min_density = 0
max_density = 155

veh_headway = df_wide['Hdwy'].to_numpy().copy()
loc_density = np.round(1e6/veh_headway, 4)
# loc_density[loc_density > max_density] = max_density
loc_density[loc_density < min_density] = min_density
df_wide['Density'] = loc_density

# Speed distributions
df = df_wide.query('LaneInd==1')
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
# Generate training data with vehicle spacing (Density)
# =============================================================================

# Params
sps = sorted(df.Speed.to_list())
max_speed = 100
speed_85per = 90 #max(sps[int(len(sps)*0.85)], 80)

# Trajectory dataset
df = df_wide.query('LaneInd==1')
datasets = [df, df, df]
datasets_name = ["Wide_cong_more_sp60p","Wide_cong_more_sp60p","Wide_cong_more_sp60p"]

# Define a mapping function
sp_colmap = cmg.MplColorMapper(cmap_name='jet_r', min_val=min_speed, 
                               max_val=max_speed, lookuptable_dim=256)
dn_colmap = cmg.MplColorMapper(cmap_name='jet', min_val=min_density, 
                               max_val=max_density, lookuptable_dim=256)

for i in range(len(datasets)):
    
    traj_data = datasets[i]
    df_name = datasets_name[i]
    
    # Sample probe trajectories from whole dataset
    probe_vehs = sampleProbes(traj_data, probe_per=5, load=False)
    probe_coords = getProbeCoords(traj_data, probe_vehs, t_int=1)
    sp_probe_grid, occ_grid = genProbeTimeSpace(probe_coords, params, sp_colmap, veh_len=5)
    dn_probe_grid, occ_grid = genProbeDensityTimeSpace(probe_coords, params, dn_colmap, veh_len=5)
    probe_grid = np.concatenate((sp_probe_grid, dn_probe_grid), axis=2)
    
    # Full vehicle trajectories
    if i == 0:
        full_vehs = traj_data.VehNo.unique()
        full_coords = getProbeCoords(traj_data, full_vehs, t_int=1)
        full_grid, full_occ_grid = genProbeTimeSpace(full_coords, params, sp_colmap, veh_len=5)
        localSpeed = localSpeedfield(full_coords[:,:3], params, speed_85per)
    
    # Generate training data
    inp_spacetime, out_spacetime = genTrainingData(probe_grid, localSpeed, 'B')
    noempty_indx = np.argwhere(inp_spacetime.reshape(inp_spacetime.shape[0],-1).sum(axis=1) != 0)
    np.save('./inp_data_{}.npy'.format(df_name), inp_spacetime[noempty_indx.reshape(-1),:,:,:])
    np.save('./noempty_indx_{}.npy'.format(df_name), noempty_indx)
    if i == 0:
        np.save('./out_data_{}.npy'.format(df_name[:-1]), out_spacetime)

# t=np.random.choice(noempty_indx.reshape(-1))
# plt.figure()
# plt.imshow(out_spacetime[t,:,:].reshape(80,60), cmap='jet_r')
# plt.colorbar()
# plt.figure()
# plt.imshow(inp_spacetime[t,:,:,:3], cmap='jet_r', vmin=0, vmax=1)
# plt.colorbar()
# plt.figure()
# plt.imshow(inp_spacetime[t,:,:,3:], cmap='jet', vmin=0, vmax=1)
# plt.colorbar()
# plt.figure()
# plt.imshow(full_grid[:,t:t+60,:], cmap='jet_r', vmin=0, vmax=100)


# s=2200
# e=2300
# plt.figure()
# plt.imshow(probe_grid[:,s:e,:3], cmap='jet_r', vmin=0, vmax=1, aspect='auto')
# plt.colorbar()
# plt.figure()
# plt.imshow(full_grid[:,s:e,:3], cmap='jet_r', vmin=0, vmax=1, aspect='auto')
# plt.colorbar()
# plt.figure()
# plt.imshow(localSpeed[:,s:e], cmap='jet_r', vmin=0, vmax=100)
# plt.colorbar()

# s_coords = full_coords[full_coords[:,0] > s]
# e_coords = s_coords[s_coords[:,0] < e]
# plt.figure()
# plt.scatter(e_coords[:,0], e_coords[:,1], c=e_coords[:,2], cmap='jet_r')


# # Problem setting (for presentation)
# s=2200
# e=2300
# p = probe_grid[:,s:e,:3]
# p_bin = (p.sum(axis=2) != 0)
# p_alp = p_bin.astype(int)[:,:,None]
# p_whi = np.concatenate((p, p_alp), axis=-1)
# fig = plt.figure(figsize=(5,5))
# im1 = plt.imshow(p_whi, cmap='jet_r', vmin=0, vmax=1, aspect='auto')
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel('$Time$ [sec]', fontsize=14)
# # plt.ylabel('$Space$ [m]', fontsize=14)
# # plt.yticks([0,20,40,60,80])
# # ytick_locs, ytick_labels = plt.yticks()
# # ytick_labels = ['800','600','400','200','0']
# # plt.yticks(ytick_locs, ytick_labels, fontsize=14)
# # plt.xticks([0,20,40,60,80,100], fontsize=14)
# # plt.ylim([80,0])
# plt.tight_layout()

# f = full_grid[:,s:e,:3]
# f_bin = (f.sum(axis=2) != 0)
# f_alp = f_bin.astype(int)[:,:,None]
# f_whi = np.concatenate((f, f_alp), axis=-1)
# fig = plt.figure(figsize=(5,5))
# im1 = plt.imshow(f_whi, cmap='jet_r', vmin=0, vmax=1, aspect='auto')
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel('$Time$ [sec]', fontsize=14)
# # plt.ylabel('$Space$ [m]', fontsize=14)
# # plt.yticks([0,20,40,60,80])
# # ytick_locs, ytick_labels = plt.yticks()
# # ytick_labels = ['800','600','400','200','0']
# # plt.yticks(ytick_locs, ytick_labels, fontsize=14)
# # plt.xticks([0,20,40,60,80,100], fontsize=14)
# # plt.ylim([80,0])
# plt.tight_layout()

# v = localSpeed[:,s:e]
# fig = plt.figure(figsize=(5,5))
# im1 = plt.imshow(v, cmap='jet_r', vmin=0, vmax=100, aspect='auto')
# # plt.xticks([])
# # plt.yticks([])
# plt.xlabel('$Time$ [sec]', fontsize=14)
# plt.ylabel('$Space$ [m]', fontsize=14)
# plt.yticks([0,20,40,60,80])
# ytick_locs, ytick_labels = plt.yticks()
# ytick_labels = ['800','600','400','200','0']
# plt.yticks(ytick_locs, ytick_labels, fontsize=14)
# # plt.xticks([0,20,40,60,80,100], fontsize=14)
# plt.ylim([80,0])
# plt.tight_layout()


# =============================================================================
# Generate training data (no Density) - multiple lanes
# =============================================================================

# Params
sps = sorted(df_wide.Speed.to_list())
max_speed = 100
speed_85per = 90 #max(sps[int(len(sps)*0.85)], 80)

# Trajectory dataset
datasets = [df_wide, df_wide, df_wide]
datasets_name = ["Wide_cong_more_sp5p_2lanes","Wide_cong_more_sp5p_2lanes","Wide_cong_more_sp5p_2lanes"]

# Define a mapping function
sp_colmap = cmg.MplColorMapper(cmap_name='jet_r', min_val=min_speed, 
                               max_val=max_speed, lookuptable_dim=256)

localSpeed = []
for i in range(len(datasets)):
    
    traj_data_multi = datasets[i]
    df_name = datasets_name[i]
    
    probe_grid = []
    lanes = traj_data_multi.LaneInd.unique()
    for ld in lanes:
        traj_data = traj_data_multi.query('LaneInd==@ld')
        
        # Sample probe trajectories from whole dataset
        probe_vehs = sampleProbes(traj_data, probe_per=5, load=False)
        probe_coords = getProbeCoords(traj_data, probe_vehs, t_int=1)
        sp_probe_grid, occ_grid = genProbeTimeSpace(probe_coords, params, sp_colmap, veh_len=5)
        probe_grid.append(sp_probe_grid)
    
        # Full vehicle trajectories
        if i == 0:
            full_vehs = traj_data.VehNo.unique()
            full_coords = getProbeCoords(traj_data, full_vehs, t_int=1)
            full_grid, full_occ_grid = genProbeTimeSpace(full_coords, params, sp_colmap, veh_len=5)
            localSpeed_perlane = localSpeedfield(full_coords[:,:3], params, speed_85per)
            localSpeed.append(np.expand_dims(localSpeed_perlane, axis=-1))
    
    # Generate training data
    probe_grid = np.concatenate(probe_grid, axis=-1)
    if i == 0:
        localSpeed = np.concatenate(localSpeed, axis=-1)
    inp_spacetime, out_spacetime = genTrainingData(probe_grid, localSpeed, 'C')
    noempty_indx = np.argwhere(inp_spacetime.reshape(inp_spacetime.shape[0],-1).sum(axis=1) != 0)
    np.save('./inp_data_{}.npy'.format(df_name), inp_spacetime[noempty_indx.reshape(-1),:,:,:])
    np.save('./noempty_indx_{}.npy'.format(df_name), noempty_indx)
    if i == 0:
        np.save('./out_data_{}.npy'.format(df_name[:-1]), out_spacetime)



# =============================================================================
# Generate test dataset
# =============================================================================

# Test data parameters
link_id = 4; lane_id = 2
min_speed = 0; max_speed = 100; speed_95per = 90
l_dn = 80; l_up = 40; min_density = 0; max_density = 155
test_time_start = 7440; test_time_end = 7800; test_params = params
test_params['time_start'] = test_time_start; 
test_params['time_end'] = test_time_end


# Test data
df_test = readData("Abudhabi_Alain_Road_Wide free.fzp", link_id, lane_id, test_params)
veh_headway = df_test['Hdwy'].to_numpy().copy()
loc_density = np.round(1e6/veh_headway, 4)
loc_density[loc_density > max_density] = max_density
loc_density[loc_density < min_density] = min_density
df_test['Density'] = loc_density
df_name = "Wide_free"
df_nos = 6

# Define a mapping function
sp_colmap = cmg.MplColorMapper(cmap_name='jet_r', min_val=min_speed, 
                               max_val=max_speed, lookuptable_dim=256)
dn_colmap = cmg.MplColorMapper(cmap_name='jet', min_val=min_density, 
                               max_val=max_density, lookuptable_dim=256)

# Generate testing dataset (Eulerian coordinates)
for p in [5, 10, 20, 30, 40, 50, 60, 70, 80]:
    
    print('Probe: ', p)
    
    # For the required number of sampling
    for i in range(df_nos):
        
        print('\tDataset: ', i+1)
        traj_data = df_test
    
        # Sample probe trajectories from whole dataset
        probe_vehs = sampleProbes(traj_data, probe_per=p, load=False)
        probe_coords = getProbeCoords(traj_data, probe_vehs, t_int=1)
        sp_probe_grid, occ_grid = genProbeTimeSpace(probe_coords, test_params, sp_colmap, veh_len=5)
        dn_probe_grid, occ_grid = genProbeDensityTimeSpace(probe_coords, test_params, dn_colmap, veh_len=5)
        probe_grid = np.concatenate((sp_probe_grid, dn_probe_grid), axis=2)
        np.save('./inp_map_{}_sp{}p{}.npy'.format(df_name, str(p), str(i+1)), probe_grid)
        
        if (p == 5) and (i == 0):
            full_vehs = traj_data.VehNo.unique()
            full_coords = getProbeCoords(traj_data, full_vehs, t_int=1)
            localSpeed = localSpeedfield(full_coords[:,:3], test_params, speed_95per)
            np.save('./out_map_{}.npy'.format(df_name), localSpeed)
        
        # Generate testing data
        inp_spacetime, out_spacetime = genTrainingData(probe_grid, localSpeed, 'B')
        noempty_indx = np.argwhere(inp_spacetime.reshape(inp_spacetime.shape[0],-1).sum(axis=1) != 0)
        np.save('./inp_data_{}_sp{}p{}.npy'.format(df_name, str(p), str(i+1)), inp_spacetime[noempty_indx.reshape(-1),:,:,:])
        np.save('./noempty_indx_{}_sp{}p{}.npy'.format(df_name, str(p), str(i+1)), noempty_indx)
        
        # Full vehicle trajectories
        if (p == 5) and (i == 0):
            np.save('./out_data_{}.npy'.format(df_name), out_spacetime)
            

# Generate vehicle trajectory data (lagrangian coordinates)
full_vehs = traj_data.VehNo.unique()
veh_trajs = getVehTrajectories(traj_data, full_vehs, test_params)
out_file = open('veh_traj_{}.pkl'.format(df_name), 'wb')
pkl.dump(veh_trajs, out_file)
out_file.close()


# =============================================================================
# Eulerian coords to Lagrangian trajectory analysis
# =============================================================================

# Parameters
l_dn = 80
l_up = 40
max_speed = 100
speed_95per = 90
dt_name = 'Wide_cong more'
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
localSpeed = localSpeedfield(full_coords[:,:3], params, speed_95per)
np.save('./out_map_{}.npy'.format(dt_name), localSpeed)


plt.figure(figsize=(12,10))
plt.imshow(localSpeed, cmap='jet_r', vmin=0, vmax=100)
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Section length (m)', fontsize=12)
plt.title('Traffic speeds (1m x 1sec)', fontsize=13)


# =============================================================================
# Analyse the dataset
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

# Flow - density relationship
labels = ['Slow moving data', 'Free flowing data', 'Congested data']
colors = ['tab:blue','tab:green','tab:orange']
plt.figure(figsize=(8,6))
for i, df in enumerate([df_cong, df_free, df_cong_more]):
    num_samples = 1000
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
plt.xlim([0, 160])
plt.ylim([0, 4000])
plt.grid()

# Flow - density relationship
labels = ['Slow moving data', 'Free flowing data', 'Congested data']
colors = ['tab:blue','tab:green','tab:orange']
for i, df in enumerate([df_cong, df_free, df_cong_more]):
    plt.figure(figsize=(8,6))
    num_samples = 1000
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
    plt.xlim([0, 160])
    plt.ylim([0, 4000])
    plt.grid()


# Speed - density relationship
labels = ['Slow moving data', 'Free flowing data', 'Congested data']
speeds = [df_cong.Speed.to_list(), df_free.Speed.to_list(), df_cong_more.Speed.to_list()]
plt.figure(figsize=(8,6))
plt.hist(speeds, bins=30, rwidth=0.8, alpha=0.8, stacked=True, density=True)
plt.xlabel('Speed (kmp/hr)', fontsize=13); plt.xticks(fontsize=11)
plt.ylabel('Frequency', fontsize=13)
plt.legend(labels, loc='upper left', fontsize=11)
plt.title('Speed distributions', fontsize=13)
plt.grid()

# Flow - density relationship (comparison with HighD)`

# Flow density of simulated data (normalized)
q_sim = np.empty(0); k_sim = np.empty(0)
for i, df in enumerate([df_cong, df_free, df_cong_more]):
    num_samples = 1000
    q = df.Flow.to_numpy()
    k = df.Density.to_numpy()
    rand_indx = np.random.randint(0, df.Speed.shape[0], num_samples)
    x = k[rand_indx]
    y = q[rand_indx]
    x_filter = x[y <= 5000]
    y_filter = y[y <= 5000]
    q_sim = np.append(q_sim, y_filter)
    k_sim = np.append(k_sim, x_filter)
k_sim_norm = (k_sim - k_sim.min())/(k_sim.max() - k_sim.min())
q_sim_norm = (q_sim - q_sim.min())/(q_sim.max() - q_sim.min())

# Flow density of HighD data (normalized)
df_fd_lane = pd.read_csv('df_hw25_lane4.csv')
q_highd = np.empty(0); k_highd = np.empty(0)
num_samples = 1000
q = df_fd_lane.Flow.to_numpy()
k = df_fd_lane.Density.to_numpy()
rand_indx = np.random.randint(0, df_fd_lane.Speed.shape[0], num_samples)
x = k[rand_indx]
y = q[rand_indx]
x_filter = x[y <= 5000]
y_filter = y[y <= 5000]
q_highd = np.append(q_highd, y_filter)
k_highd = np.append(k_highd, x_filter)
q_highd_norm = (q_highd - q_highd.min())/(q_highd.max() - q_highd.min())
k_highd_norm = (k_highd - k_highd.min())/(k_highd.max() - k_highd.min())

plt.figure(figsize=(8,6))
plt.scatter(k_sim, q_sim, s=18, ec='tab:blue', fc='white', lw=1, label='Simulated data', alpha=1.0)
plt.scatter(k_highd, q_highd, s=18, ec='tab:orange', fc='white', lw=1, label='HighD data', alpha=1.0)
plt.xlabel('Density (vehs/km)', fontsize=13)
plt.ylabel('Flow (vehs/hr)', fontsize=13)
plt.legend(fontsize=13)
plt.title('Flow-density scatter plot', fontsize=14)
plt.xlim([0, 200])
# plt.ylim([0, 1])
plt.grid()


# =============================================================================
# Generate training dataset
# =============================================================================

# more parameters
l_dn = 80 # Car-following influence limit downstream of vehicle
l_up = 40 # Car-following influence limit upstream of vehicle
max_speed = 100
speed_95per = 90

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