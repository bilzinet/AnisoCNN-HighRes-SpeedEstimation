# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 22:32:57 2020
@author: Bilal Thonnam Thodi (btt1@nyu.edu)
RGB mapping of scalar traffic speeds
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

class MplColorMapper:
    '''
    Class for matplotlib color map creation and speed conversion
    Input:
        cmap_name: Color map name from matplotlib 
                   (https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html)
        min_val: Minimum value of scalar traffic speed
        max_val: Maximum value of scalar traffic speed
        lookuptable_dim: Dimension of lookup table for inverting RGB
    '''
    
    def __init__(self, cmap_name, min_val, max_val, lookuptable_dim):
        '''
        Class initializer
        '''
        self.min_val = min_val
        self.max_val = max_val
        self.cmap_name = cmap_name
        self.lookuptable_dim = lookuptable_dim
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
    
    def get_rgb(self, value):
        '''
        Convert to three-dimensional RGB array for defined color map.
        '''
        return self.scalarMap.to_rgba(value)
    
    def get_sval(self, rgb_arr):
        '''
        Convert the RGB array to scalar speed. 
        The invertion is finding closest scalar value from RGB array.
        '''
        rgb_arr = mpl.colors.to_rgb(rgb_arr)
        r = np.linspace(0, 1, self.lookuptable_dim)
        norm = mpl.colors.Normalize(0, 1)
        mapvals = self.cmap(norm(r))[:,:3]
        distance = np.sum((mapvals - rgb_arr)**2, axis=1)
        return r[np.argmin(distance)]*(self.max_val - self.min_val)
    
    def get_sval_mult(self, rgb_arrays):
        '''
        Invertion operation for multiple rgb queries.
        '''
        con_speeds = []
        for i in range(rgb_arrays.shape[0]):
            rgb_arr = rgb_arrays[i,:]
            con_speeds.append(self.get_sval(rgb_arr))
        return con_speeds
    
def plot_RGB_ranges(conversion_params):
    '''
    Plot the range of three-dimensional RGB array for the given
    scalar speed range.
    '''
    min_val = conversion_params['min_val']
    max_val = conversion_params['max_val']
    cmap_name = conversion_params['cmap_name']
    lookup_tab_dim = conversion_params['lookup_table_dim']
    colmap = MplColorMapper(cmap_name, min_val, max_val, lookup_tab_dim)
    
    speeds = np.linspace(min_val, max_val, 10000)
    speeds_rgba = colmap.get_rgb(speeds)
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(speeds, speeds_rgba[:,0])
    plt.plot(speeds, speeds_rgba[:,1])
    plt.plot(speeds, speeds_rgba[:,2])
    plt.xlabel("Traffic Speed", fontsize=14)
    plt.ylabel("RGB value", fontsize=14)
    plt.legend(["Red", "Green", "Blue"], fontsize=12)
    
    return fig, speeds_rgba[:,:3]
        
    
#min_val = 0
#max_val = 60
#cmap_name = 'jet_r'
#lookup_tab_dim = 256
#colmap = MplColorMapper(cmap_name, min_val, max_val, lookup_tab_dim)
#
#speeds = np.linspace(0, 60, 100)
#rgb_arrays = colmap.get_rgb(speeds)
#con_speeds = colmap.get_sval_mult(rgb_arrays)
#np.sum(np.abs(speeds-con_speeds))
#
#plt.plot(speeds)
#plt.plot(con_speeds)
#
#con_speeds = []
#for speed in speeds:    
#    rgb = colmap.get_rgb(speed)
#    con_speeds.append(colmap.get_sval(rgb))
#    
#    
#    
#
#plt.plot(speeds)
#plt.plot(con_speeds)
    