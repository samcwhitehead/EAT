# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:33:24 2018

@author: Fruit Flies

A handful of useful functions to be used throughout expresso analysis code
"""
#------------------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
#-----------------------------------------------------------------------------
# tool for pulling out indices of an array with elements above thresh value
def idx_by_thresh(signal,thresh = 0.1):
    import numpy as np
    idxs = np.squeeze(np.argwhere(signal > thresh))
    try:
        split_idxs = np.squeeze(np.argwhere(np.diff(idxs) > 1))
    except IndexError:
        #print 'IndexError'
        return None
    #split_idxs = [split_idxs]
    if split_idxs.ndim == 0:
        split_idxs = np.array([split_idxs])
    #print split_idxs
    try:
        idx_list = np.split(idxs,split_idxs)
    except ValueError:
        #print 'value error'
        np.split(idxs,split_idxs)
        return None
    #idx_list = [x[1:] for x in idx_list]
    idx_list = [x for x in idx_list if len(x)>0]
    return idx_list

#-----------------------------------------------------------------------------
# handle nans
def nan_helper(y):
    return np.isnan(y),lambda z: z.nonzero()[0]

#------------------------------------------------------------------------------
# interpolate through nan values (missing track points)
def interp_nans(y,min_length=1):
    z = y.copy()
    #print('z is a {}'.format(type(z)))
    nans, x = nan_helper(z)
    #eliminate small data chunks (likely spurious)
    not_nan_idx = idx_by_thresh(~nans)
    for nidx in not_nan_idx:    
        if len(nidx) < min_length:
            #print(nidx)
            z[nidx] = np.nan
            nans[nidx] = np.nan
    #interpolate through remaining points
    z[nans] = np.interp(x(nans),x(~nans),z[~nans])
    return z
    
#------------------------------------------------------------------------------
# rolling window that avoids looping
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#------------------------------------------------------------------------------
# hampel filter 
def hampel(x,k=7, t0=3):
    '''taken from stack overflow
    x= 1-d numpy array of numbers to be filtered
    k= number of items in window/2 (# forward and backward wanted to capture in median filter)
    t0= number of standard deviations to use; 3 is default
    '''
    dk = int((k-1)/2)
    y = x.copy() #y is the corrected series
    L = 1.4826
    
    # calculate rolling median
    rolling_median = np.nanmedian(rolling_window(y,k),-1)
    rolling_median = np.concatenate((y[:dk], rolling_median, y[-dk:]))
    
    # compare rolling median to value at each point
    difference = np.abs(rolling_median-y)
    median_abs_deviation= np.nanmedian(rolling_window(difference,k),-1)
    median_abs_deviation= np.concatenate((difference[:dk], median_abs_deviation, 
                                          difference[-dk:]))
    
    # determine where data exceeds t0 standard deviations from the local median
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    
    y[outlier_idx] = rolling_median[outlier_idx]
    
    return(y)

#------------------------------------------------------------------------------
# rolling average filter  
def moving_avg(x,k=3):
    '''taken from stack overflow
    x= 1-d numpy array of numbers to be filtered
    k= number of items in window/2 (# forward and backward wanted to capture in filter)
    '''
    dk = int((k-1)/2)
    y = x.copy() #y is the corrected series
    
    # calculate rolling median
    rolling_mean = np.nanmean(rolling_window(y,k),-1)
    rolling_mean = np.concatenate((y[:dk], rolling_mean, y[-dk:]))
    
    return(rolling_mean)
#------------------------------------------------------------------------------
# interpolation for tracking data
def interpolate_tracks(X, dist_thresh=0.1, slope_thresh=1.0, N_pad=2, 
                       PLOT_FLAG=False):
    X_out = X.copy()
    nan_idx = np.isnan(X)
    
    # find chunks of tracked data separated by nans
    nan_idx_diff = np.diff(np.asarray(nan_idx,dtype=int))
    if nan_idx[0] == 1:
        nan_idx_diff = np.insert(nan_idx_diff,0,0)
    else:
        nan_idx_diff = np.insert(nan_idx_diff,0,-1)
        
    track_start = np.where(nan_idx_diff == -1)[0]
    track_end = np.where(nan_idx_diff == 1)[0] - 1
    
    # make sure to include end
    if (track_end.size < track_start.size) and (track_start.size > 1):
        track_end = np.insert(track_end,-1,len(X)-1)
    elif (track_end.size < track_start.size) and (track_start.size == 1):
        track_end = np.array([len(X)-1])
    
    # loop through chunks and interpolate between them if they meet criteria
    for ith in np.arange(len(track_start)-1):
        start_idx = track_start[ith]
        end_idx = track_end[ith]
        next_start_idx = track_start[ith+1]
        next_end_idx = track_end[ith + 1]
        
        # find distance between end of current track and start of next track
        dX = np.abs(X[next_start_idx] - X[end_idx])
        
        # find difference in slope between end of current track and start of next
        prev_pad_idx = np.arange(np.max([end_idx-N_pad, start_idx]),end_idx)+1
        slope_prev = np.mean(np.diff(X[prev_pad_idx]))
    
        next_pad_idx = np.arange(next_start_idx,np.min([next_start_idx+N_pad,
                                                        next_end_idx]))
        slope_next = np.mean(np.diff(X[next_pad_idx]))
        
        mean_slope = np.mean([slope_prev,slope_next])
        if mean_slope == 0 or np.isnan(mean_slope):
            dSlope = np.inf
        else:
            dSlope = np.abs((slope_next - slope_prev)/mean_slope)
        #print(dSlope)
        
        # check for distance and slope criteria
        if (dX <= dist_thresh) or (dSlope <= slope_thresh):
            X_out[start_idx:next_end_idx] = interp_nans(X[start_idx:next_end_idx])
             
            
    if PLOT_FLAG:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(X)),X_out,'r')    
        ax.plot(np.arange(len(X)),X,'k')
    
    return X_out
    
#------------------------------------------------------------------------------
# remove short duration tracks from time series--likely spurious
def remove_nubs(y,min_length=50):
    z = y.copy()
    nans, x = nan_helper(z)
    not_nan_idx = idx_by_thresh(~nans)
    for nidx in not_nan_idx:    
        if len(nidx) < min_length:
            #print(nidx)
            z[nidx] = np.nan
            if (nidx[-1]+1) < len(z):
                if not np.isnan(z[nidx[-1]+1]):
                    z[nidx[-1]+1] = np.nan
            if (nidx[0]-1) >= 0:
                if not np.isnan(z[nidx[0]-1]):
                    z[nidx[0]-1] = np.nan
            nans[nidx] = np.nan
    return z