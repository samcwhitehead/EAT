# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:33:24 2018

@author: Fruit Flies

A handful of useful functions to be used throughout expresso analysis code
"""
#------------------------------------------------------------------------------
import numpy as np

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