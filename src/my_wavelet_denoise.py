# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 11:33:19 2017

@author: Fruit Flies
"""
import numpy as np
import pywt
from pywt import threshold as pywtthresh
from statsmodels.robust import mad
import matplotlib.pyplot as plt

from changepy import pelt
from changepy.costs import normal_var, normal_mean

#------------------------------------------------------------------------------
def wcoeff_sparsity_metric(coeffs):
    wlevel_max = len(coeffs)-1
    spars_met = np.zeros((wlevel_max,1))
    for wl in np.arange(1,wlevel_max+1):
        spars_met[wl-1] = np.max(np.abs(coeffs[wl]))/np.sum(np.abs(coeffs[wl]))  
    #spars_met = np.flipud(spars_met)
    return spars_met 
#------------------------------------------------------------------------------
def wavelet_denoise(data,wtype='db4',wlevel=2,s_thresh=0.08, plotFlag=False):
    
    coeffs = pywt.wavedec(np.squeeze(data),wtype, level=wlevel,mode='symmetric')
    
     # find wavelet level based on sparsity metric
    sparsity_metric = wcoeff_sparsity_metric(coeffs)
    while (np.max(sparsity_metric) > s_thresh) and (wlevel > 2):
        wlevel = wlevel - 1 
        coeffs = pywt.wavedec(np.squeeze(data),wtype,level=wlevel,mode='symmetric')
        sparsity_metric = wcoeff_sparsity_metric(coeffs)
    
    sigma = mad(coeffs[-1])
    uthresh = sigma*np.sqrt(2*np.log(data.size))

    denoised = coeffs[:]
    denoised[1:] = ( pywtthresh(i, value=uthresh, mode='soft') for i in denoised[1:])
    
    data_denoised = pywt.waverec(denoised, wtype,mode='symmetric')
    
    if plotFlag:
        plt.figure()
        plt.plot(data)
        plt.plot(data_denoised,'r')
    
    if (data.size != data_denoised.size):
        data_denoised = data_denoised[0:data.size]
        
    #print(wlevel)
    return data_denoised

#------------------------------------------------------------------------------    
def wavelet_denoise_mln(data,wtype='db4',wlevel=2,plotFlag=False):
    
    coeffs = pywt.wavedec(np.squeeze(data),wtype, level=wlevel,mode='symmetric')
    sigma = [mad(cfs) for cfs in coeffs[1:]]
    uthresh = [sig*np.sqrt(2*np.log(data.size)) for sig in sigma]
    
    denoised = coeffs[:]
    denoised[1:] = ( pywtthresh(i, value=uthresh[j], mode='soft') for j,i in
                        enumerate(denoised[1:]))
    
    data_denoised = pywt.waverec(denoised, wtype,mode='symmetric')
    
    if plotFlag:
        plt.figure()
        plt.plot(data)
        plt.plot(data_denoised,'r')
    
    if (data.size != data_denoised.size):
        data_denoised = data_denoised[0:data.size]
    return data_denoised
    
#------------------------------------------------------------------------------    
def wavelet_denoise_window(data,break_pts,wtype='db4',wlevel=2,plotFlag=False):
    
    if 0 not in break_pts:
        break_pts.insert(0,0)
    if len(data) not in break_pts:
        break_pts.append(len(data))
        
    data_windows = [data[break_pts[i]:break_pts[i+1]] for i in range(len(break_pts)-1)]
    data_denoised_list = []
    for data_win in data_windows:
        max_level = pywt.dwt_max_level(len(data_win),8)
        level = np.min([max_level,wlevel])
        coeffs = pywt.wavedec(np.squeeze(data_win),wtype, level=level,mode='symmetric')
        sigma = mad(coeffs[-1])
        uthresh = sigma*np.sqrt(2*np.log(data_win.size))
    
        denoised = coeffs[:]
        denoised[1:] = ( pywtthresh(i, value=uthresh, mode='soft') for i in denoised[1:])
    
        data_denoised_win = pywt.waverec(denoised, wtype,mode='symmetric')
        data_denoised_list.append(data_denoised_win)
    
    data_denoised = [x for data_list in data_denoised_list for x in data_list ]   
    
    if plotFlag:
        plt.figure()
        plt.plot(data)
        plt.plot(data_denoised,'r')
    
    if (data.size != data_denoised.size):
        data_denoised = data_denoised[0:data.size]
    return data_denoised