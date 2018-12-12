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
def wcoef_plot(coeffs):
    """
    adapted: http://jseabold.net/blog/2012/02/23/wavelet-regression-in-python/
    
    """

    n_levels = len(coeffs)
    fig, ax_arr = plt.subplots(n_levels) 

    for i in range(n_levels):
        ax_arr[i].stem(coeffs[i])

    return fig
#------------------------------------------------------------------------------
def wcoeff_sparsity_metric(coeffs):
    wlevel_max = len(coeffs)-1
    spars_met = np.zeros((wlevel_max,1))
    for wl in np.arange(1,wlevel_max+1):
        spars_met[wl-1] = np.max(np.abs(coeffs[wl]))/np.sum(np.abs(coeffs[wl]))  
        #spars_met[wl-1] = np.sum(np.abs(coeffs[wl]))/np.sqrt(np.sum(coeffs[wl]**2))  
    #spars_met = np.flipud(spars_met)
    return spars_met 

#------------------------------------------------------------------------------
def wavelet_denoise(data,wtype='db4',wlevel=2,s_thresh=0.1, plotFlag=False):
    #s_thresh=0.08
    coeffs = pywt.wavedec(np.squeeze(data),wtype, level=wlevel,mode='symmetric')
    
    if plotFlag:
        wcoef_plot(coeffs)
     # find wavelet level based on sparsity metric
    
    sparsity_metric = wcoeff_sparsity_metric(coeffs)
    print('sparsity metric = {}'.format(sparsity_metric))
    while (np.max(sparsity_metric) > s_thresh) and (wlevel > 2):
        wlevel = wlevel - 1 
        coeffs = pywt.wavedec(np.squeeze(data),wtype,level=wlevel,mode='symmetric')
        sparsity_metric = wcoeff_sparsity_metric(coeffs)
        #print('sparsity metric = {}'.format(sparsity_metric))
        
    sigma = mad(coeffs[-1])
    uthresh = sigma*np.sqrt(2*np.log(data.size))

    denoised = coeffs[:]
    denoised[1:] = ( pywtthresh(i, value=uthresh, mode='soft') for i in denoised[1:])
    
    data_denoised = pywt.waverec(denoised, wtype,mode='symmetric')
    
    if plotFlag:
        plt.figure()
        plt.plot(data)
        plt.plot(data_denoised,'r')
        plt.tight_layout()
        print('wlevel = {}'.format(wlevel))
        
    if (data.size != data_denoised.size):
        data_denoised = data_denoised[0:data.size]
        
    #print(wlevel)
    return data_denoised

#------------------------------------------------------------------------------    
def wavelet_denoise_mln(data,wtype='db4',wlevel=2,s_thresh=0.1,s_thresh_low=0.01,
                        plotFlag=False):
    
    coeffs = pywt.wavedec(np.squeeze(data),wtype, level=wlevel,mode='symmetric')
    sparsity_metric = wcoeff_sparsity_metric(coeffs)
    
    wlevel_opt = np.where(sparsity_metric < s_thresh)[0][-1] + 1 
    coeffs = pywt.wavedec(np.squeeze(data),wtype, level=wlevel_opt,mode='symmetric')
    sm_opt = wcoeff_sparsity_metric(coeffs)
    
    # see https://www.acert.cornell.edu/PDFs/IEEEAccess4_3862_2016.pdf 
    #   (SRIVASTAVA, Anderson, and Freed, 2016)
    low_s_idx = np.where(sm_opt <= s_thresh_low)[0]
    mid_s_idx = np.where((sm_opt > s_thresh_low) & (sm_opt < s_thresh))[0]
    
    print('under construction')
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