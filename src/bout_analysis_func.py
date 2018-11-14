# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 10:39:45 2016

@author: Fruit Flies
"""
from __future__ import division

import numpy as np
from scipy import signal, interpolate

import matplotlib.pyplot as plt

from changepy import pelt
from changepy.costs import normal_mean, normal_meanvar

from my_wavelet_denoise import wavelet_denoise, wavelet_denoise_mln
from v_expresso_gui_params import analysisParams
from v_expresso_utils import interp_nans, hampel

#---------------------------------------------------------------------------------------
# returns denoised channel signal 
def process_signal(dset, analysis_params=analysisParams):
    
    wtype = analysis_params['wtype']
    wlevel = analysis_params['wlevel']
    medfilt_window =analysis_params['medfilt_window']
    dset_denoised = wavelet_denoise(dset, wtype, wlevel) 
    #dset_denoised_hampel = hampel(dset_denoised, k=7, t0=3)
    dset_denoised_med = signal.medfilt(dset_denoised,medfilt_window)

    return dset_denoised_med

#---------------------------------------------------------------------------------------
# returns slopes for intervals defined by changepoint detection
def fit_piecewise_slopes(dset_denoised_med,frames, 
                         analysis_params=analysisParams, var_user_flag=False):
    
    #wtype = analysis_params['wtype']
    #wlevel = analysis_params['wlevel']
    clip_level = 1 
    
    # calculate derivative of signal at each point
    sp_dset = interpolate.InterpolatedUnivariateSpline(frames,
                                              np.squeeze(dset_denoised_med))
#    sp_dset = interpolate.UnivariateSpline(frames,np.squeeze(dset_denoised_med),
#                                           s=np.sqrt(dset_denoised_med.size))
    sp_der = sp_dset.derivative(n=1)
    
    dset_der = sp_der(frames)
    #dset_der = wavelet_denoise(dset_der, wtype, wlevel) 
    
    # try to exclude outliers for robust variance estimation
    N = len(dset_der) - 1
    dset_der_clipped = dset_der[(dset_der > np.percentile(dset_der,clip_level)) & \
                        (dset_der < np.percentile(dset_der,100-clip_level))]
    
    # variance estimation for changepoint detector
    if var_user_flag == False:
        iq_range = np.percentile(dset_der_clipped,75) - \
                    np.percentile(dset_der_clipped,25)
        var_user = (iq_range/2.0)**2 #2*iq_range 
    
    # find changepoints
    changepts = pelt(normal_mean(dset_der,var_user),len(dset_der)) 
    
    if 0 not in changepts:
        changepts.insert(0,0)
    if N not in changepts:
        changepts.insert(len(changepts),N)
            
    # 'fit' slope in intervals
    piecewise_fits = np.empty(len(changepts)-1)
    piecewise_fit_dur = np.empty(len(changepts)-1)
    piecewise_fit_dist = np.empty_like(dset_der)
    
    for i in range(0,len(changepts)-1):
        ipt1 = changepts[i]
        ipt2 = changepts[i+1] + 1
        fit_temp = np.median(dset_der[ipt1:ipt2])
        #fit_temp = np.mean(dset_der[ipt1:ipt2])
        piecewise_fits[i] = fit_temp
        piecewise_fit_dist[ipt1:ipt2] = fit_temp*np.ones_like(dset_der[ipt1:ipt2])
        piecewise_fit_dur[i] = len(range(ipt1,ipt2))
    
    return (dset_der, changepts, piecewise_fits, piecewise_fit_dist, piecewise_fit_dur)    

#------------------------------------------------------------------------------                
def bout_analysis(dset,frames, analysis_params=analysisParams, 
                  var_user_flag=False, debug_mode=False):
    
    #=============================
    # params for data processing
    #=============================
    #wlevel = analysis_params['wlevel'] 
    #wtype = analysis_params['wtype']
    #medfilt_window = analysis_params['medfilt_window']
    min_bout_duration = analysis_params['min_bout_duration']
    min_bout_volume = analysis_params['min_bout_volume']
    min_pos_slope = analysis_params['min_pos_slope']
    
    mad_thresh = analysis_params['mad_thresh']
    #var_user = analysis_params['var_user']
    #--------------------------------------------------------------------------
    # process data and find changepoint intervals (+ their avg derivative)
    dset_denoised_med = process_signal(dset, analysis_params=analysis_params)
    
    dset_der,changepts,piecewise_fits,_,piecewise_fit_dur = fit_piecewise_slopes(
                            dset_denoised_med,frames,var_user_flag=var_user_flag)
    
    #--------------------------------------------------------------------------
    # try to handle any significant positive-slope bumps in signal (errors)
    pos_slope_logical = (piecewise_fit_dur >= min_bout_duration) & \
                    (piecewise_fits > min_pos_slope)
                
    if np.sum(pos_slope_logical) > 0:
        
        pos_slope_idx = np.where(pos_slope_logical)[0]
        nan_ind_list = [] 
        
        for idx in pos_slope_idx:
            
            chgpt1 = np.squeeze(changepts[np.squeeze(idx)])
            chgpt2 = np.squeeze(changepts[np.squeeze(idx)+1] + 1)
            
            neg_slope_prev = np.where(dset_der[:chgpt1] <= 0.0)[0]
            try:
                ind1 = neg_slope_prev[-1]
                ref_val = dset_denoised_med[ind1]
            except IndexError:
                ind1 = 1
                ref_val = dset_denoised_med[chgpt2]
            
            below_ref_idx = np.where(dset_denoised_med[chgpt2:] <= ref_val)[0]
            if below_ref_idx.size < 1:
                ind2 = len(dset_denoised_med)-1
            else:
                ind2 = below_ref_idx[0] + chgpt2
            #next_val = next(x for x in dset_denoised_med[chgpt2:] if x <= ref_val)
            #next_val_idx = np.where(dset_denoised_med[chgpt2:] == next_val)[0]
            #ind2 = next_val_idx[0] + chgpt2
            
            nan_ind_list.append([ind1, ind2])
            #dset_der[ind1:ind2] = np.nan
            #dset_denoised_med[ind1:ind2] = np.nan
        
        for nan_ind in nan_ind_list:
            dset_der[nan_ind[0]:nan_ind[1]] = np.nan
            dset_denoised_med[nan_ind[0]:nan_ind[1]] = np.nan
        #dset_der_temp = interp_nans(dset_der_temp)
        dset_denoised_med = interp_nans(dset_denoised_med)
        dset_der,changepts,piecewise_fits,_,_ = fit_piecewise_slopes(dset_denoised_med,
                                            frames,var_user_flag=var_user_flag)
    #--------------------------------------------------------------------------
    # determine which intervals represent feeding bouts
                                            
    dset_der_neg = dset_der[(dset_der < 0)]
    #dset_der_median = np.median(dset_der_clipped, axis=0)
    #diff = np.abs(dset_der_clipped - dset_der_median)
    dset_der_median = np.median(dset_der_neg, axis=0)
    diff = np.abs(dset_der_neg - dset_der_median)
    med_abs_deviation = np.median(diff)
    
    #piecewise_fits_dev = np.sqrt((piecewise_fits - dset_der_median)**2) / med_abs_deviation
    piecewise_fits_dev = (piecewise_fits - dset_der_median) / med_abs_deviation
    modified_z_score = 0.6745 * piecewise_fits_dev
    bout_ind = (modified_z_score < -1*mad_thresh)
    #bout_ind = (piecewise_fits_dev < mad_thresh) #~z score of 1 #(mean_pw_slope - std_pw_slope)
    bout_ind = bout_ind.astype(int)
    bout_ind_diff = np.diff(bout_ind)
    
    #print(bout_ind)
    #print(bout_ind_diff)
    #piecewise_fits_dev = (piecewise_fits - np.median(dset_der)) / mad_slope
    #bout_ind = (piecewise_fits_dev < mad_thresh) #~z score of 1 #(mean_pw_slope - std_pw_slope)
    #bout_ind = bout_ind.astype(int)
    #bout_ind_diff = np.diff(bout_ind)
    
    #plt.figure()
    #plt.plot(bout_ind)
    
    bouts_start_ind = np.where(bout_ind_diff == 1)[0] + 1 
    bouts_end_ind = np.where(bout_ind_diff == -1)[0] + 1 
    
    if bout_ind[0] == 1:
        bouts_start_ind = np.insert(bouts_start_ind,0,0)
    #print(bouts_start_ind)
    #print(bouts_end_ind)
    
    if len(bouts_start_ind) != len(bouts_end_ind):
        minLength = np.min([len(bouts_start_ind), len(bouts_end_ind)])
        bouts_start_ind = bouts_start_ind[0:minLength]
        bouts_end_ind = bouts_end_ind[0:minLength]
        
    #print(bouts_start_ind)
    #print(bouts_end_ind)
    
    changepts_array = np.asarray(changepts)
    bouts_start = changepts_array[bouts_start_ind]
    bouts_end = changepts_array[bouts_end_ind]
    
    bouts = np.vstack((bouts_start, bouts_end))
    volumes = dset_denoised_med[bouts_start] - dset_denoised_med[bouts_end]
    
    bout_durations = bouts_end - bouts_start
    good_ind = (bout_durations > min_bout_duration) & (volumes > min_bout_volume)
    
    bouts = bouts[:,good_ind]
    volumes = volumes[good_ind]
    
    # create plot to check how the bout detection is functioning
    if debug_mode:
        
        #====================================================
        # show time series with change points, z scores, etc
        #====================================================
        
        # create array of modified z-score to plot
        z_score_array = np.ones(dset_denoised_med.shape)
        for kth in np.arange(len(changepts)-1):
            z_score_array[changepts[kth]:changepts[kth+1]] = modified_z_score[kth]
            #print(modified_z_score[kth])
            
        # make plot window
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(12, 9))
        
        # plot data 
        ax1.plot(frames,dset)
        ax2.plot(frames, dset_denoised_med)
        for i in np.arange(bouts.shape[1]):
            ax2.plot(frames[bouts[0,i]:bouts[1,i]], 
                     dset_denoised_med[bouts[0,i]:bouts[1,i]],'r-')
            ax1.axvspan(frames[bouts[0,i]],frames[bouts[1,i]-1], 
                                 facecolor='grey', edgecolor='none', alpha=0.3)
        ax2.plot(changepts[1:-1], dset_denoised_med[changepts[1:-1]], 'go')
        
        # plot time derivative of data
        ax3.plot(frames, dset_der)
        for i in np.arange(bouts.shape[1]):
            ax3.plot(frames[bouts[0,i]:bouts[1,i]], 
                     dset_der[bouts[0,i]:bouts[1,i]],'r-')
        ax3.plot(changepts[1:-1], dset_der[changepts[1:-1]], 'go')
        
        #plot z score
        ax4.plot(frames,z_score_array,'m-')
        ax4.plot(frames,-1*mad_thresh*np.ones(dset_denoised_med.shape),'k--')
        
        # set axis limits
        ax1.set_xlim([frames[0],frames[-1]])
        ax1.set_ylim([np.amin(dset),np.amax(dset)])    
        
        # set axis labels
        ax1.set_ylabel('Liquid [nL]')
        ax2.set_ylabel('Liquid [nL]')
        ax3.set_ylabel('dL/dt')
        ax4.set_ylabel('Modified Z Score')
        ax4.set_xlabel('Frames [num]')
        ax1.set_title('Raw Data')
        ax2.set_title('Smoothed Data')
        
        #====================================================
        # show histogram of slopes
        #====================================================
        fig_hist, ax_hist = plt.subplots()
        nbins = 100
        ax_hist.hist(dset_der,nbins)
            
    return (dset_denoised_med, bouts, volumes)
    
    
