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
from v_expresso_utils import interp_nans, hampel, moving_avg

#---------------------------------------------------------------------------------------
# check that data set actually contains data 
def check_data_set(dset,t):
    dset_check = (dset != -1)
    N_unique_vals = np.unique(dset).size
    bad_data_flag = (np.sum(dset_check) == 0) or (N_unique_vals < 2)
    
    if bad_data_flag:
        dset = np.array([])
        frames = np.array([])
        t = np.array([])
        #dset_smooth = np.array([])
        #bouts = np.array([])
        #volumes = np.array([])
        #print('Problem with loading data - invalid data set')
    else:    
        frames = np.arange(0,dset.size)
        
        dset = dset[dset_check]
        frames = frames[np.squeeze(dset_check)]
        t = t[dset_check]
        
        new_frames = np.arange(0,np.max(frames)+1)
        sp_raw = interpolate.InterpolatedUnivariateSpline(frames, dset)
        sp_t = interpolate.InterpolatedUnivariateSpline(frames, t)
        dset = sp_raw(new_frames)
        t = sp_t(new_frames)
        frames = new_frames
    
    return (bad_data_flag, dset, t, frames)
#---------------------------------------------------------------------------------------
# returns denoised channel signal 
def process_signal(dset, wtype='db4',wlevel=5,medfilt_window=7,
                   pos_der_thresh=10.0, PRE_FILT_FLAG=True, MLN_FLAG=False):
    
    # remove points with high slope/filter for outliers before wavelet denoise
    if PRE_FILT_FLAG:
        dset_der = np.diff(np.squeeze(dset))
        dset_der = np.insert(dset_der,0,0)
        good_ind = (np.abs(dset_der) < pos_der_thresh)
        dset_good = dset[good_ind]
        
        frames = np.arange(len(dset))
        frames_good = frames[good_ind]
        dset = np.interp(frames, frames_good, dset_good)
        
        #dset = moving_avg(dset,k=5)
        #dset = hampel(dset, k=7, t0=4)
        
    if MLN_FLAG:
        dset_denoised = wavelet_denoise_mln(dset, wtype, wlevel) 
    else:
        dset_denoised = wavelet_denoise(dset, wtype, wlevel)
    #dset_denoised_hampel = hampel(dset_denoised, k=7, t0=3)
    dset_denoised_med = signal.medfilt(dset_denoised,medfilt_window)
    #dset_denoised_med = moving_avg(dset_denoised_med,k=5)
    
    return dset_denoised_med

# ---------------------------------------------------------------------------------------
# returns slopes for intervals defined by changepoint detection
def fit_piecewise_slopes(dset_denoised_med,frames, 
                         analysis_params=analysisParams, var_user_flag=False):
    
    # wtype = analysis_params['wtype']
    # wlevel = analysis_params['wlevel']
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
        ipt2 = changepts[i+1] # + 1
        fit_temp = np.median(dset_der[ipt1:ipt2])
        #fit_temp = np.mean(dset_der[ipt1:ipt2])
        piecewise_fits[i] = fit_temp
        piecewise_fit_dist[ipt1:ipt2] = fit_temp*np.ones_like(dset_der[ipt1:ipt2])
        piecewise_fit_dur[i] = len(range(ipt1,ipt2))
    
    return (dset_der, changepts, piecewise_fits, piecewise_fit_dist, piecewise_fit_dur)    

#------------------------------------------------------------------------------
# function to generate plot of channel data with meal bouts marked
def plot_channel_bouts(dset, dset_smooth, t, bouts, figsize=(12, 7),
                       bout_color='r',bout_bar_color='grey',bout_bar_alpha=0.3):
                           
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True,figsize=figsize)
            
    ax1.set_ylabel('Liquid [nL]')
    ax2.set_ylabel('Liquid [nL]')
    ax2.set_xlabel('Time [s]')
    ax1.set_title('Raw Data')
    ax2.set_title('Smoothed Data')
    
    ax1.plot(t,dset)
    ax2.plot(t, dset_smooth)
    for i in np.arange(bouts.shape[1]):
        ax2.plot(t[bouts[0,i]:bouts[1,i]], dset_smooth[bouts[0,i]:bouts[1,i]],
                                                         '-',color=bout_color)
        ax2.axvspan(t[bouts[0,i]],t[bouts[1,i]-1], 
                         facecolor=bout_bar_color, 
                         edgecolor='none', 
                         alpha=bout_bar_alpha)
        ax1.axvspan(t[bouts[0,i]],t[bouts[1,i]-1], 
                         facecolor=bout_bar_color, 
                         edgecolor='none', 
                         alpha=bout_bar_alpha)
        
    ax1.set_xlim([t[0],t[-1]])
    ax1.set_ylim([np.amin(dset),np.amax(dset)])
    # plt.show()

    return (fig, ax1, ax2)    

#------------------------------------------------------------------------------                
def bout_analysis(dset,frames, analysis_params=analysisParams, 
                  var_user_flag=False, debug_mode=False):
    
    #=============================
    # params for data processing
    #=============================
    wlevel = analysis_params['wlevel'] 
    wtype = analysis_params['wtype']
    medfilt_window = analysis_params['medfilt_window']
    min_bout_duration = analysis_params['min_bout_duration']
    min_bout_volume = analysis_params['min_bout_volume']
    min_pos_slope = analysis_params['min_pos_slope']
    pos_der_thresh = analysis_params['pos_der_thresh']
    #hampel_k = analysis_params['hampel_k']
    #hampel_sigma = analysis_params['hampel_sigma']
    mad_thresh = analysis_params['mad_thresh']
    #var_user = analysis_params['var_user']
    
    #--------------------------------------------------------------------------
    # process data and find changepoint intervals (+ their avg derivative)
    dset_denoised_med = process_signal(dset, wtype=wtype, wlevel=wlevel,
                                       medfilt_window=medfilt_window, 
                                       pos_der_thresh=pos_der_thresh)
    
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
                ind1 = neg_slope_prev[-1] + 1
                ref_val = dset_denoised_med[ind1 - 1]
            except IndexError:
                #ind1 = 1
                #ref_val = dset_denoised_med[chgpt2]
                ind1 = 1
                ref_val = dset_denoised_med[0]
            
            below_ref_idx = np.where(dset_denoised_med[chgpt2:] <= ref_val)[0]
            if below_ref_idx.size < 1:
                ind2 = len(dset_denoised_med) - 2
                dset_denoised_med[-1] = ref_val
            else:
                ind2 = below_ref_idx[0] + chgpt2 - 1
                dset_denoised_med[below_ref_idx[0] + chgpt2] = ref_val
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
        dset_der,changepts,piecewise_fits,_,piecewise_fit_dur = fit_piecewise_slopes(dset_denoised_med,
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
    evap_ind = np.logical_and(~bout_ind, (piecewise_fits < 0))
    
    # get bout start/stop times by finding when z score shifts below/above thresh
    bout_ind = bout_ind.astype(int)
    bout_ind_diff = np.diff(bout_ind)
    bouts_start_ind = np.where(bout_ind_diff == 1)[0] + 1 
    bouts_end_ind = np.where(bout_ind_diff == -1)[0] + 1 
    
    # make sure if the expt starts with a meal this is included
    if bout_ind[0] == 1:
        bouts_start_ind = np.insert(bouts_start_ind,0,0)
    
    # make sure there are an equal number of starts and stops
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
    
    # estimate evaporation rate and see if any bouts can be explained by this
    evap_rates = piecewise_fits[evap_ind]
    evap_durations = piecewise_fit_dur[evap_ind]
    mean_evap_rate = np.dot(evap_rates,evap_durations)/np.sum(evap_durations)
    
    evap_vol_pred = mean_evap_rate*bout_durations
    max_err = np.zeros(bouts.shape[1])
    for i in np.arange(bouts.shape[1]):
        i1 = bouts[0,i]
        i2 = bouts[1,i]
        max_err[i] = -1*np.max(np.abs((dset[i1:i2]-dset_denoised_med[i1:i2])))
    noise_est_vol = -1.0*(max_err + evap_vol_pred)
    good_evap_ind = (noise_est_vol < volumes) 
    #good_ind = good_ind & good_evap_ind
    
    # take only bouts that meet criteria
    bouts = bouts[:,good_ind]
    volumes = volumes[good_ind]
    bout_durations = bout_durations[good_ind]
    
    #--------------------------------------------------------------------------
    # create plots to check how the bout detection is functioning
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
        ax2.plot(frames[changepts_array[1:-1]], dset_denoised_med[changepts[1:-1]], 'go')
        
        # plot time derivative of data
        ax3.plot(frames,np.zeros_like(frames),'k--')
        ax3.plot(frames, dset_der)
        for i in np.arange(bouts.shape[1]):
            ax3.plot(frames[bouts[0,i]:bouts[1,i]], 
                     dset_der[bouts[0,i]:bouts[1,i]],'r-')
        ax3.plot(frames[changepts_array[1:-1]], dset_der[changepts[1:-1]], 'go')
        
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
        #fig_hist, ax_hist = plt.subplots()
        #nbins = 100
        #ax_hist.hist(dset_der,nbins)
        
        #====================================================
        # visualize estimated evaporation rate
        #====================================================
        evap_ind = evap_ind.astype(int)
        evap_ind_diff = np.diff(evap_ind)
        evap_start_ind = np.where(evap_ind_diff == 1)[0] + 1 
        evap_end_ind = np.where(evap_ind_diff == -1)[0] + 1 
        
        if evap_ind[0] == 1:
            evap_start_ind = np.insert(evap_start_ind,0,0)
    
        # make sure there are an equal number of starts and stops
        if len(evap_start_ind) != len(evap_end_ind):
            minEvapLength = np.min([len(evap_start_ind), len(evap_end_ind)])
            evap_start_ind = evap_start_ind[0:minEvapLength]
            evap_end_ind = evap_end_ind[0:minEvapLength]
        
        evap_start = changepts_array[evap_start_ind]
        evap_end = changepts_array[evap_end_ind]
        
        # make plot
        fig, (ax1_evap, ax2_evap) = plt.subplots(2, sharex=True, sharey=True, figsize=(6,6))
        
        evap_length_max = 0
        for i in np.arange(len(evap_start_ind)):
            idx1 = evap_start[i]
            idx2 = evap_end[i]
            
            evap_length = idx2 - idx1 
            if evap_length > evap_length_max:
                evap_length_max = evap_length
            
            ax1_evap.plot(np.arange(evap_length), dset[idx1:idx2] - dset[idx1])
            ax2_evap.plot(np.arange(evap_length), 
                          dset_denoised_med[idx1:idx2] - dset_denoised_med[idx1])
        
        
        evap_line = mean_evap_rate*np.arange(evap_length_max)
        ax1_evap.plot(np.arange(evap_length_max), evap_line, 'k--')
        ax2_evap.plot(np.arange(evap_length_max), evap_line, 'k--')
        
         # set axis limits
        ax1_evap.set_xlim([0,evap_length_max])
        ax1_evap.set_ylim([-15,0])    
        
        # set axis labels
        ax1_evap.set_ylabel('Liquid [nL]')
        ax2_evap.set_ylabel('Liquid [nL]')
        ax2_evap.set_xlabel('Frames [num]')
        ax1_evap.set_title('Evaporation Estimation')
        
        #====================================================
        # compare volumes from raw data to smoothed
        #====================================================
        #fig, ax = plt.subplots()
        #for i in np.arange(bouts.shape[1]):
        #    idx1 = bouts[0,i]
        #    idx2 = bouts[1,i]
        #    #ax.plot(i,np.sum(np.diff(dset[idx1:idx2])),'ro')
        #    ax.plot(i,-1*np.max(np.abs((dset[idx1:idx2]-dset_denoised_med[idx1:idx2]))),'ro')
        #ax.plot(np.arange(bouts.shape[1]),-1*volumes,'bo')
        #ax.plot(np.arange(bouts.shape[1]),mean_evap_rate*bout_durations,'go')
        
    return (dset_denoised_med, bouts, volumes)
    
    
