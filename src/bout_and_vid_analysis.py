# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:00:55 2018

@author: Fruit Flies

Tools for combining tracking and feeding data from Visual Expresso expts
"""
#------------------------------------------------------------------------------
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm 

import re
import h5py

from load_hdf5_data import load_hdf5
from bout_analysis_func import check_data_set, plot_channel_bouts, bout_analysis
from v_expresso_gui_params import (analysisParams, trackingParams)
from v_expresso_image_lib import (visual_expresso_main, 
                                    process_visual_expresso, 
                                    plot_body_cm, plot_body_vel, 
                                    plot_body_angle, plot_moving_v_still, 
                                    plot_cum_dist, hdf5_to_flyTrackData,
                                    save_vid_time_series, save_vid_summary, 
                                    batch_plot_cum_dist, batch_plot_heatmap)

#from PIL import ImageTk, Image
import ast 
import csv
from openpyxl import Workbook
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# function to convert channel entries to basic filepath string for comparison
def channel2basic(channel_entry):

    filepath, bank_name, channel_name = channel_entry.split(', ',2)
    filepath_no_ext= os.path.splitext(filepath)[0]
    basic_entry = filepath_no_ext + '_' + bank_name + '_' + channel_name
    
    return basic_entry

#------------------------------------------------------------------------------
# function to convert video entries to basic filepath string for comparison
def vid2basic(vid_entry):
    basic_entry = os.path.splitext(vid_entry)[0]
    return basic_entry
 
#------------------------------------------------------------------------------
# function to convert basic entries back to channel listbox format
def basic2channel(basic_entry):
    ent_split = basic_entry.split('_')
    channel_curr = '_'.join(ent_split[-2:])
    bank_curr = ent_split[-3]
    filepath_no_ext = '_'.join(ent_split[:-3])
    filepath = filepath_no_ext + '.hdf5'
    
    channel_entry = ', '.join([filepath, bank_curr, channel_curr])
    if sys.version_info[0] < 3:
        channel_entry = unicode(channel_entry)
    else:
        channel_entry = str(channel_entry)
    
    return channel_entry

#------------------------------------------------------------------------------
# function to convert video entries to basic filepath string for comparison
def basic2vid(basic_entry, file_ext='.avi'):
    vid_entry = basic_entry + file_ext
    return vid_entry

#------------------------------------------------------------------------------
# function that will take a list of basic entries and group them by expt
def group_expts_basic(basic_entry_list):
    dir_list = [os.path.dirname(ent) for ent in basic_entry_list]
    dir_list_unique = list(set(dir_list))
    
    expt_idx = np.zeros(len(dir_list),dtype=int)
    N_expt = []
    for ith, dr in enumerate(dir_list_unique):
        idx = [i for i, j in enumerate(dir_list) if j == dr]
        N_expt.append(len(idx))
        expt_idx[idx] = ith
        
    return expt_idx, N_expt
    
#------------------------------------------------------------------------------
# function to check bouts based on tracking data
def bout_analysis_wTracking(filename, bank_name, channel_name, bts=[], 
                            vols=[], time=[], dset_sm=[], bout_params=analysisParams, 
                            saveFlag=False, plotFlag=False, debugBoutFlag=False,
                            debugTrackingFlag=False):
    
    MOVE_FRAC_THRESH = bout_params['feeding_move_frac_thresh']
    MAX_DIST_THRESH = bout_params['feeding_dist_max']
    MIN_DIST_THRESH = bout_params['feeding_dist_min']
    MIN_DIST_THRESH_X = bout_params['feeding_dist_min_x']
    MIN_DIST_THRESH_Y = bout_params['feeding_dist_min_y']
    MAX_VEL_THRESH  = bout_params['feeding_vel_max']
    dist_max_prctile_lvl = 90 # 75
    dist_min_prctile_lvl = 50 # 25 
    vel_max_prctile_lvl = 50 #50
    
    #--------------------------------
    # load channel data and get bouts
    #--------------------------------
    if (np.asarray(bts).size > 0) and (np.asarray(vols).size > 0):
        bouts = bts
        volumes = vols
        dset_smooth = dset_sm
        t = time 
    else:
        dset, t = load_hdf5(filename,bank_name,channel_name)        
        
        bad_data_flag, dset, t, frames = check_data_set(dset,t)
        
        if not bad_data_flag:
            dset_smooth, bouts, volumes = bout_analysis(dset,frames,
                                                    debug_mode=debugBoutFlag)
        else:
            print('Problem loading data set')
            dset_smooth = np.nan
            bouts_corrected = np.nan
            volumes_corrected = np.nan
            return (dset_smooth, bouts_corrected,volumes_corrected)
        
    #--------------------------------
    # load tracking data
    #--------------------------------
    file_path, filename_hdf5 = os.path.split(filename)
    data_prefix = os.path.splitext(filename_hdf5)[0]
    filename_vid = data_prefix + '_' + bank_name + "_" + \
                            channel_name + '.avi' 
    filename_vid_analysis = data_prefix + '_' + bank_name + "_" + \
                            channel_name + '_TRACKING_PROCESSED.hdf5' 
    filename_vid_track = data_prefix + '_' + bank_name + "_" + \
                            channel_name + '_TRACKING.hdf5' 
                            
    filename_vid_full = os.path.join(file_path, filename_vid)
    filename_vid_full = os.path.abspath(filename_vid_full)
    
    filename_vid_analysis_full = os.path.join(file_path, filename_vid_analysis)
    filename_vid_analysis_full = os.path.abspath(filename_vid_analysis_full)
    
    if os.path.exists(filename_vid_analysis_full):
        flyTrackData = hdf5_to_flyTrackData(file_path, filename_vid_analysis)
    elif os.path.exists(filename_vid_full):
        print('{} has not yet been analyzed. Doing so now...'.format(filename_vid_full))
        flyTrackData = visual_expresso_main(file_path, filename_vid, 
                            DEBUG_BG_FLAG=debugTrackingFlag, 
                            DEBUG_CM_FLAG=debugTrackingFlag, 
                            SAVE_DATA_FLAG=True, ELLIPSE_FIT_FLAG = False, 
                            PARAMS=trackingParams)
        flyTrackData = process_visual_expresso(file_path, filename_vid_track,
                            SAVE_DATA_FLAG = True, DEBUG_FLAG = False)
    else:
        print('Error: no matching video file found. check directory')
        bouts_corrected = bouts
        volumes_corrected = volumes
        return (dset_smooth, bouts_corrected,volumes_corrected)
        
    #-------------------------------------------------
    # look at fly position/velocity during meal bouts
    #-------------------------------------------------
    channel_t = t 
    vid_t = flyTrackData['t']
    xcm_smooth = flyTrackData['xcm_smooth']
    ycm_smooth = flyTrackData['ycm_smooth']
    dist_mag = np.sqrt(xcm_smooth**2 + ycm_smooth**2)
    vel_mag = flyTrackData['vel_mag']
    moving_ind = flyTrackData['moving_ind']
    N_bouts = bouts.shape[1] 
    
    bout_check = np.zeros(N_bouts)
    for ith in np.arange(N_bouts):
        # first need to match up timing between video and channel 
        vid_start_idx = np.argmin(np.abs(vid_t - channel_t[bouts[0,ith]]))
        vid_end_idx = np.argmin(np.abs(vid_t - channel_t[bouts[1,ith]]))
        vid_bout_dur = float(vid_end_idx - vid_start_idx)
        
        # check that the fly is not moving for a certain fraction of bout
        total_moving_bout = float(np.sum(moving_ind[vid_start_idx:vid_end_idx]))
        moving_chk = total_moving_bout/vid_bout_dur
        moving_chk = (moving_chk < MOVE_FRAC_THRESH)
        
        # check that the fly doesn't exceed a certain velocity
        vel_mag_bout = vel_mag[vid_start_idx:vid_end_idx]
        vel_max_prctile = np.percentile(vel_mag_bout,vel_max_prctile_lvl)
        max_vel_check = (vel_max_prctile < MAX_VEL_THRESH)
        
        # check that fly is close to cap tip
#        dist_mag_bout = dist_mag[vid_start_idx:vid_end_idx]
#        dist_max_prctile = np.percentile(dist_mag_bout,dist_max_prctile_lvl)
#        dist_min_prctile = np.percentile(dist_mag_bout,dist_min_prctile_lvl)
#        max_dist_check = (dist_max_prctile < MAX_DIST_THRESH)
        x_bout = np.abs(xcm_smooth[vid_start_idx:vid_end_idx])
        y_bout = np.abs(ycm_smooth[vid_start_idx:vid_end_idx])
        dist_min_prctile_x = np.percentile(x_bout,dist_min_prctile_lvl)
        dist_min_prctile_y = np.percentile(y_bout,dist_min_prctile_lvl)
        min_dist_check = (dist_min_prctile_x < MIN_DIST_THRESH_X) & \
                            (dist_min_prctile_y < MIN_DIST_THRESH_Y)
            
        
        # make sure that all 3 boolean values are true. this means that:
        #   1) the fly is moving for less than MOVE_FRAC_THRESH of the time
        #   2) the fly doesn't go too far from the tip during the meal
        #   3) the fly gets sufficiently close to the tip
        bout_check[ith] = min_dist_check & max_vel_check #moving_chk & #max_dist_check & 
        
    # now remove the bouts that fail to meet distance/moving criteria
    bout_check = np.asarray(bout_check,dtype=int)
    bouts_corrected = bouts[:,(bout_check == 1)]
    volumes_corrected = volumes[(bout_check == 1)]
    
    # plot results?
    if plotFlag:
        fig_cor, ax1_cor, ax2_cor = plot_channel_bouts(dset,dset_smooth,t,
                                                       bouts_corrected)
        fig, ax1, ax2 = plot_channel_bouts(dset,dset_smooth,t, bouts)   
        fig_cor.suptitle('With tracking')
        fig.suptitle('No tracking')                                            
    
    # in addition to standard bout debugging plots, create plot that shows 
    #   location and velocity during putative bout
    if debugBoutFlag:
        fig, ax1, ax2 = plot_channel_bouts(dset,dset_smooth,t, bouts_corrected)
        color = 'green'  
        
        # plot distance from tip on top of raw data        
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylabel('Dist. from tip [cm]', color=color)                                           
        ax1_twin.plot(vid_t, dist_mag, color=color)
        ax1_twin.tick_params(axis='y', labelcolor=color)
        ax1_twin.set_xlim([0, np.max(vid_t)])
        
        # plot speed on top of smooth data
        ax2_twin = ax2.twinx()
        ax2_twin.set_ylabel('Speed [cm/s]', color=color)                                           
        ax2_twin.plot(vid_t, flyTrackData['vel_mag'], color=color)
        ax2_twin.tick_params(axis='y', labelcolor=color)
        ax2_twin.set_xlim([0, np.max(vid_t)])
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
    # save results?
    if saveFlag:
        print('under construction')
        
    return (dset_smooth, bouts_corrected, volumes_corrected)


#------------------------------------------------------------------------------
# function to create dictionary with both tracking and feeding data
def merge_v_expresso_data(dset,dset_smooth,channel_t,frames,bouts, volumes, 
                          flyTrackData, boutParams=analysisParams):
    
    flyCombinedData = dict()
    for key, value in flyTrackData.items():
        flyCombinedData[key] = value
    
    filename = flyCombinedData['filename']
    filename_split = filename.split('_')
    if filename_split[-2] == 'TRACKING':
        filename_new = '_'.join(filename_split[:-2]) + '_COMBINED_DATA.hdf5'
    else:
        filename_new = '_'.join(filename_split[:-1]) + '_COMBINED_DATA.hdf5'
        
    flyCombinedData.update({'channel_dset' : dset ,
                            'channel_dset_smooth' : dset_smooth ,
                            'channel_t' : channel_t , 
                            'channel_frames' : frames , 
                            'bouts' : bouts , 
                            'volumes' : volumes , 
                            'boutParams' : boutParams ,
                            'filename' : filename_new })
    
    return flyCombinedData

#------------------------------------------------------------------------------
# function to save flyCombinedData dict to hdf5 file
def flyCombinedData_to_hdf5(flyCombinedData):
    filename = flyCombinedData['filename']
    filepath = flyCombinedData['filepath']
    
    save_filename = os.path.join(filepath,filename)
    
    with h5py.File(save_filename,'w') as f:
        #----------------------------------
        # time and parameters
        #----------------------------------
        f.create_dataset('Time/t', data=flyCombinedData['t'])
        f.create_dataset('Time/channel_t', data=flyCombinedData['channel_t'])
        f.create_dataset('Params/pix2cm', data=flyCombinedData['PIX2CM'])
        f.create_dataset('Params/trackingParams', 
                         data=str(flyCombinedData['trackingParams']))
        f.create_dataset('Params/boutParams', 
                         data=str(flyCombinedData['boutParams']))
        
        #----------------------------------
        # body velocity
        #----------------------------------      
        f.create_dataset('BodyVel/vel_x', data=flyCombinedData['xcm_vel'])
        f.create_dataset('BodyVel/vel_y', data=flyCombinedData['ycm_vel'])
        f.create_dataset('BodyVel/vel_mag', data=flyCombinedData['vel_mag'])       
        f.create_dataset('BodyVel/moving_ind', data=flyCombinedData['moving_ind'])
                         
        #----------------------------------
        # body position and image info
        #----------------------------------         
        f.create_dataset('BodyCM/xcm_smooth', data=flyCombinedData['xcm_smooth'])
        f.create_dataset('BodyCM/ycm_smooth', data=flyCombinedData['ycm_smooth'])
        f.create_dataset('BodyCM/cum_dist', data=flyCombinedData['cum_dist'])
        try:
            f.create_dataset('BodyCM/interp_idx', 
                             data=flyCombinedData['interp_idx'])
        except TypeError:
            f.create_dataset('BodyCM/interp_idx', data=np.nan)
                         
        # unprocessed data/information from tracking output file
        f.create_dataset('BodyCM/xcm', data=flyCombinedData['xcm'])
        f.create_dataset('BodyCM/ycm', data=flyCombinedData['ycm'])
        f.create_dataset('ROI/roi', data=flyCombinedData['ROI'])                 
        f.create_dataset('BG/bg', data=flyCombinedData['BG'])
        f.create_dataset('CAP_TIP/cap_tip', data=flyCombinedData['cap_tip'])
        f.create_dataset('CAP_TIP/cap_tip_orientation',
                         data=flyCombinedData['cap_tip_orientation'])
        try:
            f.create_dataset('BodyAngle/body_angle', 
                             data=flyCombinedData['body_angle'])  
        except TypeError:
            f.create_dataset('BodyAngle/body_angle', data=np.nan)   

        #----------------------------------
        # feeding data
        #----------------------------------   
        f.create_dataset('Feeding/dset_raw', data=flyCombinedData['channel_dset'])
        f.create_dataset('Feeding/dset_smooth', 
                         data=flyCombinedData['channel_dset_smooth'])   
        f.create_dataset('Feeding/bouts', data=flyCombinedData['bouts'])
        f.create_dataset('Feeding/volumes', data=flyCombinedData['volumes'])
        
        
#------------------------------------------------------------------------------
# function to load hdf5 file as flyCombinedData dict
def hdf5_to_flyCombinedData(filepath, filename):
    # file information    
    flyCombinedData = {'filepath' : filepath , 
                    'filename' : filename}
    
    data_prefix = os.path.splitext(filename)[0]
    xp_name = data_prefix.split('_')[-3]
    channel_name = 'channel_' + data_prefix.split('_')[-1]   
    
    # bank and channel names
    flyCombinedData['xp_name'] = xp_name
    flyCombinedData['channel_name'] = channel_name
    
    # tracking results
    filename_full = os.path.join(filepath,filename)
    filename_full = os.path.abspath(filename_full)
    with h5py.File(filename_full,'r') as f:
        
        #--------------------------
        # Image info
        #--------------------------
        flyCombinedData['ROI'] = f['ROI']['roi'].value
        flyCombinedData['cap_tip'] = f['CAP_TIP']['cap_tip'].value
        flyCombinedData['cap_tip_orientation'] = \
                        f['CAP_TIP']['cap_tip_orientation'].value
        flyCombinedData['BG'] = f['BG']['bg'].value
        
        #--------------------------
        # params
        #--------------------------
        flyCombinedData['PIX2CM'] = f['Params']['pix2cm'].value
        if (sys.version_info[0] < 3):
            flyCombinedData['trackingParams'] = ast.literal_eval(f['Params']['trackingParams'].value)
            flyCombinedData['boutParams'] = ast.literal_eval(f['Params']['boutParams'].value)
        
        #--------------------------
        # Time
        #--------------------------
        t = f['Time']['t'].value 
        channel_t = f['Time']['channel_t'].value
        
        flyCombinedData['frames'] = np.arange(len(t))
        flyCombinedData['t'] = t
        flyCombinedData['channel_frames'] = np.arange(len(channel_t))
        flyCombinedData['channel_t'] = channel_t
        
        #--------------------------
        # Trajectory info
        #--------------------------
        flyCombinedData['xcm'] = f['BodyCM']['xcm'].value
        flyCombinedData['ycm'] = f['BodyCM']['ycm'].value 
        flyCombinedData['xcm_smooth'] = f['BodyCM']['xcm_smooth'].value 
        flyCombinedData['ycm_smooth'] = f['BodyCM']['ycm_smooth'].value 
        flyCombinedData['cum_dist'] = f['BodyCM']['cum_dist'].value 
        try:
            flyCombinedData['interp_idx'] = f['BodyCM']['interp_idx'].value 
        except KeyError:
            flyCombinedData['interp_idx'] = np.nan
        
        flyCombinedData['xcm_vel'] = f['BodyVel']['vel_x'].value 
        flyCombinedData['ycm_vel'] = f['BodyVel']['vel_y'].value 
        flyCombinedData['vel_mag'] = f['BodyVel']['vel_mag'].value 
        flyCombinedData['moving_ind'] = f['BodyVel']['moving_ind'].value 
        
        try:
            flyCombinedData['body_angle'] = f['BodyAngle']['body_angle'].value
        except KeyError:
            flyCombinedData['body_angle'] = np.nan
        
        #--------------------------
        # Feeding data
        #--------------------------
        flyCombinedData['channel_dset'] = f['Feeding']['dset_raw'].value
        flyCombinedData['channel_dset_smooth'] = f['Feeding']['dset_smooth'].value
        flyCombinedData['bouts'] = f['Feeding']['bouts'].value
        flyCombinedData['volumes'] = f['Feeding']['volumes'].value
        
    return flyCombinedData

#------------------------------------------------------------------------------
# function to match up expresso channel and camera timing
def interp_channel_time(flyCombinedData):
    
    cam_t = flyCombinedData['t']
    channel_t = flyCombinedData['channel_t']
    dset_raw = flyCombinedData['channel_dset']
    dset_smooth = flyCombinedData['channel_dset_smooth']
    bouts = flyCombinedData['bouts']
    
    bouts_start_t = channel_t[bouts[0,:]]
    bouts_end_t = channel_t[bouts[1,:]]
    
    bouts_idx1_cam = np.zeros(bouts_start_t.shape,dtype=int) 
    bouts_idx2_cam = np.zeros(bouts_end_t.shape,dtype=int) 
    
    for ith in np.arange(len(bouts_idx1_cam)):
        bouts_idx1_cam[ith] = np.argmin(np.abs(bouts_start_t[ith] - cam_t))
        bouts_idx2_cam[ith] = np.argmin(np.abs(bouts_end_t[ith] - cam_t))
        
    bouts_cam = np.vstack((bouts_idx1_cam, bouts_idx2_cam))
    dset_cam = np.interp(cam_t, channel_t, dset_raw)
    dset_smooth_cam = np.interp(cam_t, channel_t, dset_smooth)
    
    return dset_cam, dset_smooth_cam, bouts_cam

#------------------------------------------------------------------------------
# function to calculate dwell time after each meal
def get_dwell_time(bouts, channel_t, dist_mag, vid_t,
                   fz_rad=analysisParams['food_zone_rad']):
    N_meals = bouts.shape[1]
    dwell_times = np.full((N_meals, 1), np.nan)
    censoring = np.full((N_meals, 1), 0)
    for meal_num in range(N_meals):
        # find ending of current meal
        meal_end_t = channel_t[bouts[1,meal_num]]
        
        # find first time after meal end that fly moves away from cap tip
        post_meal_idx = np.where((vid_t > meal_end_t) & (dist_mag > fz_rad))[0]
        
        if (post_meal_idx.size == 0):
            dwell_times[meal_num] = np.max(vid_t) - meal_end_t
            censoring[meal_num] = 1 
        else:
            leave_idx = post_meal_idx[0]
            dwell_times[meal_num] = vid_t[leave_idx] - meal_end_t 
            censoring[meal_num] = 0
            
    return (dwell_times, censoring)

#------------------------------------------------------------------------------
# function to plot feeding-bout-end aligned data
def plot_bout_aligned_var(basic_entries, var='vel_mag', window=300, 
                                N_meals=4,figsize=(6,10), saveFlag=False):
    
    data_suffix = '_COMBINED_DATA.hdf5'
    expt_idx, N_expt = group_expts_basic(basic_entries)
    cmap_name_list = ['Reds','Blues','Greens','Purples','Oranges','YlOrBr', \
                        'PuBu','YlOrRd','BuGn']
    cmap_mat = np.array([])
    for jth in np.arange(len(N_expt)):
        cmap_curr = cm.get_cmap(cmap_name_list[jth])
        c_val = np.arange(N_expt[jth],dtype=float) + 1.0
        normalized_c_val = c_val/np.max(c_val)
        color_vecs = cmap_curr(tuple(normalized_c_val))
        if jth == 0:
            cmap_mat = (color_vecs)
        else:
            cmap_mat = np.vstack((cmap_mat, color_vecs))
        
    # create plot
    fig, ax_arr = plt.subplots(N_meals,figsize=figsize,sharex=True, sharey=True)
    
    #expt_idx_counter = expt_idx[0]
    #cc = 0
    for ith, ent in enumerate(basic_entries):
        # load combined data file
        filename_full = ent + data_suffix
        filepath, filename = os.path.split(filename_full)
        flyCombinedData = hdf5_to_flyCombinedData(filepath, filename)
        
        # get color for plot
#        expt_idx_curr = expt_idx[ith]
#        if expt_idx_curr != expt_idx_counter:
#            cc = 0
#            color_vec_curr = cmap_list[expt_idx_curr][cc]
#        else:
#            color_vec_curr = cmap_list[expt_idx_curr][cc]
#            cc += 1
        color_vec_curr = cmap_mat[ith,:]
        
        
        # assign data and select variable to plot
        t = flyCombinedData['t']
        t_plot = t[:window] - t[0]
        if var == 'vel_mag':
            var_curr = flyCombinedData[var]
        elif var == 'dist_mag':
            var_curr = np.sqrt(flyCombinedData['xcm_smooth']**2 + \
                                    flyCombinedData['ycm_smooth']**2)
        else:
            print('do not have this option yet')
            return
        
        # since we're plotting tracking results, need bout index on cam time
        _, _, bouts_cam = interp_channel_time(flyCombinedData)
        
        # now loop through and plot
        N_meals_curr = np.min([bouts_cam.shape[1], N_meals])        
        for kth in np.arange(N_meals_curr):
            idx1 = bouts_cam[1,kth]
            idx2 = np.min([bouts_cam[1,kth] + window, len(t)-1])
            var_to_plot = var_curr[idx1:idx2]
            ax_arr[kth].plot(t_plot, var_to_plot, '-', color=color_vec_curr)
     
    for ax_num in np.arange(N_meals):
        if var == 'vel_mag':
            ax_arr[ax_num].set_ylabel('Speed (cm/s)')
        elif var == 'dist_mag':
            ax_arr[ax_num].set_ylabel('Distance from tip (cm)')
        else:
            print('do not have this option yet')
            return
        
        ax_arr[ax_num].set_title('Aligned to meal {} end'.format(ax_num+1))
        if ax_num == (N_meals - 1 ):
            ax_arr[ax_num].set_xlabel('Time (s)')
    
    return fig
#------------------------------------------------------------------------------
# function to save time series of combined data
def save_comb_time_series(data_filenames, savedir):
    print("Saving combined time series...")
    # loop through each data file
    for data_fn in data_filenames:
        if not os.path.exists(os.path.abspath(data_fn)):
            print(data_fn + ' not yet analyzed--failed to save')
        else:
            # load combined data file
            filepath, filename = os.path.split(data_fn)
            filename_noExt, _ = os.path.splitext(filename)
            flyCombinedData = hdf5_to_flyCombinedData(filepath, filename)
            
            # get save name for csv file
            csv_filename = os.path.join(savedir, filename_noExt + ".csv")
            column_headers=['Frame', 'Time (s)','Channel Raw (nL)', 
                            'Channel Smoothed (nL)', 'Feeding (bool)', 
                            'X Position (cm)','Y Position (cm)', 
                            'Cumulative Dist. (cm)','X Velocity (cm/s)',
                            'Y Velocity (cm/s)','Speed (cm/s)','Moving Idx']
            #---------------------------------------------------------
            # load in relevant kinematics data
            t = flyCombinedData['t']
            frames = np.arange(0,t.size)
            
            xcm = flyCombinedData['xcm_smooth']
            ycm = flyCombinedData['ycm_smooth']
            cum_dist = flyCombinedData['cum_dist']
            
            x_vel = flyCombinedData['xcm_vel']
            y_vel = flyCombinedData['ycm_vel']
            vel_mag = flyCombinedData['vel_mag']
            moving_ind = flyCombinedData['moving_ind']
            
            #---------------------------------------------------------
            # load in relevant bout data
            # (make sure that time values match up for bouts and tracking)
            dset_cam, dset_smooth_cam, bouts_cam = interp_channel_time(flyCombinedData)
            
            # generate array with same length as t that has "1" if fly is 
            # feeding and "0" otherwise
            feeding_boolean = np.zeros([1, dset_cam.size])
            for i in np.arange(bouts_cam.shape[1]):
                feeding_boolean[0, bouts_cam[0, i]:bouts_cam[1, i]] = 1
            
            #---------------------------------------------------------
            # combine data into one matrix 
            row_mat = np.vstack((frames, t, dset_cam, dset_smooth_cam, 
                                 feeding_boolean, xcm, ycm, cum_dist, x_vel, 
                                 y_vel, vel_mag, moving_ind ))
            row_mat = np.transpose(row_mat)
            
            #------------------------------------------------------------------
            # generate csv file, write column headings, then write data in rows
            if sys.version_info[0] < 3:
                out_path = open(csv_filename,mode='wb')
            else:
                out_path = open(csv_filename, 'w', newline='')
            
            # csv writer object    
            save_writer = csv.writer(out_path)
            
            # write headings for filename + data column labels
            save_writer.writerow([filepath])
            save_writer.writerow(column_headers)
            
            # write data rows
            for row in row_mat:
                save_writer.writerow(row)
                
            out_path.close()
            print("Done saving {}".format(csv_filename))
            
    print("Completed saving time series data")
#------------------------------------------------------------------------------
# function to save time series of combined data
def save_comb_summary(entry_list, xlsx_filename, 
                      fz_rad=analysisParams['food_zone_rad']):
    print('Saving to {} ...'.format(xlsx_filename))         
    # info for converting file types
    data_suffix = '_COMBINED_DATA.hdf5'
    
    # variables to put in summary page
    summary_heading = ['Filename', 'Bank', 'Channel',  'Number of Meals', 
                       'Total Volume (nL)', 'Total Duration Eating (s)',
                        'Latency to Eat (s)', 'Cumulative Dist. (cm)',
                        'Average Speed (cm/s)', 'Fraction Time Moving', 
                        'Pre Meal Dist. (cm)', 'Food Zone Frac. (pre meal)',
                        'Food Zone Frac. (post meal)']    
    
    # variables to put in events page
    events_heading = ['Filename', 'Bank', 'Channel', 'Meal Number', 
                      'Start Time (s)', 'End Time (s)', 'Duration (s)', 
                      'Volume (nL)', 'Dwell Time (s)', 
                      'Dwell Time Censoring (bool)']
    
    # ---------------------------------
    # INITIALIZE WORKBOOK
    # ---------------------------------
    wb = Workbook()    
    
    ws_summary = wb.active
    ws_summary.title = "Summary"
    ws_events = wb.create_sheet("Events")
    
    # add column headers to pages
    ws_summary.append(summary_heading)
    ws_events.append(events_heading)
    # -------------------------------------------------------
    # LOOP THROUGH DATA AND WRITE TO XLSX FILE
    # -------------------------------------------------------
    for ent in entry_list:
        filename_full = ent + data_suffix
        if not os.path.exists(os.path.abspath(filename_full)):
            # in case we don't have analysis for this file yet
            print(ent + ' not yet analyzed--failed to save')
        else:
            # ------------------------------------------------------------
            # load combined data file
            filepath, filename = os.path.split(filename_full)
            flyCombinedData = hdf5_to_flyCombinedData(filepath, filename)
            filename_noExt, _ = os.path.splitext(filename)
            
            # get bank and channel info
            xp_regex = "XP\d\d"
            channel_regex = "channel_\d"
            xp_name = re.findall(xp_regex,filename_noExt)[0]
            channel_name = re.findall(channel_regex,filename_noExt)[0]
            
            # -------------------------------------------
            # load in relevant bout data (feeding)
            bouts = flyCombinedData['bouts']
            #bouts = np.asarray(bouts)
            volumes = flyCombinedData['volumes']
            channel_t = flyCombinedData['channel_t']
            
            num_meals = float(bouts.shape[1])
            total_volume = np.sum([float(vol) for vol in volumes])
            
            if (num_meals < 1):
                latency = -1.0
                duration_eating = 0.0 
                bout_start_t = np.nan
                bout_end_t = np.nan
                
            else:
                bout_start_t = [channel_t[bouts[0,ith]] for ith in range(bouts.shape[1])]
                bout_end_t = [channel_t[bouts[1,ith]] for ith in range(bouts.shape[1])]
                latency = bout_start_t[0]
                duration_eating = np.sum(np.asarray(bout_end_t) - np.asarray(bout_start_t))
            
            #---------------------------------------------------------
            # load in relevant kinematics data (tracking)
            
            cum_dist = flyCombinedData['cum_dist']
            vel_mag = flyCombinedData['vel_mag']
            moving_ind = flyCombinedData['moving_ind']
            vid_t = flyCombinedData['t']
            dist_mag = np.sqrt(flyCombinedData['xcm_smooth']**2 + 
                        flyCombinedData['ycm_smooth']**2)
            
            avg_speed = np.nanmean(vel_mag[moving_ind])
            cum_dist_max = cum_dist[-1]
            frac_moving = float(np.sum(moving_ind)) / float(moving_ind.size)
            
            # ---------------------------------------------------------------
            # also get leave times, fraction of time spent in food zone 
            # (before and after first meal), and distance walked prior to first
            # meal
            dwell_times, censoring = get_dwell_time(bouts, channel_t, 
                                                    dist_mag, vid_t, 
                                                    fz_rad=fz_rad)
            
            if (num_meals < 1):
                pre_meal_pathlength = cum_dist_max 
                fz_idx = (dist_mag <= fz_rad)
                fz_frac_pre = float(np.sum(fz_idx))/float(dist_mag.size )
                fz_frac_post = 0.0 
            else:
                first_meal_ind = bouts[0,0]
                pre_meal_pathlength = cum_dist[first_meal_ind-1]
                fz_idx = (dist_mag <= fz_rad)
                pre_ind = (vid_t < bout_start_t[0])
                post_ind = (vid_t >= bout_start_t[0])
                fz_frac_pre = float(np.sum(fz_idx & pre_ind))/float(dist_mag.size)
                fz_frac_post = float(np.sum(fz_idx & post_ind))/float(dist_mag.size)
            #---------------------------------------------------------
            # WRITE DATA TO XLSX FILE
            #---------------------------------------------------------
            # summary info (one line per fly)
            row_list = [filename_noExt, xp_name, channel_name, 
                                 num_meals, total_volume, duration_eating, 
                                 latency, cum_dist_max, avg_speed, frac_moving,
                                 pre_meal_pathlength,fz_frac_pre,fz_frac_post]

            ws_summary.append(row_list)
            
            # event info (multiple lines per fly)
            for jth in range(bouts.shape[1]):
                bout_start_curr = bout_start_t[jth]
                bout_end_curr = bout_end_t[jth]
                volume_curr = float(volumes[jth])
                duration_curr = bout_end_curr - bout_start_curr 
                
                row_curr = [filename_noExt, xp_name, channel_name, jth+1, 
                            bout_start_curr, bout_end_curr,duration_curr, 
                            volume_curr, float(dwell_times[jth]), 
                            float(censoring[jth])]
                ws_events.append(row_curr) 
    
    # -------------------------------------
    # SAVE XLSX FILE
    # -------------------------------------           
    wb.save(xlsx_filename)
    print('Completed saving {}'.format(xlsx_filename))           
            
                
         
            
            

        
    