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
    dir_list = os.path.dirname(basic_entry_list)
    #dir_list_unique = list(set(dir_list))
    
    expt_idx = []
    for dr in dir_list:
        idx = [i for i, j in enumerate(dir_list) if j == dr]
        expt_idx.append(idx)
        
    return expt_idx
    
#------------------------------------------------------------------------------
# function to check bouts based on tracking data
def bout_analysis_wTracking(filename, bank_name, channel_name, 
                            bout_params=analysisParams, saveFlag=False, 
                            plotFlag=False, debugBoutFlag=False,
                            debugTrackingFlag=False):
    
    MOVE_FRAC_THRESH = bout_params['feeding_move_frac_thresh']
    DIST_THRESH = bout_params['feeding_dist_thresh']
    dist_prctile_level = 75
    #--------------------------------
    # load channel data and get bouts
    #--------------------------------
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
    #vel_mag = flyTrackData['vel_mag']
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
        moving_chk_bool = (moving_chk < MOVE_FRAC_THRESH)
        
        # check that fly is close to cap tip
        dist_mag_bout = dist_mag[vid_start_idx:vid_end_idx]
        dist_mag_prctile = np.percentile(dist_mag_bout,dist_prctile_level)
        dist_check_bool = (dist_mag_prctile < DIST_THRESH) 
        
        bout_check[ith] = moving_chk_bool & dist_check_bool #& moving_chk_bool #& 
        
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
                         data=str(flyCombinedData['PARAMS']))
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
        except KeyError:
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
        except KeyError:
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
# function to save time series of combined data
def save_comb_time_series(data_filenames):
    print('under construction')

#    for data_fn in data_filenames:
#        if not os.path.exists(os.path.abspath(data_fn)):
#            print(data_fn + ' not yet analyzed--failed to save')
#        else:
#
#        filepath = os.path.splitext(h5_fn)[0]  
#        with h5py.File(data_fn,'r') as f:
#            csv_filename = filepath + ".csv" 
#            t = f['Time']['t'].value
#            
#            # get kinematics
#            try:
#                xcm = f['BodyCM']['xcm_smooth'].value 
#                ycm = f['BodyCM']['ycm_smooth'].value 
#                cum_dist = f['BodyCM']['cum_dist'].value 
#                x_vel = f['BodyVel']['vel_x'].value 
#                y_vel = f['BodyVel']['vel_y'].value 
#                vel_mag = f['BodyVel']['vel_mag'].value 
#                moving_ind = f['BodyVel']['moving_ind'].value 
#                
#                column_headers=['Time (s)','X Position (cm)','Y Position (cm)', 
#                            'Cumulative Dist. (cm)','X Velocity (cm/s)',
#                            'Y Velocity (cm/s)','Speed (cm/s)','Moving Idx']
#                row_mat = np.vstack((t, xcm, ycm, cum_dist, x_vel, y_vel, 
#                                     vel_mag, moving_ind))
#                row_mat = np.transpose(row_mat) 
#            except KeyError:
#                xcm = f['BodyCM']['xcm'].value 
#                xcm = f['BodyCM']['ycm'].value 
#                column_headers = ['Time (s)', 'X CM (cm)', 'Y CM (cm)']
#                row_mat = np.vstack((t, xcm, ycm))
#                row_mat = np.transpose(row_mat) 
#
#        if sys.version_info[0] < 3:
#            out_path = open(csv_filename,mode='wb')
#        else:
#            out_path = open(csv_filename, 'w', newline='')
#        save_writer = csv.writer(out_path)
#        
#        save_writer.writerow([filepath])
#        save_writer.writerow(column_headers)
#           
#        for row in row_mat:
#            save_writer.writerow(row)
#            
#        out_path.close()
#    