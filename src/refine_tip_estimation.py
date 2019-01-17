# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:30:47 2019

@author: Fruit Flies
"""
#------------------------------------------------------------------------------
import sys
import os 
import h5py
import cv2

if sys.version_info[0] < 3:
    #from Tkinter import *
    import tkFileDialog
else:
    from tkinter import filedialog as tkFileDialog


from v_expresso_image_lib import get_cap_tip, hdf5_to_flyTrackData

#------------------------------------------------------------------------------
# select file to refine tip of 
#data_dir = 'H:/v_expresso data/Feeding_annotation_videos/'
#data_filename = '83_batch3_XP05_channel_5_TRACKING_PROCESSED.hdf5'

data_filename_full_list = tkFileDialog.askopenfilenames(initialdir=sys.path[0],
                              title='Select *_TRACKING_PROCESSED.hdf5 to refine tip') 

#------------------------------------------------------------------------------

for data_filename_full in data_filename_full_list:
    
    data_dir, data_filename = os.path.split(data_filename_full)
    
    # get file paths for data and video info files
    data_filename_split = data_filename.split('_')
    data_header = '_'.join(data_filename_split[:-2])
    hdf5_name = '_'.join(data_filename_split[:-5])
    bank_name = data_filename_split[-5]
    channel_name = '_'.join(data_filename_split[-4:-2])
    vid_info_name = os.path.join(data_dir, hdf5_name + '_VID_INFO.hdf5')
    
    #--------------------------------------------------------------------------
    # open a new window to reselect the cap tip location
    flyTrackData = hdf5_to_flyTrackData(data_dir,data_filename)
    BG = flyTrackData['BG']
    cap_tip_old = flyTrackData['cap_tip']
    print('old cap tip:')
    print(cap_tip_old)
    
    BG = cv2.equalizeHist(BG)
    cap_tip_new = get_cap_tip(BG)
    
    print('new cap tip value:')
    print(cap_tip_new)
    
    #--------------------------------------------------------------------------
    # write the new value of the cap tip into the VID_INFO file
    if os.path.exists(vid_info_name):
        with h5py.File(vid_info_name,'r+') as f:
            cap_tip = f['CAP_TIP/' + bank_name + '_' + channel_name]
            cap_tip[...] = cap_tip_new
            
        
    # check that it read out correctly
    if os.path.exists(vid_info_name):
        with h5py.File(vid_info_name,'r') as f:
            cap_tip_test = f['CAP_TIP/' + bank_name + '_' + channel_name].value
            print('The saved value is:')
            print(cap_tip_test)