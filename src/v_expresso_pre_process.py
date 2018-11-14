# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:31:17 2018

Crops MULTIPLE large Expresso videos into individual channel videos

@author: Fruit Flies
"""

#------------------------------------------------------------------------------
import os
import numpy as np
import cv2
import h5py
import progressbar

from v_expresso_image_lib import (get_roi, get_cropped_im, get_cap_tip,
                                  get_pixel2cm)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def roi_and_cap_tip_single_video(filepath, xp_names, channel_numbers):
    
    # get path information for loading/saving data    
    dirpath, filename = os.path.split(filepath)
    savepath = dirpath
    
    # open video and get vid info
    cap = cv2.VideoCapture(filepath)
    data_prefix = os.path.splitext(filename)[0]
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    
    # convert channel numbers into channel names
    channel_names = []
    for ch_list in channel_numbers:
        channel_name_curr = ["channel_" + str(ch) for ch in ch_list]
        channel_names.append(channel_name_curr)
    #initialize list for ROIs and cap_tips
    roi_list = [] 
    cap_tip_list = []
    cap_tip_orientation_list = []
    
    #get base image from which to select ROIs
    _, frame = cap.read(1)
    im0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im_for_roi = im0.copy()    
    
    # get ROI locations
    roi_counter = 0
    for (ith, xp_name) in enumerate(xp_names):
        for channel_name in channel_names[ith]:
            frame_header= "Select ROI for " + data_prefix +  xp_name + ", " + channel_name
            print(frame_header)
            
            # get ROI
            r = get_roi(im0,frame_header,fullScreenFlag=True)
            roi_list.append(r)
            im0[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 0  #black out
            
            # get cap tip
            im_roi = get_cropped_im(im_for_roi,r) 
            cap_tip = get_cap_tip(im_roi)
            cap_tip_list.append(np.float32(cap_tip))
            
            # get cap tip orientation
            roi_mid_x = r[2]/2.0
            roi_mid_y = r[3]/2.0
            
            #vial is oriented verically relative to camera coordinates
            if r[3] > r[2]:
                # capillary tip is at the top of the image (lower y)
                if cap_tip[1] < roi_mid_y:
                    cap_tip_orientation_list.append('T')
                # capillary tip is at the bottom of the image (higher y)    
                else:
                    cap_tip_orientation_list.append('B')
            
            #vial is oriented horizontally relative to camera coordinates
            else:
                # capillary tip is on the left 
                if cap_tip[0] < roi_mid_x:
                    cap_tip_orientation_list.append('L')
                                     
                # capillary tip is on the right     
                else:
                    cap_tip_orientation_list.append('L')
            
            # get pixel to cm conversion ; only for first video            
            if (roi_counter == 0):
                im_for_meas = im_roi.copy()
                PIX2CM = get_pixel2cm(im_for_meas)
                
            roi_counter += 1 
            
    cap.release()
    
    # save results to hdf5 file
    save_filename = os.path.join(savepath,data_prefix + "_VID_INFO.hdf5")
    
    cc = 0
    with h5py.File(save_filename,'w') as f:
        f.create_dataset('Params/FPS', data=FPS)
        f.create_dataset('Params/N_FRAMES', data=N_FRAMES)
        f.create_dataset('Params/PIX2CM', data=PIX2CM)
        for (ith, xp_name) in enumerate(xp_names):
            for channel_name in channel_names[ith]:
                f.create_dataset('ROI/' + xp_name + '_' + channel_name, 
                                 data=roi_list[cc])
                f.create_dataset('CAP_TIP/' + xp_name + '_' + channel_name, 
                                 data=cap_tip_list[cc])
                f.create_dataset('CAP_TIP_ORIENTATION/' + xp_name + '_' + channel_name,
                                 data=cap_tip_orientation_list[cc])
                
                cc += 1 
    
    #return (roi_list, cap_tip_list, cap_tip_orientation_list, FPS, N_FRAMES, PIX2CM)

#------------------------------------------------------------------------------
def crop_and_save_single_video(filepath, xp_names, channel_numbers):
    
    # get path information for loading/saving data    
    dirpath, filename = os.path.split(filepath)
    savepath = dirpath
    data_prefix = os.path.splitext(filename)[0]
    vid_info_filename = os.path.join(savepath,data_prefix + "_VID_INFO.hdf5")
    
    # check if ROIs etc have been defined
    if not os.path.exists(vid_info_filename):
        print('Error: no pre-processed information for this video:')
        print(vid_info_filename)
        return
    
    # get channel names from numbers
    channel_names = []
    for ch_list in channel_numbers:
        channel_name_curr = ["channel_" + str(ch) for ch in ch_list]
        channel_names.append(channel_name_curr)
    
    #=================================================
    # read video info from hdf5 files 
    #=================================================
    roi_list = []
    
    with h5py.File(vid_info_filename,'r') as f:
        FPS = f['Params']['FPS'].value
        N_FRAMES = f['Params']['N_FRAMES'].value
        #PIX2CM = f['Params']['PIX2CM'].value
        
        for (ith, xp_name) in enumerate(xp_names):
            for channel_name in channel_names[ith]:
                roi_curr = f['ROI'][xp_name + '_' + channel_name].value
                roi_list.append(roi_curr)
    #=================================================
    # define list of video writers
    #=================================================
    writer_list = []
    cap = cv2.VideoCapture(filepath)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
   
    for (xp_num,xp_name) in enumerate(xp_names):
       for (ch_num, ch_name) in enumerate(channel_names[xp_num]):
            idx = xp_num*len(channel_names[xp_num])+ch_num
            #print(idx)
            roi_curr = roi_list[idx]        
            output_name = data_prefix + '_' + xp_name + "_" + \
                            ch_name + '.avi'
            output_path = os.path.join(savepath, output_name)  
            
            
            writer = cv2.VideoWriter(output_path,fourcc,FPS,
                                     (int(roi_curr[2]), int(roi_curr[3])))
            writer_list.append(writer)                         
                                     
    #=================================================
    # loop through video and grab/save cropped images
    #=================================================
    
    cc = 0
    widgets = [progressbar.FormatLabel('Processing ' + data_prefix), ' ', 
               progressbar.Percentage(), ' ',
               progressbar.Bar('/'), ' ', progressbar.RotatingMarker()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=(N_FRAMES-1))
    pbar.start()
    while cc < (N_FRAMES-1):
        ret, frame = cap.read()
        
        if not ret:
            print('Error reading frame')
            break
        
        for (xp_num,xp_name) in enumerate(xp_names):
            for (ch_num, ch_name) in enumerate(channel_names[xp_num]):
                
                idx = xp_num*len(channel_names[xp_num])+ch_num
                r = roi_list[idx]        
                frame_copy = frame.copy()
                frame_crop = frame_copy[int(r[1]):int(r[1]+r[3]),
                                        int(r[0]):int(r[0]+r[2]),:]
                
                vid_writer = writer_list[idx]
                vid_writer.write(frame_crop)
                
        pbar.update(cc)
        cc += 1
    
    pbar.finish()            
    
    #=================================================
    # close video writers
    #=================================================    
    for writer in writer_list:
        writer.release()
        
    cap.release()
    
#----------------------------------------------


