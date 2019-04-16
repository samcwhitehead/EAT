# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:30:47 2019

Script/function to allow users to manually adjust their estimate of the 
capillary tip location
 
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


from v_expresso_image_lib import (get_cap_tip, hdf5_to_flyTrackData, 
                                  invert_coord_transform,
                                  perform_coord_transform)
#------------------------------------------------------------------------------
# MAIN FUNCTION
#------------------------------------------------------------------------------
def refine_tip(data_filename_full):    
    data_dir, data_filename = os.path.split(data_filename_full)
    
    # get file paths for data and video info files
    data_filename_split = data_filename.split('_')
    data_header = '_'.join(data_filename_split[:-2])
    hdf5_name = '_'.join(data_filename_split[:-5])
    bank_name = data_filename_split[-5]
    channel_name = '_'.join(data_filename_split[-4:-2])
    vid_info_name = os.path.join(data_dir, hdf5_name + '_VID_INFO.hdf5')
    
    print('Re-doing capillary tip estimation for {}'.format(data_header))
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
    
    #--------------------------------------------------------------------------
    # recalculate the tracking data with new capillary tip
    print('Re-centering tracking data...')
    xcm_old = flyTrackData['xcm']
    ycm_old = flyTrackData['ycm']
    xcm_smooth_old = flyTrackData['xcm_smooth']
    ycm_smooth_old = flyTrackData['ycm_smooth']
    cap_tip_orient = flyTrackData['cap_tip_orientation']    
    pix2cm = flyTrackData['PIX2CM']
    #ROI = flyTrackData['ROI']
    
    # -------------------------------------
    # undo original coordinate transform 
    xcm_pix, ycm_pix = invert_coord_transform(xcm_old, ycm_old, pix2cm, 
                                              cap_tip_old, cap_tip_orient)
    xcm_smooth_pix, ycm_smooth_pix = invert_coord_transform(xcm_smooth_old, 
                                                            ycm_smooth_old, 
                                                            pix2cm, cap_tip_old,
                                                            cap_tip_orient)
    # ... and then redo with new cap tip                                            
    xcm_new, ycm_new = perform_coord_transform(xcm_pix, ycm_pix, pix2cm,
                                               cap_tip_new,cap_tip_orient)
    xcm_smooth_new, ycm_smooth_new = perform_coord_transform(xcm_smooth_pix, 
                                                            ycm_smooth_pix, 
                                                            pix2cm, cap_tip_new,
                                                            cap_tip_orient)    
    #--------------------------------------------------------------------------
    # save these results
    print('saving new results...')
    with h5py.File(data_filename_full,'r+') as f:
        xcm = f['BodyCM/xcm']
        xcm[...] = xcm_new
        ycm = f['BodyCM/ycm']
        ycm[...] = ycm_new
        
        xcm_smooth = f['BodyCM/xcm_smooth']
        xcm_smooth[...] = xcm_smooth_new
        ycm_smooth = f['BodyCM/ycm_smooth']
        ycm_smooth[...] = ycm_smooth_new
        
    print('Completed update')
    
    
# -----------------------------------------------------------------------------
# Run main function
     # -----------------------------------------------------------------------------     
if __name__ == '__main__':
    # allow users to select which files to adjust
    data_filename_full_list = tkFileDialog.askopenfilenames(initialdir=sys.path[0],
                              title='Select *_TRACKING_PROCESSED.hdf5 to refine tip') 
    # re-do tip location estimate
    for data_fn in data_filename_full_list:
        refine_tip(data_fn)
    