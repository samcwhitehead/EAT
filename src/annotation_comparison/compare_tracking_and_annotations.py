# -*- coding: utf-8 -*-
"""
Created on Mon Oct 01 09:03:42 2018

@author: Fruit Flies

Script to compare the output of the automated fly tracking in the visual 
Expresso system with user-annotated video frames. 

"""
#------------------------------------------------------------------------------
import os
#import sys
import cv2
import numpy as np
import h5py
import fnmatch
import matplotlib.pyplot as plt

from v_expresso_gui_params import trackingParams
from v_expresso_image_lib import (visual_expresso_main, process_visual_expresso) 
                                   
#------------------------------------------------------------------------------

DATA_PATH = 'H:/v_expresso data/ANNOTATED_VID_FRAMES/' 
ANN_SUFFIX = 'VID_ANNOTATION.hdf5'
SAVE_PATH = 'H:/v_expresso data/ANNOTATED_VID_FRAMES/Summary/'
 
RUN_TRACKING_FLAG = False

VIZ_FLAG = False
SAVE_VIZ_FLAG = False

PLOT_FLAG = True
HIST_FLAG = True
CDF_FLAG  = True 
SAVE_PLOTS_FLAG = True

# params
DELTA = 10 
#------------------------------------------------------------------------------

# find all annotation files within annotation directory
annotation_filenames = [] 
for root, dirs, files in os.walk(DATA_PATH, topdown=False):
   for name in files:
      #print(os.path.join(root, name))
      if ANN_SUFFIX in name:
          annotation_filenames.append(os.path.abspath(os.path.join(root, name)))
          
#------------------------------------------------------------------------------
# exclude frames list
exclude_frames =   {'0hr_1mM_batch1_XP04_channel_4' : [],
                    '0hr_10mM_batch3_XP05_channel_2' : [3, 26, 30, 34, 40, 92],
                    '0hr_1000mM_batch3_XP04_channel_5' : [],
                    '8hr_10mM_batch2_XP04_channel_3' : [],
                    '8hr_100mM_batch1_XP05_channel_5' : [],
                    '8hr_1000mM_batch2_XP05_channel_1' : [],
                    '16hr_1mM_batch3_XP04_channel_4' : [],
                    '16hr_10mM_batch1_XP05_channel_4' : [28, 35, 54, 67],
                    '16hr_100mM_batch3_XP04_channel_4' : [],
                    '16hr_100mM_batch3_XP05_channel_5' : [14, 26, 32, 76, 86, 96],
                    '24hr_1mM_batch1_XP05_channel_1' : [],
                    '24hr_100mM_batch2_XP04_channel_1' : [8,19,64],
                    '24hr_1000mM_batch3_XP04_channel_1' : [],
                    '32hr_1mM_batch2_XP04_channel_3' : [],
                    '32hr_10mM_batch1_XP04_channel_3' : [],
                    '32hr_10mM_batch1_XP05_channel_1' : [49],
                    '32hr_1000mM_batch2_XP05_channel_2' : [22,74,81],
                    '40hr_1mM_batch1_XP04_channel_1' : [17, 20, 22, 38, 46, 48,
                                                        53, 63],
                    '40hr_1mM_batch1_XP05_channel_3' : [2, 5, 6, 9, 10, 11, 14,
                                                        15, 17, 18, 20, 25, 30,
                                                        33, 34, 46, 49, 50, 51, 
                                                        56, 59,
                                                        60, 62, 63, 67, 72, 76,
                                                        77, 80, 83, 91, 92, 99],
                    '40hr_100mM_batch3_XP04_channel_3' : []
                    }
#------------------------------------------------------------------------------
# if tracking analysis has not already been run on videos, run it here
if RUN_TRACKING_FLAG:
    for ann_fn in annotation_filenames:
        #split file path to get video filename
        dir_curr, fn_curr = os.path.split(ann_fn)
        fn_curr_split = fn_curr.split('_')
        bank_name = fnmatch.filter(fn_curr_split,'XP*')[0]
        
        # all the filenames are variations on each other--here we're just 
        # looking for places to cut off the string and switch the form a little 
        idx1 = fn_curr_split.index('VID')
        idx2 = fn_curr_split.index(bank_name)
        
        vid_fn = '_'.join(fn_curr_split[:idx1]) + '.avi'
        vid_info_fn = '_'.join(fn_curr_split[:idx2]) + '_VID_INFO.hdf5'
        
        vid_fn_full = os.path.join(dir_curr,vid_fn)
        vid_info_fn_full = os.path.join(dir_curr,vid_info_fn)
        
        # if files exist, run analysis
        if os.path.exists(vid_fn_full) and os.path.exists(vid_info_fn_full):
            file_path, filename = os.path.split(vid_fn_full)
            flyTrackData = visual_expresso_main(file_path, filename, 
                            DEBUG_BG_FLAG=False, DEBUG_CM_FLAG=False, 
                            SAVE_DATA_FLAG=True, ELLIPSE_FIT_FLAG = False, 
                            PARAMS=trackingParams)
                            
            filename_prefix = os.path.splitext(filename)[0]
            track_filename = filename_prefix + "_TRACKING.hdf5"  
                      
            flyTrackData_smooth = process_visual_expresso(file_path, track_filename,
                                SAVE_DATA_FLAG = True, DEBUG_FLAG = False)
                                
            print('Completed tracking for %s \n' % (vid_fn_full))
           
#------------------------------------------------------------------------------
# loop through movies and compare with annotations
           
xcm_diff_all = [] 
ycm_diff_all = []
ann_frame_num_all = [] 

machine_xcm_all = [] 
machine_ycm_all = [] 
ann_xcm_all = [] 
ann_ycm_all = [] 
exclude_frames_all = [] 
file_ID_all = [] 

for ann_file in annotation_filenames:
    
    # get corresponding machine tracking filename
    dir_curr, fn_curr = os.path.split(ann_file)
    fn_curr_split = fn_curr.split('_')
    idx = fn_curr_split.index('VID')
    
    machine_fn = '_'.join(fn_curr_split[:idx]) + "_TRACKING_PROCESSED.hdf5"  
    machine_fn_full = os.path.join(dir_curr,machine_fn)

    # get file ID info so that we can exclude the correct frames 
    vid_prefix = '_'.join(fn_curr_split[:idx])   
    time_dir, conc_folder = os.path.split(dir_curr)
    parent_dir, time_folder = os.path.split(time_dir)
    condition_prefix = time_folder + '_' + conc_folder
    
    exclude_frame_ID = condition_prefix + '_' + vid_prefix
            
    # load annotation data    
    with h5py.File(ann_file,'r') as f:
        ann_frame_num  = f['Time']['frame_num'].value
        ann_xcm = f['BodyCM']['xcm'].value
        ann_ycm = f['BodyCM']['ycm'].value
    
    # load machine data
    with h5py.File(machine_fn_full,'r') as f:
       dataset_names = list(f.keys())
    
    if os.path.exists(machine_fn_full):
        with h5py.File(machine_fn_full,'r') as f:
            machine_t = f['Time']['t'].value
            machine_frame_num = (machine_t/np.mean(np.diff(machine_t)))
            machine_frame_num = machine_frame_num.astype('int32')
            machine_xcm = f['BodyCM']['xcm_smooth'].value
            machine_ycm = f['BodyCM']['ycm_smooth'].value
            
            cap_tip_orientation = f['CAP_TIP']['cap_tip_orientation'].value
            cap_tip = f['CAP_TIP']['cap_tip'].value
            PIX2CM = f['Params']['pix2cm'].value
            
    else:
        print('No data file found--skipping')
        continue
    
    # undo coordinate transformation
    
    if cap_tip_orientation == 'T':
        machine_xcm_pix = machine_xcm/PIX2CM + cap_tip[0] 
        machine_ycm_pix = machine_ycm/PIX2CM + cap_tip[1]
    elif cap_tip_orientation == 'B': 
        machine_xcm_pix = -1*machine_xcm/PIX2CM + cap_tip[0]
        machine_ycm_pix = -1*machine_ycm/PIX2CM + cap_tip[1] 
    elif cap_tip_orientation == 'L':   
        machine_xcm_pix = machine_ycm/PIX2CM + cap_tip[0] 
        machine_ycm_pix = machine_xcm/PIX2CM + cap_tip[1]
    else:
        machine_xcm_pix = -1*machine_ycm/PIX2CM + cap_tip[0] 
        machine_ycm_pix = -1*machine_xcm/PIX2CM + cap_tip[1] 

    #===================================================
    # compare annotated data points to machine
    delta_x = ann_xcm - machine_xcm_pix[ann_frame_num - DELTA]
    delta_y = ann_ycm - machine_ycm_pix[ann_frame_num - DELTA]
    
    # get current frames to exclude
    exclude_frames_curr = exclude_frames[exclude_frame_ID]
    
    # add comparisons from each movie to larger list
    xcm_diff_all.append(np.asarray(delta_x,dtype=np.float32))
    ycm_diff_all.append(np.asarray(delta_y,dtype=np.float32))
    ann_frame_num_all.append(ann_frame_num)
    
    machine_xcm_all.append(machine_xcm_pix[ann_frame_num - DELTA])
    machine_ycm_all.append(machine_ycm_pix[ann_frame_num - DELTA])
    ann_xcm_all.append(ann_xcm)
    ann_ycm_all.append(ann_ycm)
    exclude_frames_all.append(exclude_frames_curr)
    file_ID_all.append(exclude_frame_ID)
    # give some indication of progress
    #print('Completed comparison for %s \n' % (ann_file))
    
    
#------------------------------------------------------------------------------
if VIZ_FLAG:
    ALPHA = 0.5 
    mov_num_list = np.arange(len(xcm_diff_all))
    #mov_num_list = [0]
    for mov_num in mov_num_list:
        ann_fn = annotation_filenames[mov_num]
        ann_frames_curr = ann_frame_num_all[mov_num] 
        
        ann_xcm_curr = ann_xcm_all[mov_num]
        ann_ycm_curr = ann_ycm_all[mov_num]
        machine_xcm_curr = machine_xcm_all[mov_num]
        machine_ycm_curr = machine_ycm_all[mov_num]
        
        dir_curr, fn_curr = os.path.split(ann_fn)
        fn_curr_split = fn_curr.split('_')
        #bank_name = fnmatch.filter(fn_curr_split,'XP*')[0]
        idx1 = fn_curr_split.index('VID')
        #idx2 = fn_curr_split.index(bank_name)
        
        vid_prefix = '_'.join(fn_curr_split[:idx1])
        vid_fn = vid_prefix + '.avi'
        vid_filename = os.path.join(dir_curr, vid_fn)
        
        
        
        # prepare savepath, in case images are to be saved
        if SAVE_VIZ_FLAG:
            time_dir, conc_folder = os.path.split(dir_curr)
            parent_dir, time_folder = os.path.split(time_dir)
            folder_name = time_folder + '_' + conc_folder
            new_dir = os.path.join(parent_dir,'Summary', folder_name)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
        
        # open video and read frames
        cap = cv2.VideoCapture(vid_filename)
        cv2.namedWindow('Annotation comparison',cv2.WINDOW_NORMAL)
        
        ith = 0    
        while ith < len(ann_frames_curr):
            
            cap.set(1,ann_frames_curr[ith])
            ret, frame = cap.read()
            
            if ret:
                frame_overlay = frame.copy()
                if not np.isnan(ann_xcm_curr[ith]):
                    cv2.circle(frame_overlay,(int(ann_xcm_curr[ith]), 
                                              int(ann_ycm_curr[ith])),
                                               1,[0,255,0],-1)
                else:
                    print('no annotation value for this frame')
                cv2.circle(frame_overlay,(int(machine_xcm_curr[ith]), 
                                          int(machine_ycm_curr[ith])),
                                           1,[0,0,255],-1)
                
                frame_sum = cv2.addWeighted(src1=frame_overlay, alpha=ALPHA, src2=frame,
                                            beta=1. - ALPHA, gamma=0, dst=frame_overlay)
                cv2.imshow('Annotation comparison',frame_sum)
                
                if SAVE_VIZ_FLAG:
                    frame_filename = vid_prefix + \
                                        '_frame_%03d'%(ith) + '.png'
                    frame_filepath_full = os.path.join(new_dir,frame_filename)
                    cv2.imwrite(frame_filepath_full,frame_sum)
                    cv2.waitKey(1)
                else:    
                    k = cv2.waitKey(0) & 0xff
                    if k == 27:
                        break
            else:
                print('error loading frame')
            
            ith += 1
        
        #cv2.destroyAllWindows()
        cap.release()

#------------------------------------------------------------------------------
if PLOT_FLAG:
    
    # collect data together into list
    mov_num_list = np.arange(len(xcm_diff_all))
    data_to_plot_list = [] 
    #mov_num_list = [0]
    for mov_num in np.arange(len(xcm_diff_all)):
        xcm_diff_curr = xcm_diff_all[mov_num]
        ycm_diff_curr = ycm_diff_all[mov_num]
        diff_curr = np.sqrt(xcm_diff_curr**2 + ycm_diff_curr**2)
        
        # exclude bad frames
        exclude_frames_curr = exclude_frames_all[mov_num]
        exclude_frames_curr = np.asarray(exclude_frames_curr,dtype=np.int32)
        #diff_curr = np.ma.array(diff_curr, mask = False)
        #diff_curr_masked = np.ma.masked_where(np.arange(len(diff_curr)) == exclude_frames_curr,diff_curr)
        diff_curr_good = np.delete(diff_curr, exclude_frames_curr)
        diff_curr_good = diff_curr_good[~np.isnan(diff_curr_good)]
        data_to_plot_list.append(diff_curr_good)

    #==============================    
    # make plot    
    fig, ax = plt.subplots(1,1,figsize=(14,7))
    pos = np.arange(len(xcm_diff_all))
    parts = ax.violinplot(data_to_plot_list, pos, showmedians=True,vert=False) ;
    
    for pc in parts['bodies']:
        pc.set_facecolor('#add8e6')
        pc.set_edgecolor('black')
        pc.set_alpha(1) 
    
    parts['cmedians'].set_edgecolor('black')  
    parts['cmedians'].set_linewidth(2)
    parts['cmins'].set_edgecolor('black')
    parts['cmaxes'].set_edgecolor('black')  
    parts['cbars'].set_edgecolor('black')
     
    ax.set_ylim((pos[0]-1,pos[-1]+1))
    ax.set_xlim((0,30))
    
    ax.grid(True)
    
    #ax.set_xlabel('Movie Number')
    ax.set_yticks(pos)
    ax.set_yticklabels(file_ID_all,size='x-small')
    ax.set_xlabel('Annotation to Machine Diff [pix]')
    fig.subplots_adjust(hspace = 0,left=0.17,right=0.95)
    
    if SAVE_PLOTS_FLAG:
        fig.savefig(os.path.join(SAVE_PATH,'error_violin.png'))
#------------------------------------------------------------------------------
if HIST_FLAG:
    
    # collect data together into list
    mov_num_list = np.arange(len(xcm_diff_all))
    data_to_plot_array = np.array([],dtype=np.float32)
    #mov_num_list = [0]
    for mov_num in np.arange(len(xcm_diff_all)):
        xcm_diff_curr = xcm_diff_all[mov_num]
        ycm_diff_curr = ycm_diff_all[mov_num]
        diff_curr = np.sqrt(xcm_diff_curr**2 + ycm_diff_curr**2)
        
        # exclude bad frames
        exclude_frames_curr = exclude_frames_all[mov_num]
        exclude_frames_curr = np.asarray(exclude_frames_curr,dtype=np.int32)
        #diff_curr = np.ma.array(diff_curr, mask = False)
        #diff_curr_masked = np.ma.masked_where(np.arange(len(diff_curr)) == exclude_frames_curr,diff_curr)
        diff_curr_good = np.delete(diff_curr, exclude_frames_curr)
        diff_curr_good = diff_curr_good[~np.isnan(diff_curr_good)]
        data_to_plot_array = np.concatenate((data_to_plot_array, diff_curr_good))

    #==============================    
    # make plot    
    n_bins = 30
    
    fig, ax = plt.subplots()
    
    ax.hist(data_to_plot_array, n_bins, normed=0, histtype='step' ) ;
    ax.grid(True)
    
    ax.set_xlabel('Annotation to Machine Diff [pix]')
    ax.set_ylabel('PDF')
    ax.set_title('Comparing tracking and annotation')
    #plt.xscale('log')
    ax.autoscale(enable=True,axis='x',tight=True)
    if SAVE_PLOTS_FLAG:
        fig.savefig(os.path.join(SAVE_PATH,'error_hist.png'))
        
#------------------------------------------------------------------------------
if CDF_FLAG:
    
    n_bins = 100
    fig, ax = plt.subplots()
    
    # collect data together into list
    mov_num_list = np.arange(len(xcm_diff_all))
    data_to_plot_array = np.array([],dtype=np.float32)
    #mov_num_list = [0]
    for mov_num in np.arange(len(xcm_diff_all)):
        xcm_diff_curr = xcm_diff_all[mov_num]
        ycm_diff_curr = ycm_diff_all[mov_num]
        diff_curr = np.sqrt(xcm_diff_curr**2 + ycm_diff_curr**2)
        
        # exclude bad frames
        exclude_frames_curr = exclude_frames_all[mov_num]
        exclude_frames_curr = np.asarray(exclude_frames_curr,dtype=np.int32)
        #diff_curr = np.ma.array(diff_curr, mask = False)
        #diff_curr_masked = np.ma.masked_where(np.arange(len(diff_curr)) == exclude_frames_curr,diff_curr)
        diff_curr_good = np.delete(diff_curr, exclude_frames_curr)
        diff_curr_good = diff_curr_good[~np.isnan(diff_curr_good)]
        
        # plot cdf for each movie
        counts, bin_edges = np.histogram (diff_curr_good, bins=n_bins, 
                                      normed=True)
        cdf = np.cumsum (counts)
        ax.plot (bin_edges[1:], cdf/cdf[-1],linewidth=0.75)
        data_to_plot_array = np.concatenate((data_to_plot_array, diff_curr_good))

    #==============================    
    # make plot    
    
    
    #fig, ax = plt.subplots()
    
    counts, bin_edges = np.histogram (data_to_plot_array, bins=n_bins, 
                                      normed=True)
    cdf = np.cumsum (counts)
    ax.plot (bin_edges[1:], cdf/cdf[-1],'k-',linewidth=3)
    #ax.hist(data_to_plot_array, n_bins, normed=1, histtype='step',cumulative=True ) ;
    ax.grid(True)
    
    ax.set_xlabel('Annotation to Machine Diff [pix]')
    ax.set_ylabel('CDF')
    ax.set_title('Comparing tracking and annotation')
    plt.xscale('log')
    ax.autoscale(enable=True,axis='both',tight=True)
    if SAVE_PLOTS_FLAG:
        fig.savefig(os.path.join(SAVE_PATH,'error_cdf.png'))