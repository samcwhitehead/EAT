# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:09:09 2017

@author: Fruit Flies
"""
#------------------------------------------------------------------------------
import sys
import os


import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis
from v_expresso_gui_params import analysisParams

from bout_annotation_comparison import (get_channel_data_in_folder, 
                                        load_csv_bouts, merge_meal_bouts_dt,
                                        merge_meal_bouts_overlap, match_bouts)
#------------------------------------------------------------------------------

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(src_dir)
data_dir = os.path.join(parent_dir,'dat','bout_annotations','data')
annot_dir_1 = os.path.join(parent_dir,'dat','bout_annotations','annotations_saumya')
annot_dir_2 = os.path.join(parent_dir,'dat','bout_annotations','annotations_esther')
save_path = os.path.join(parent_dir,'dat','bout_annotations','compare_annotators')

#------------------------------------------------------------------------------
# params

PLOT_FLAG_1 = True # plots for individual data channels
PLOT_FLAG_2 = True # summary plots

SAVE_FLAG_1 = True # save individual channel plots
SAVE_FLAG_2 = True # save summary plot

DELTA_T = 6        # if bouts are separated in time by less than this->merge
AGREE_FRAC = 0.70   # fraction of overlap required for bouts to "agree" 
min_bout_duration = analysisParams['min_bout_duration']
min_bout_volume = analysisParams['min_bout_volume']

#------------------------------------------------------------------------------
# fetch available data
data_filename_list, filekeyname_list, groupkeyname_list =  \
                                    get_channel_data_in_folder(data_dir)
                                    
agree_vol = np.array([])
agree_overlap_frac = np.array([])
only_ann_1_vol = np.array([])
only_ann_2_vol = np.array([])

#------------------------------------------------------------------------------
# loop through data and plot results for each file

for ind in np.arange(len(data_filename_list)):
#for ind in np.arange(3):
    data_filename   = data_filename_list[ind]   # NAME OF HDF5 FILE
    filekeyname     = filekeyname_list[ind]     # NAME OF BANK 
    groupkeyname    = groupkeyname_list[ind]    # NAME OF CHANNEL
    
    #======================================
    # get file paths/names sorted
    #======================================
    if sys.version_info[0] < 3:
        filekeyname = unicode(filekeyname) 
        groupkeyname = unicode(groupkeyname) 
    
    data_filename_no_ext = data_filename.split('.')[0]
    
    annot_1_filename =  str(data_filename_no_ext) + '_' +  filekeyname + \
                            '_' + groupkeyname + '_ANNOTATION.csv'
    annot_2_filename =  str(data_filename_no_ext) + '_' +  filekeyname + \
                            '_' + groupkeyname + '_ANNOTATION.csv'
    
    # annotation files should have same format but different paths
    annot_1_filepath = os.path.join(annot_dir_1, annot_1_filename)
    annot_2_filepath = os.path.join(annot_dir_2, annot_2_filename)
    
    if not (os.path.exists(annot_1_filepath) and os.path.exists(annot_2_filepath)):
        continue
    #======================================
    # load actual data from hdf5
    #======================================
    data_file = os.path.join(data_dir,data_filename)     
    dset, t = load_hdf5(data_file,filekeyname,groupkeyname)
        
    dset_check = (dset != -1)
    if (np.sum(dset_check) == 0):
        messagestr = "Bad dataset: " + data_file
        print(messagestr)
    
    dset_size = dset.size     
    frames = np.arange(0,dset_size)
    
    dset = dset[dset_check]
    frames = frames[np.squeeze(dset_check)]
    t = t[dset_check]
    
    new_frames = np.arange(0,np.max(frames)+1)
    sp_raw = interpolate.InterpolatedUnivariateSpline(frames, dset)
    sp_t = interpolate.InterpolatedUnivariateSpline(frames, t)
    dset = sp_raw(new_frames)
    t = sp_t(new_frames)
    frames = new_frames
        
    dset_smooth, bouts_data, _ = bout_analysis(dset,frames)
    N_PTS = dset_smooth.size
    #======================================
    # load bouts from csv
    #======================================
     # get annotations from first folder
    bouts_ann_1, volumes_ann_1 = load_csv_bouts(annot_1_filepath, dset_smooth)
    # exclude bouts below min volume
    if bouts_ann_1.size > 2:
        bouts_ann_1_dur = bouts_ann_1[1,:]-bouts_ann_1[0,:]
        ann_1_good_ind = (bouts_ann_1_dur > min_bout_duration)
        ann_1_good_ind = (ann_1_good_ind & (volumes_ann_1 > min_bout_volume)) 
        
        bouts_ann_1 = bouts_ann_1[:,ann_1_good_ind]
        volumes_ann_1 = volumes_ann_1[ann_1_good_ind]
        
     # get annotations from second folder
    bouts_ann_2, volumes_ann_2 = load_csv_bouts(annot_2_filepath, dset_smooth)
    # exclude bouts below min volume
    if bouts_ann_2.size > 2:
        bouts_ann_2_dur = bouts_ann_2[1,:]-bouts_ann_2[0,:]
        ann_2_good_ind = (bouts_ann_2_dur > min_bout_duration)
        ann_2_good_ind = (ann_2_good_ind & (volumes_ann_2 > min_bout_volume)) 
        
        bouts_ann_2 = bouts_ann_2[:,ann_2_good_ind]
        volumes_ann_2 = volumes_ann_2[ann_2_good_ind]

#------------------------------------------------------------------------------        
    #======================================
    # make plots
    #======================================   
    if PLOT_FLAG_1:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, 
                                        figsize=(17, 7))
                
        ax1.set_ylabel('Liquid [nL]')
        ax2.set_ylabel('Liquid [nL]')
        ax2.set_xlabel('Time [s]')
        ax1.set_title(data_filename + ', ' + filekeyname + ', ' + groupkeyname,
                  fontsize=20)
        ax2.set_title('Smoothed Data')
        
        ax1.plot(t,dset,'k')
        ax2.plot(t, dset_smooth,'k')
        
        # plot results of first annotation
        for i in np.arange(bouts_ann_1.shape[1]):
            ax2.plot(t[bouts_ann_1[0,i]:bouts_ann_1[1,i]], 
                     dset_smooth[bouts_ann_1[0,i]:bouts_ann_1[1,i]],'.',
                        ms=2,color='yellow',alpha=0.5)
            ax2.axvspan(t[bouts_ann_1[0,i]],t[bouts_ann_1[1,i]-1], 
                             facecolor='yellow', edgecolor='none', alpha=0.5)
            ax1.axvspan(t[bouts_ann_1[0,i]],t[bouts_ann_1[1,i]-1], 
                             facecolor='yellow', edgecolor='none', alpha=0.5)
        
        # plot results of second annotation
        for j in np.arange(bouts_ann_2.shape[1]):
            ax2.plot(t[bouts_ann_2[0,j]:bouts_ann_2[1,j]], 
                     dset_smooth[bouts_ann_2[0,j]:bouts_ann_2[1,j]],'.',ms=2,
                        color='red',alpha=0.5)
            ax2.axvspan(t[bouts_ann_2[0,j]],t[bouts_ann_2[1,j]-1], 
                             facecolor='red', edgecolor='none', alpha=0.5)
            ax1.axvspan(t[bouts_ann_2[0,j]],t[bouts_ann_2[1,j]-1], 
                             facecolor='red', edgecolor='none', alpha=0.5)
        ax1.set_xlim([t[0],t[-1]])
        ax1.set_ylim([np.amin(dset),np.amax(dset)])     
        
        ax1.grid(True)
        ax2.grid(True)
        #======================================
        # save plots
        #======================================
        if SAVE_FLAG_1:
            fn_str = str(data_filename_no_ext) + '_' +  filekeyname + \
                            '_' + groupkeyname + '.png'
            plot_save_path = os.path.join(save_path,fn_str)
            plt.savefig(plot_save_path)
            plt.close(fig)
#------------------------------------------------------------------------------
        
    #======================================
    # compare agreement
    #======================================
    
    # if only user detects meals
    if len(volumes_ann_1) > 0 and len(volumes_ann_2) == 0:
        for vol in volumes_ann_1:
            only_ann_1_vol = np.append(only_ann_1_vol, vol)
    
    # if only machine detects meals
    elif len(volumes_ann_1) == 0 and len(volumes_ann_2) > 0:
        for vol in volumes_ann_2:
            only_ann_2_vol = np.append(only_ann_2_vol, vol)
    
    elif len(volumes_ann_1) > 0 and len(volumes_ann_2) > 0:
        
        #======================================
        # merge meal bouts for comparison
        #====================================== 
        # merge both annotated and machine bouts based on temporal separation
        bouts_ann_1_dt_merge, vol_ann_1_dt_merge = merge_meal_bouts_dt(bouts_ann_1,
                                                          volumes_ann_1,
                                                          DELTA_T) 
        bouts_ann_2_dt_merge, vol_ann_2_dt_merge = merge_meal_bouts_dt(
                                                        bouts_ann_2,
                                                        volumes_ann_2,
                                                        DELTA_T) 
                                                        
        # merge bouts based on machine/user overlap
        bouts_ann_1_merge, vol_ann_1_merge, bouts_ann_2_merge, \
        vol_ann_2_merge = merge_meal_bouts_overlap(bouts_ann_1_dt_merge, 
                                                     vol_ann_1_dt_merge, 
                                                     bouts_ann_2_dt_merge, 
                                                     vol_ann_2_dt_merge,
                                                     N_PTS)
        
        # match meals with linear assignment, find overlap                                 
        agree_vol_curr, overlap_frac_curr, only_1_curr, only_2_curr = \
            match_bouts(bouts_ann_1_merge, vol_ann_1_merge, bouts_ann_2_merge, 
                        vol_ann_2_merge, N_PTS)
        
        # append values to arrays             
        agree_vol = np.append(agree_vol,agree_vol_curr)
        agree_overlap_frac = np.append(agree_overlap_frac,overlap_frac_curr)
        only_ann_1_vol = np.append(only_ann_1_vol , only_1_curr)
        only_ann_2_vol = np.append(only_ann_2_vol,  only_2_curr)

#------------------------------------------------------------------------------
#======================================
# summary plot
#======================================
if PLOT_FLAG_2:
    agree_vol_good = agree_vol[(agree_overlap_frac >= AGREE_FRAC)]
    agree_vol_bad = agree_vol[(agree_overlap_frac < AGREE_FRAC)] 
                      
    colors = ['blue','grey','yellow','red']  
    bins = np.arange(0,140,10)              
    
    fig, ax = plt.subplots()
    ax.hist([agree_vol_good,agree_vol_bad,only_ann_1_vol,only_ann_2_vol],
            bins, histtype='bar',stacked=True,color=colors)
    
    ax.set_ylabel('Count',fontsize=16)
    ax.set_xlabel('Bout Volume [nL]',fontsize=16)
    ax.legend(['Agree, Overlap > {}'.format(AGREE_FRAC), \
            'Agree, Overlap < {}'.format(AGREE_FRAC), 'Ann. #1 Only', \
            'Ann. #2 Only'], loc='upper right',shadow=False)        
    
    total_bout_num = len(agree_vol) + len(only_ann_1_vol) + \
                                                len(only_ann_2_vol)
    ax.set_title('N = ' + str(total_bout_num) + \
                            ' bouts, Overlap Thresh = ' + str(AGREE_FRAC),
                             fontsize=18)
    if SAVE_FLAG_2:
        fn_str = 'compare_annotators.png'
        plot_save_path = os.path.join(save_path,fn_str)
        plt.savefig(plot_save_path)
        #plt.close(fig)
