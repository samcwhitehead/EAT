# -*- coding: utf-8 -*-
"""
Created on Tue Nov 06 11:36:52 2018

@author: Fruit Flies

Script for comparing user-annotated expresso bout data with machine output

"""
#------------------------------------------------------------------------------
import sys
import os

import numpy as np
from scipy import interpolate, optimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import csv
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis
from v_expresso_gui_params import analysisParams

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# get paths to raw data, annotations, etc.
annotator_name = 'saumya'

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(src_dir)
bout_ann_dir = os.path.join(parent_dir,'dat','bout_annotations')
data_dir = os.path.join(bout_ann_dir,'data')
annot_dir = os.path.join(bout_ann_dir,'annotations_{}'.format(annotator_name))
machine_dir = os.path.join(bout_ann_dir,'machine')
save_path = os.path.join(annot_dir,'plots')

#------------------------------------------------------------------------------
# define params
PLOT_FLAG_1 = True # do you want plots for each file?    
SAVE_FLAG_1 = True # do you to save plots for each file?        
PLOT_FLAG_2 = True  # do you want a summary plot? 
SAVE_FLAG_2 = True # do you to save summary plot?                                 
SKIP_IND = []       # indices of files to skip
DELTA_T = 6        # if bouts are separated in time by less than this->merge
AGREE_FRAC = 0.50   # fraction of overlap required for bouts to "agree" 
                                    
min_bout_duration = analysisParams['min_bout_duration']
min_bout_volume = analysisParams['min_bout_volume']

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# function to retrieve all channels in data directory
def get_channel_data_in_folder(data_path):
    hdf5_filename_list = [] 
    bank_list = []
    channel_list = [] 
    
    #data_dir = join(data_path)
    data_filenames = [f for f in os.listdir(data_path) if 
                        os.path.isfile(os.path.join(data_path, f)) and 
                        f.endswith('.hdf5')]             
    
    for filename in data_filenames:
        data_full_path = os.path.join(data_path,filename)
        
        with h5py.File(data_full_path,'r') as f:
            grp_key_list = [key for key in list(f.keys()) if key.startswith('XP')]
            for grp_key in grp_key_list:
                grp = f[grp_key]                
                channels = grp.keys()
                for ch in channels:
                   hdf5_filename_list.append(filename) 
                   bank_list.append(grp_key)
                   channel_list.append(ch)
    
    return (hdf5_filename_list, bank_list, channel_list)
#------------------------------------------------------------------------------
# function to read bouts from csv file. alt version in run_annotation_check.py
def load_csv_bouts(csv_filename,dset_smooth):
    csv_file_list = [] 
    with open(csv_filename, 'rU') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            csv_file_list.append(row)
    bouts_start = [] 
    bouts_end = []        
    for ith in np.arange(2,len(csv_file_list)): 
        row_curr = csv_file_list[ith]
        try:        
            bouts_start.append(int(row_curr[1].split('.')[0]))
            bouts_end.append(int(row_curr[2].split('.')[0]))
        except ValueError:
            continue 
    volumes = dset_smooth[bouts_start] - dset_smooth[bouts_end]     
    bouts = np.vstack((bouts_start, bouts_end))
    
    return (bouts, volumes)   

#------------------------------------------------------------------------------
# function to combine annotations from two annotators 
def combine_bout_lists(bouts_1,bouts_2,dset_smooth):
    dset_length = dset_smooth.size
    feeding_signal = np.squeeze(np.zeros([dset_length,1]))
    
    for ith in np.arange(bouts_1.shape[1]):
        feeding_signal[bouts_1[0,ith]:bouts_1[1,ith]] = 1
    for jth in np.arange(bouts_2.shape[1]):
        feeding_signal[bouts_2[0,jth]:bouts_2[1,jth]] = 1    
    
    feeding_signal_diff = np.diff(feeding_signal)
    bouts_comb_start = np.where(feeding_signal_diff == 1)[0] + 1
    bouts_comb_end = np.where(feeding_signal_diff == -1)[0] + 1
    
    bouts_comb = np.vstack((bouts_comb_start,bouts_comb_end))
    volumes_comb = dset_smooth[bouts_comb_start] - dset_smooth[bouts_comb_end] 
    
    return (bouts_comb, volumes_comb)

#------------------------------------------------------------------------------
# merge meal bouts that are separated in time by less than delta_t
def merge_meal_bouts_dt(bout_array, volumes, delta_t):
    N_bouts = bout_array.shape[1]    
    #merge_ind = np.squeeze(np.zeros([N_bouts-1,1]))
    
    bout_start_merge = [bout_array[0,0]]
    bout_end_merge = [] 
    volumes_merge = [volumes[0]] 
    vol_cc = 0 
    
    for bn in np.arange(N_bouts-1):
        bout_sep = bout_array[0,bn+1] - bout_array[1,bn]  
        if bout_sep > delta_t:
            bout_end_merge.append(bout_array[1,bn])
            bout_start_merge.append(bout_array[0,bn+1])
            
            volumes_merge.append(volumes[bn+1])
            vol_cc += 1 
        else:
            volumes_merge[vol_cc] = volumes_merge[vol_cc] + volumes[bn+1]
    
    bout_end_merge.append(bout_array[1,-1])
    
    bouts_merge = np.vstack((bout_start_merge,bout_end_merge))  
    volumes_merge = np.asarray(volumes_merge)
    return (bouts_merge, volumes_merge)      


#------------------------------------------------------------------------------
# merge meal bouts based on whether or not multiple detected bouts from one 
#   source (e.g. user annotation) contains multiple machine bouts (in this case
#   merge machine bouts)
def merge_meal_bouts_overlap(bouts_1, volumes_1, bouts_2, volumes_2, N_pts):
    
    bouts_1_merge = bouts_1.copy()
    bouts_2_merge = bouts_2.copy()
    volumes_1_merge = volumes_1.copy()
    volumes_2_merge = volumes_2.copy()
    
    # make matrices where each row corresponds to the time series with 
    # a "1" if bout is occuring at the time point, and "0" otherwise
    sig_mat_1 = np.zeros([bouts_1.shape[1],N_pts])
    for pth in np.arange(bouts_1.shape[1]):
        sig_mat_1[pth,bouts_1[0,pth]:bouts_1[1,pth]]=1 
    
    sig_mat_2 = np.zeros([bouts_2.shape[1],N_pts])
    for qth in np.arange(bouts_2.shape[1]):
        sig_mat_2[qth,bouts_2[0,qth]:bouts_2[1,qth]]=1
    
    # in this matrix, the [i, j] element is the overlap of ith meal from bout 
    # set 1 with the jth meal from bout set 2
    mat_prod = np.matmul(sig_mat_1,np.transpose(sig_mat_2))
    
    """
    # all-zero columns indicate that there is a machine meal with no 
    #  matching annotated meal. vice-versa for all-zero rows  
    row_sum = np.sum(mat_prod,axis=0)
    machine_only_ind = np.squeeze(np.where(row_sum == 0)[0])
    if machine_only_ind.size == 1:
        machine_only_vol.append(vol_machine_merge[machine_only_ind])
    elif machine_only_ind.size > 1:
        machine_only_vol.extend(vol_machine_merge[machine_only_ind])
    
    col_sum = np.sum(mat_prod,axis=1)
    ann_only_ind = np.where(col_sum == 0)[0]
    if ann_only_ind.size == 1:
        user_only_vol.append(vol_ann_merge[ann_only_ind])
    elif ann_only_ind.size > 1:
        user_only_vol.extend(vol_ann_merge[ann_only_ind])
    """
    
    # now, for each row with at least one non-zero entry, this means
    # that a user-detected meal (corresponding to row number) has 
    # non-zero overlap with a machine-detected meal. we want to take
    # all the machine-detected meals and merge them to get a better 
    # estimate of overlap:
    mat_prod_logical = (mat_prod > 0)
    
    # these are cases in which the annotator marks a bout that contains
    #  multiple machine bouts
    col_sum_logical = np.sum(mat_prod_logical,axis=1)
    multi_bout_ind_1 = np.where(col_sum_logical > 1)[0] 
    cols_to_delete = np.array([],dtype=int)
    for row_idx in multi_bout_ind_1:
        to_merge_idx = np.where(mat_prod[row_idx,:] > 0)[0]
        bouts_2_merge[1,to_merge_idx[0]] = bouts_2_merge[1,to_merge_idx[-1]]
        volumes_2_merge[to_merge_idx[0]] = np.sum(volumes_2_merge[to_merge_idx])
        cols_to_delete = np.append(cols_to_delete,to_merge_idx[1:])
    
    bouts_2_merge = np.delete(bouts_2_merge,cols_to_delete,axis=1)
    volumes_2_merge = np.delete(volumes_2_merge,cols_to_delete)
    
    # now do opposite case
    row_sum_logical = np.sum(mat_prod_logical,axis=0)
    multi_bout_ind_2 = np.where(row_sum_logical > 1)[0] 
    rows_to_delete = np.array([],dtype=int)
    for col_idx in multi_bout_ind_2:
        to_merge_idx = np.where(mat_prod[:,col_idx] > 0)[0]
        bouts_1_merge[1,to_merge_idx[0]] = bouts_1_merge[1,to_merge_idx[-1]]
        volumes_1_merge[to_merge_idx[0]] = np.sum(volumes_1_merge[to_merge_idx])
        rows_to_delete = np.append(rows_to_delete,to_merge_idx[1:])
        
    bouts_1_merge = np.delete(bouts_1_merge,rows_to_delete,axis=1)
    volumes_1_merge = np.delete(volumes_1_merge,rows_to_delete)
    
    return (bouts_1_merge, volumes_1_merge, bouts_2_merge, volumes_2_merge)
    
#------------------------------------------------------------------------------
# function to match bouts (linear assignment)
def match_bouts(bouts_1, volumes_1, bouts_2, volumes_2,N_pts):
    # initialize arrays    
    agree_vol = np.array([])
    agree_overlap_frac = np.array([])
    only_1_vol = np.array([])
    only_2_vol = np.array([])
    
    #take time midpoint of bouts to try to match bouts with each other
    bouts_1_mid = np.mean(bouts_1,axis=0)
    bouts_2_mid = np.mean(bouts_2,axis=0)
    
    bouts_1_mid = np.expand_dims(bouts_1_mid,axis=1)
    bouts_2_mid = np.expand_dims(bouts_2_mid,axis=1)
    
    cost_mat = cdist(bouts_1_mid, bouts_2_mid)
    #cost_mat = cdist(bouts_machine_mid, bouts_ann_mid)
    
    # use hungarian method to match bouts based on midpoint distance
    row_ind, col_ind = optimize.linear_sum_assignment(cost_mat)
    
    # get feeding signal (0s and 1s) from bouts
    feeding_signal_1 = np.squeeze(np.zeros([N_pts,1],
                                             dtype=bool))
    for pth in np.arange(bouts_1.shape[1]):
        feeding_signal_1[bouts_1[0,pth]:bouts_1[1,pth]]=1 
    
    feeding_signal_2 = np.squeeze(np.zeros([N_pts,1], dtype=bool))
    for qth in np.arange(bouts_2.shape[1]):
        feeding_signal_2[bouts_2[0,qth]:bouts_2[1,qth]]=1
    
    # record overlapping events
    for ith in np.arange(len(row_ind)):
        mean_vol = np.mean([volumes_1[row_ind[ith]], 
                            volumes_2[col_ind[ith]]])
        i1 = np.min([bouts_1[0,row_ind[ith]], 
                     bouts_2[0,col_ind[ith]]])
        i2 = np.max([bouts_1[1,row_ind[ith]], 
                     bouts_2[1,col_ind[ith]]])     
        
        overlap = np.sum((feeding_signal_1[i1:i2] & \
                           feeding_signal_2[i1:i2] ))                      
        bout_dur_1 = np.sum(feeding_signal_1[i1:i2])
        bout_dur_2 = np.sum(feeding_signal_2[i1:i2])
        overlap_frac = overlap / np.mean([bout_dur_1, bout_dur_2]) 
        
        agree_vol = np.append(agree_vol,mean_vol)
        agree_overlap_frac = np.append(agree_overlap_frac,overlap_frac)    
    
    # bouts only identified by user
    for jth in np.arange(len(bouts_1_mid)):
        if jth not in row_ind:
            only_1_vol = np.append(only_1_vol,volumes_1[jth])
    
    # bouts only identified by machine
    for kth in np.arange(len(bouts_2_mid)):
        if kth not in col_ind:
            only_2_vol = np.append(only_2_vol, volumes_2[kth])
            
    return (agree_vol,agree_overlap_frac,only_1_vol, only_2_vol)

#------------------------------------------------------------------------------
# function to run analysis on single bout
def check_bout_agreement(data_filename,bank_curr,channel_curr, data_dir, 
                         annot_dir,analysis_params=analysisParams,
                         PLOT_FLAG=False,SAVE_FLAG=False):
    
    #======================================
    # initialize lists for storage
    #======================================
    agree_vol = np.array([])
    agree_overlap_frac = np.array([])
    machine_only_vol = np.array([])
    user_only_vol = np.array([])
                       
    #======================================
    # get file paths/names sorted
    #======================================
    if sys.version_info[0] < 3:
        bank_curr = unicode(bank_curr) 
        channel_curr = unicode(channel_curr) 
    
    data_filename_no_ext = data_filename.split('.')[0]
    
    annot_filename =  str(data_filename_no_ext) + '_' +  bank_curr + \
                            '_' + channel_curr + '_ANNOTATION.csv'
    
    # check to see if annotation exists for this file
    annot_filepath = os.path.join(annot_dir, annot_filename)
    if not os.path.exists(annot_filepath):
        #print('No annotation file for {}, {}, {}'.format(data_filename_no_ext, 
        #      bank_curr,channel_curr))
        return (agree_vol,agree_overlap_frac,machine_only_vol,user_only_vol)     
    
    
    #======================================
    # load raw data from hdf5
    #======================================
    data_file = os.path.join(data_dir,data_filename)     
    dset, t = load_hdf5(data_file,bank_curr,channel_curr)
        
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
        
    dset_smooth, bouts_machine, volumes_machine = bout_analysis(dset,frames,
                                                   analysis_params=analysis_params)
    N_PTS = dset_smooth.size # total number of time points
    
    #======================================
    # load bouts from csv
    #======================================
     # get annotations from first folder
    bouts_ann, volumes_ann = load_csv_bouts(annot_filepath, dset_smooth)

    # exclude bouts below min duration/volume
    if bouts_ann.size > 2:
        bouts_ann_dur = bouts_ann[1,:]-bouts_ann[0,:]
        ann_good_ind = (bouts_ann_dur > min_bout_duration)
        ann_good_ind = (ann_good_ind & (volumes_ann > min_bout_volume)) 
        
        bouts_ann = bouts_ann[:,ann_good_ind] 
        volumes_ann = volumes_ann[ann_good_ind]
#------------------------------------------------------------------------------        
    #======================================
    # make plots
    #======================================    
    if PLOT_FLAG:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, 
                                        figsize=(17, 7))
                
        ax1.set_ylabel('Liquid [nL]')
        ax2.set_ylabel('Liquid [nL]')
        ax2.set_xlabel('Time [s]')
        ax1.set_title(data_filename + ', ' + bank_curr + ', ' + channel_curr,
                  fontsize=20)
        ax2.set_title('Smoothed Data')
        
        ax1.plot(t,dset,'k')
        ax2.plot(t, dset_smooth,'k')
        
        # plot results of annotation
        for i in np.arange(bouts_ann.shape[1]):
            ax2.plot(t[bouts_ann[0,i]:bouts_ann[1,i]], 
                     dset_smooth[bouts_ann[0,i]:bouts_ann[1,i]],'g.',
                        ms=2,alpha=0.5)
            ax2.axvspan(t[bouts_ann[0,i]],t[bouts_ann[1,i]-1], 
                             facecolor='green', edgecolor='none', alpha=0.5)
            ax1.axvspan(t[bouts_ann[0,i]],t[bouts_ann[1,i]-1], 
                             facecolor='green', edgecolor='none', alpha=0.5)
                             
        # plot results of machine
        for k in np.arange(bouts_machine.shape[1]):
            ax2.plot(t[bouts_machine[0,k]:bouts_machine[1,k]], 
                     dset_smooth[bouts_machine[0,k]:bouts_machine[1,k]],'r.',
                        ms=2,alpha=0.5)
            ax2.axvspan(t[bouts_machine[0,k]],t[bouts_machine[1,k]-1], 
                             facecolor='red', edgecolor='none', alpha=0.5)
            ax1.axvspan(t[bouts_machine[0,k]],t[bouts_machine[1,k]-1], 
                             facecolor='red', edgecolor='none', alpha=0.5)
        
        userArtist = plt.Line2D((0,1),(0,0), color='g',linewidth=10,alpha=0.5)
        machineArtist = plt.Line2D((0,1),(0,0), color='r',linewidth=10,alpha=0.5)
        agreeArtist = plt.Line2D((0,1),(0,0), color=(0.6392,0.5137,0.3373),
                                 linewidth=10)

        ax1.legend([userArtist, machineArtist, agreeArtist], 
                             ['User', 'Machine', 'Agreement'],
                                loc='upper right',shadow=False)                     
        ax1.set_xlim([t[0],t[-1]])
        ax1.set_ylim([np.amin(dset),np.amax(dset)])     
        
        ax1.grid(True)
        ax2.grid(True)
        
        #======================================
        # save plots?
        #======================================   
        if SAVE_FLAG_1:
            plot_fn_str = str(data_filename_no_ext) + '_' +  bank_curr + \
                            '_' + channel_curr + '.png'
            plot_save_path = os.path.join(save_path, plot_fn_str)
            plt.savefig(plot_save_path)
            plt.close(fig)

#------------------------------------------------------------------------------        
    #======================================
    # compare agreement
    #======================================
    
    # if only user detects meals
    if len(volumes_ann) > 0 and len(volumes_machine) == 0:
        for vol in volumes_ann:
            user_only_vol = np.append( user_only_vol, vol)
    
    # if only machine detects meals
    elif len(volumes_ann) == 0 and len(volumes_machine) > 0:
        for vol in volumes_machine:
            machine_only_vol = np.append(machine_only_vol, vol)
    
    elif len(volumes_ann) > 0 and len(volumes_machine) > 0:
        
        #======================================
        # merge meal bouts for comparison
        #====================================== 
        # merge both annotated and machine bouts based on temporal separation
        bouts_ann_dt_merge, vol_ann_dt_merge = merge_meal_bouts_dt(bouts_ann,
                                                          volumes_ann,
                                                          DELTA_T) 
        bouts_machine_dt_merge, vol_machine_dt_merge = merge_meal_bouts_dt(
                                                        bouts_machine,
                                                        volumes_machine,
                                                        DELTA_T) 
                                                        
        # merge bouts based on machine/user overlap
        bouts_ann_merge, vol_ann_merge, bouts_machine_merge, \
        vol_machine_merge = merge_meal_bouts_overlap(bouts_ann_dt_merge, 
                                                     vol_ann_dt_merge, 
                                                     bouts_machine_dt_merge, 
                                                     vol_machine_dt_merge,
                                                     N_PTS)
        
        # match meals with linear assignment, find overlap                                 
        agree_vol_curr, overlap_frac_curr, user_only_curr, mach_only_curr = \
            match_bouts(bouts_ann_merge, vol_ann_merge, bouts_machine_merge, 
                        vol_machine_merge, N_PTS)
        
        # append values to arrays             
        agree_vol = np.append(agree_vol,agree_vol_curr)
        agree_overlap_frac = np.append(agree_overlap_frac,overlap_frac_curr)
        user_only_vol = np.append(user_only_vol,user_only_curr)
        machine_only_vol = np.append(machine_only_vol, mach_only_curr)
        
        
    return (agree_vol,agree_overlap_frac,machine_only_vol,user_only_vol)                            
#------------------------------------------------------------------------------
""" Run main function """
if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    # initialize lists
    agree_vol_all = np.array([]) 
    agree_overlap_frac_all = np.array([])
    machine_only_vol_all = np.array([])
    user_only_vol_all = np.array([])
    
    #--------------------------------------------------------------------------
    # get list of data file, bank, and channel names is data directory
    data_fn_list, bank_list, channel_list = get_channel_data_in_folder(data_dir)
    
    #--------------------------------------------------------------------------
    # loop through data and compare annotations with machine output
    
    for ind in np.arange(len(data_fn_list)): 
        
        if ind in SKIP_IND:
            continue
        
        data_filename   = data_fn_list[ind]   # NAME OF HDF5 FILE
        bank_curr     = bank_list[ind]     # NAME OF BANK 
        channel_curr    = channel_list[ind]    # NAME OF CHANNEL
        
        agree_vol, agree_overlap_frac, machine_only_vol, user_only_vol = \
                    check_bout_agreement(data_filename,bank_curr,channel_curr,
                                         data_dir,annot_dir,PLOT_FLAG=PLOT_FLAG_1) 
        

        agree_vol_all = np.append(agree_vol_all,agree_vol)
        agree_overlap_frac_all = np.append(agree_overlap_frac_all,
                                           agree_overlap_frac)
        machine_only_vol_all = np.append(machine_only_vol_all,machine_only_vol)
        user_only_vol_all = np.append(user_only_vol_all,user_only_vol)
        
        #print(ind)
    #------------------------------------------------------------------------------
    # make agreement histogram 
    if PLOT_FLAG_2:
        user_only_vol_all = np.asarray(user_only_vol_all)
        machine_only_vol_all = np.asarray(machine_only_vol_all) 
        agree_vol_all = np.asarray(agree_vol_all)               
        agree_overlap_frac_all = np.asarray(agree_overlap_frac_all)
        
        agree_vol_good = agree_vol_all[(agree_overlap_frac_all >= AGREE_FRAC)]
        agree_vol_bad = agree_vol_all[(agree_overlap_frac_all < AGREE_FRAC)] 
                          
        colors = ['blue','grey','red','green']  
        bins = np.arange(0,140,10)              
        
        fig, ax = plt.subplots()
        ax.hist([agree_vol_good,agree_vol_bad,machine_only_vol_all,user_only_vol_all],
                bins, histtype='bar',stacked=True,color=colors)
        
        ax.set_ylabel('Count',fontsize=16)
        ax.set_xlabel('Bout Volume [nL]',fontsize=16)
        ax.legend(['Agree, Overlap > {}'.format(AGREE_FRAC), \
                'Agree, Overlap < {}'.format(AGREE_FRAC), 'Machine', 'User'],
                  loc='upper right',shadow=False)        
        
        total_bout_num = len(agree_vol_all) + len(user_only_vol_all) + \
                                                    len(machine_only_vol_all)
        ax.set_title('N = ' + str(total_bout_num) + \
                                ' bouts, Overlap Thresh = ' + str(AGREE_FRAC),
                                 fontsize=18)
        if SAVE_FLAG_2:
            hist_fn_str = 'annotation_summary_{}.png'.format(annotator_name)
            hist_save_path = os.path.join(save_path, hist_fn_str)
            plt.savefig(hist_save_path)
            #plt.close(fig)

