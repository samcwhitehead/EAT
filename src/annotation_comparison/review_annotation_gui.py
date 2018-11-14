# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 09:35:29 2018

@author: Fruit Flies

GUI to review bout annotations and machine detections. decide whether they're 
good or bad.
"""
#------------------------------------------------------------------------------
import sys

from os import listdir
from os.path import isfile, join,  splitext

import numpy as np
from scipy import interpolate, optimize
#from scipy.spatial.distance import cdist, squareform
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor

import csv

from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis
from v_expresso_gui_params import analysisParams

from get_annotation_fileID import get_annotation_files_in_folder

#------------------------------------------------------------------------------

data_dir = 'F:\\Expresso GUI\\Saumya Annotations\\annotations_expresso_data\\'
annot_dir_1 = 'F:\\Expresso GUI\\Saumya Annotations\\annotations_expresso_user\\'
annot_dir_2 = 'F:\\Expresso GUI\\Saumya Annotations\\annotations_expresso_user_2\\'
save_path = 'F:\\Expresso GUI\\Saumya Annotations\\review_annotations\\'

data_filename_list, filekeyname_list, groupkeyname_list =  \
                                    get_annotation_files_in_folder(data_dir)
      
# pick which data file to look at (0-80)                            
ind = 0

#------------------------------------------------------------------------------                                   
SKIP_IND = [11,13,38,41,44,46,49]
DELTA_T = 10 # if bouts are separated in time by less than this value, merge
AGREE_FRAC = 0.70 # fraction of overlap required for bouts to "agree" 
                                    
min_bout_duration = analysisParams['min_bout_duration']
min_bout_volume = analysisParams['min_bout_volume']

#------------------------------------------------------------------------------
# function to read bouts from csv file. another version in run_annotation_check.py
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
# function to combine annotations from two annotators 
def merge_meal_bouts(bout_array, volumes, delta_t):
    N_bouts = bout_array.shape[1]
    if N_bouts < 1:
        bouts_merge = bout_array
        volumes_merge = volumes
            
    else:    
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
    return (bouts_merge, volumes_merge)      
 
#------------------------------------------------------------------------------
# function to determine bout based on ind input
def remove_bouts(click_inds,bouts_array):
    bouts_array_updated = bouts_array.copy()
    
    for click_ind in click_inds:
        bout_starts = bouts_array_updated[0,:]
        bout_ends = bouts_array_updated[1,:]
        bout_ind = np.where((bout_starts < click_ind) & (bout_ends > click_ind))[0]
        
        if (bout_ind.size > 0):
            print('SUCCESSFULLY REMOVED BOUT')
            bouts_array_updated = \
                        bouts_array_updated[:,np.arange(bouts_array_updated.shape[1])!=bout_ind]
        else:
            print('NO BOUT SELECTED')
        
    return bouts_array_updated        
    
#------------------------------------------------------------------------------
# function to plot updated annotations
def plot_updated_bouts(updated_bouts,dset,dset_smooth,t,ax_title):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, 
                                figsize=(12, 5))
        
    ax1.set_ylabel('Liquid [nL]')
    ax2.set_ylabel('Liquid [nL]')
    ax2.set_xlabel('Time [s]')
    ax1.set_title(ax_title + ' UPDATED', fontsize=20)
    ax2.set_title('Smoothed Data')
    
    ax1.plot(t,dset,'k')
    ax2.plot(t, dset_smooth,'k')
    
    # plot results of combined annotations
    for i in np.arange(updated_bouts.shape[1]):
        ax2.plot(t[updated_bouts[0,i]:updated_bouts[1,i]], 
                 dset_smooth[updated_bouts[0,i]:updated_bouts[1,i]],'b.',ms=2)
        ax2.axvspan(t[updated_bouts[0,i]],t[updated_bouts[1,i]-1], 
                         facecolor='blue', edgecolor='none', alpha=0.3)
        ax1.axvspan(t[updated_bouts[0,i]],t[updated_bouts[1,i]-1], 
                         facecolor='blue', edgecolor='none', alpha=0.3)
                         
    ax1.set_xlim([t[0],t[-1]])
    ax1.set_ylim([np.amin(dset),np.amax(dset)])     
    
    ax1.grid(True)
    ax2.grid(True)
    
#------------------------------------------------------------------------------
# begin main script
#------------------------------------------------------------------------------

# check if data file is good
if ind in SKIP_IND:
    print('BAD DATA FILE')
    quit()
    
#======================================
# print some instructions
#======================================
print(50*'-')
print("USER ANNOTATION FOR EXPRESSO DATA FILES")
print(50*'-')
print('- Double LEFT click to indicate FALSE positive')
print('- Press B key to save results and close plot window')
print('- Press N key to exit without saving')
print('')
print('*Make sure to do all button presses/clicks when plot window is selected')
print(50*'-')


data_filename   = data_filename_list[ind]   # NAME OF HDF5 FILE
filekeyname     = filekeyname_list[ind]     # NAME OF BANK 
groupkeyname    = groupkeyname_list[ind]    # NAME OF CHANNEL

#======================================
# get file paths/names sorted
#======================================
if sys.version_info[0] < 3:
    filekeyname = unicode(filekeyname) 
    groupkeyname = unicode(groupkeyname) 

data_filename_no_ext = splitext(data_filename)[0]
data_filename_no_ext_split = data_filename_no_ext.split('_')[0]

annot_1_filename =  str(data_filename_no_ext_split) + '_' +  filekeyname + \
                        '_' + groupkeyname + '.csv'
annot_2_filename =  str(data_filename_no_ext) + '_' +  filekeyname + '_' + \
                        groupkeyname + '.csv'
# annotation files should have same format but different paths
annot_1_filepath = join(annot_dir_1, annot_1_filename)
annot_2_filepath = join(annot_dir_2, annot_2_filename)

save_filepath = join(save_path,splitext(annot_2_filename)[0] + '.csv')
#======================================
# load actual data from hdf5
#======================================
data_file = join(data_dir,data_filename)     
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
    
dset_smooth, bouts_machine, volumes_machine = bout_analysis(dset,frames)

#======================================
# load bouts from csv
#======================================
 # get annotations from first folder
bouts_ann_1, _ = load_csv_bouts(annot_1_filepath, dset_smooth)
    
 # get annotations from second folder
bouts_ann_2, _ = load_csv_bouts(annot_2_filepath, dset_smooth)

# combine bout annotations    
bouts_ann_comb, volumes_comb = combine_bout_lists(bouts_ann_1,
                                                  bouts_ann_2,dset_smooth)
# exclude bouts below min duration/volume
if bouts_ann_comb.size > 2:
    bouts_ann_comb_dur = bouts_ann_comb[1,:]-bouts_ann_comb[0,:]
    ann_comb_good_ind = (bouts_ann_comb_dur > min_bout_duration)
    ann_comb_good_ind = (ann_comb_good_ind & (volumes_comb > min_bout_volume)) 
    
    bouts_ann_comb = bouts_ann_comb[:,ann_comb_good_ind] 

bouts_ann_merge, vol_ann_merge = merge_meal_bouts(bouts_ann_comb,
                                                  volumes_comb,DELTA_T) 
bouts_machine_merge, vol_machine_merge = merge_meal_bouts(bouts_machine,
                                                volumes_machine,DELTA_T) 

#==============================================
# create new list of bouts with combined data 
#==============================================

new_feeding_signal = np.zeros(dset_smooth.shape)

for ith in np.arange(bouts_ann_merge.shape[1]):
    i1 = bouts_ann_merge[0,ith]
    i2 = bouts_ann_merge[1,ith]
    new_feeding_signal[i1:i2] = 1 

for jth in np.arange(bouts_machine_merge.shape[1]):
    j1 = bouts_machine_merge[0,jth]
    j2 = bouts_machine_merge[1,jth]
    new_feeding_signal[j1:j2] = 1 

bout_start_new = np.where(np.diff(new_feeding_signal) > 0 )[0] + 1
bout_end_new = np.where(np.diff(new_feeding_signal) < 0 )[0] + 1 

bouts_new = np.vstack((bout_start_new,bout_end_new))   
remove_ind = [] 

#------------------------------------------------------------------------------        
#======================================
# make plot
#======================================    

fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, 
                                figsize=(12, 5))
        
ax1.set_ylabel('Liquid [nL]')
ax2.set_ylabel('Liquid [nL]')
ax2.set_xlabel('Time [s]')
ax_title = data_filename + ', ' + filekeyname + ', ' + groupkeyname
ax1.set_title(ax_title, fontsize=20)
ax2.set_title('Smoothed Data')

ax1.plot(t,dset,'k')
ax2.plot(t, dset_smooth,'k')

# plot results of combined annotations
for i in np.arange(bouts_ann_comb.shape[1]):
    ax2.plot(t[bouts_ann_comb[0,i]:bouts_ann_comb[1,i]], 
             dset_smooth[bouts_ann_comb[0,i]:bouts_ann_comb[1,i]],'g.',ms=2)
    ax2.axvspan(t[bouts_ann_comb[0,i]],t[bouts_ann_comb[1,i]-1], 
                     facecolor='green', edgecolor='none', alpha=0.3)
    ax1.axvspan(t[bouts_ann_comb[0,i]],t[bouts_ann_comb[1,i]-1], 
                     facecolor='green', edgecolor='none', alpha=0.3)
                     
# plot results of machine
for k in np.arange(bouts_machine.shape[1]):
    ax2.plot(t[bouts_machine[0,k]:bouts_machine[1,k]], 
             dset_smooth[bouts_machine[0,k]:bouts_machine[1,k]],'r.',ms=2)
    ax2.axvspan(t[bouts_machine[0,k]],t[bouts_machine[1,k]-1], 
                     facecolor='red', edgecolor='none', alpha=0.3)
    ax1.axvspan(t[bouts_machine[0,k]],t[bouts_machine[1,k]-1], 
                     facecolor='red', edgecolor='none', alpha=0.3)

userArtist = plt.Line2D((0,1),(0,0), color='g',linewidth=10,alpha=0.3)
machineArtist = plt.Line2D((0,1),(0,0), color='r',linewidth=10,alpha=0.3)
agreeArtist = plt.Line2D((0,1),(0,0), color=(0.6392,0.5137,0.3373),
                         linewidth=10)

legend = ax1.legend([userArtist, machineArtist, agreeArtist], 
                     ['User', 'Machine', 'Agreement'],
                        loc='upper right',shadow=False)                     
ax1.set_xlim([t[0],t[-1]])
ax1.set_ylim([np.amin(dset),np.amax(dset)])     

ax1.grid(True)
ax2.grid(True)

multi = MultiCursor(fig.canvas, (ax1, ax2), color='grey', lw=.5, horizOn=True, 
                    vertOn=True)


#------------------------------------------------------------------------------
# define click events
#------------------------------------------------------------------------------                
def onclick(event):
    if event.dblclick:
        t_pick = event.xdata 
        t_closest_ind = np.searchsorted(t,t_pick,side='right')
        #t_closest = t[t_closest_ind]
        
        # DOUBLE LEFT CLICK TO INDICATE TRUE POSITIVE
        if event.button == 1: 
            remove_ind.append(t_closest_ind)
        
        else:
            print(event.button)    
          
def onpress(event):

    # B KEY TO SAVE RESULTS TO FILE AND EXIT    
    if event.key.lower() == 'b':
        
        bouts_to_save = remove_bouts(remove_ind, bouts_new)
        
        bout_start_ind_array = bouts_to_save[0,:]
        bout_end_ind_array = bouts_to_save[1,:]
        
        bout_start_array = t[bout_start_ind_array]
        bout_end_array =  t[bout_end_ind_array]
        
        bouts_t = np.transpose(np.vstack((bout_start_array,bout_end_array)))
        bouts_ind = np.transpose(np.vstack((bout_start_ind_array,bout_end_ind_array)))
        row_mat = np.hstack((bouts_ind, bouts_t))
        
        if sys.version_info[0] < 3:
            save_file = open(save_filepath, 'wb')
            save_writer = csv.writer(save_file)
        
            save_writer.writerow([data_filename + ', ' + filekeyname + ', ' + groupkeyname])
            save_writer.writerow(['Bout Number'] + ['Bout Start [idx]'] + \
                ['Bout End [idx]'] + ['Bout Start [s]'] + ['Bout End [s]'])
            cc = 1            
            for row in row_mat:
                new_row = np.insert(row,0,cc)
                save_writer.writerow(new_row)
                cc += 1
            save_file.close()        
        else:
            with open(save_filepath, 'w', newline='') as save_file:
                save_writer = csv.writer(save_file)
                    
                save_writer.writerow([data_filename + ', ' + filekeyname + ', ' + groupkeyname])
                save_writer.writerow(['Bout Number'] + ['Bout Start [idx]'] + \
                    ['Bout End [idx]'] + ['Bout Start [s]'] + ['Bout End [s]'])
                cc = 1            
                for row in row_mat:
                    new_row = np.insert(row,0,cc)
                    save_writer.writerow(new_row)
                    cc += 1 
        plt.close()
        
        plot_updated_bouts(bouts_to_save,dset,dset_smooth,t,ax_title)
        return
        #quit()
    elif event.key.lower() == 'n': 
        plt.close()
        return
        
cid = fig.canvas.mpl_connect('button_press_event', onclick)  
pid = fig.canvas.mpl_connect('key_press_event', onpress)                  