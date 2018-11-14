# -*- coding: utf-8 -*-
"""
Created on Thu Nov 08 11:27:32 2018

@author: Fruit Flies

Script to search through various bout detection parameters, comparing to user
annotations
"""
import sys
import os


import numpy as np
#from scipy import interpolate
import matplotlib.pyplot as plt
import time
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from load_hdf5_data import load_hdf5
#from bout_analysis_func import bout_analysis
from v_expresso_gui_params import analysisParams

from bout_annotation_comparison import (get_channel_data_in_folder, 
                                        check_bout_agreement)
#------------------------------------------------------------------------------
# directories for data, annotations, etc
annotator_name = 'saumya'

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(src_dir)
bout_ann_dir = os.path.join(parent_dir,'dat','bout_annotations')
data_dir = os.path.join(bout_ann_dir,'data')
annot_dir = os.path.join(bout_ann_dir,'annotations_{}'.format(annotator_name))
machine_dir = os.path.join(bout_ann_dir,'machine')
save_path = 'F:/Expresso GUI/detection_param_search/'

#------------------------------------------------------------------------------
# params for bout comparison

PLOT_FLAG = False           # pmake summary hist?
SAVE_PLOT_FLAG = False      # save summary plots?

SAVE_DATA_FLAG = True       # save summary plot
PLOT_COST_HIST_FLAG = True  # visualize cost fnction distribution

DELTA_T = 6        # if bouts are separated in time by less than this->merge
AGREE_FRAC = 0.70   # fraction of overlap required for bouts to "agree" 
SKIP_IND = []       # skip any annotations?

#------------------------------------------------------------------------------
# params for bout detection (define grid)
#min_bout_duration_grid = np.arange(11)
#min_bout_volume_grid = np.arange(11)
#wlevel_grid = np.arange(1,6)
#wtype_grid = ['sym4','sym5','db3','db4']
#medfilt_window_grid = np.arange(1,25,2)
#mad_thresh_grid = np.arange(1.0, 5.2, 0.2)
min_bout_duration_grid = np.arange(2,11,4)
min_bout_volume_grid = np.arange(2,11,4)
wlevel_grid = np.arange(2,5)
wtype_grid = ['db3','db4']
medfilt_window_grid = np.arange(3,9,2)
mad_thresh_grid = np.arange(2.0, 4.5,0.25)

param_grid_list = [min_bout_duration_grid, min_bout_volume_grid, wlevel_grid,
                   wtype_grid, mad_thresh_grid ]

# make grid of indices for these lists (using letters for brevity)
A, B, C, D, E, F = np.meshgrid(np.arange(len(min_bout_duration_grid)), 
                            np.arange(len(min_bout_volume_grid)), 
                            np.arange(len(wlevel_grid)), 
                            np.arange(len(wtype_grid)), 
                            np.arange(len(medfilt_window_grid)),
                            np.arange(len(mad_thresh_grid)),indexing='ij')

# intialize grid for storing cost 
cost_grid = np.full(A.shape, np.nan)


#--------------------------------------------------------------------------
# get list of data file, bank, and channel names is data directory
data_fn_list, bank_list, channel_list = get_channel_data_in_folder(data_dir)

#------------------------------------------------------------------------------
# function to define overall cost of bout detection comparison
#  NB: FP_ and FN_weight refer to false positive/negative weighting 
#   (i.e. these should be penalized more than jsut missing overlap)
def bout_comp_cost(agree_vol,agree_overlap_frac,machine_only_vol, 
                   user_only_vol, FP_weight=2.0, FN_weight=2.0):
    
    overlap_cost = np.sum(agree_vol*(1-agree_overlap_frac))
    FP_cost = np.sum(FP_weight*machine_only_vol)
    FN_cost = np.sum(FN_weight*user_only_vol)
    
    total_cost = overlap_cost + FP_cost + FN_cost
    
    return total_cost
#------------------------------------------------------------------------------
# loop through different param values to test for 
cc = 0 

for ai, bi, ci, di, ei, fi in np.nditer([A,B,C,D,E,F]):
    
    analysis_params = analysisParams.copy()
    
    # reassign values
    analysis_params['min_bout_duration'] = min_bout_duration_grid[ai]
    analysis_params['min_bout_volume']   = min_bout_volume_grid[bi]
    analysis_params['wlevel']            = wlevel_grid[ci]
    analysis_params['wtype']             = wtype_grid[di]
    analysis_params['medfilt_window']    = medfilt_window_grid[ei]
    analysis_params['mad_thresh']        = mad_thresh_grid[fi]
    
    
    # intialize arrays for bout overlap comparison 
    agree_vol_all = np.array([])
    agree_overlap_frac_all = np.array([])
    machine_only_vol_all = np.array([])
    user_only_vol_all = np.array([])
    
    #--------------------------------------------------------------------------
    # loop through data and compare annotations with machine output
    
    for ind in np.arange(len(data_fn_list)): 
        
        if ind in SKIP_IND:
            continue
        
        data_filename   = data_fn_list[ind]   # NAME OF HDF5 FILE
        bank_curr     = bank_list[ind]     # NAME OF BANK 
        channel_curr    = channel_list[ind]    # NAME OF CHANNEL
        
        try:
            agree_vol, agree_overlap_frac, machine_only_vol, user_only_vol = \
                    check_bout_agreement(data_filename,bank_curr,channel_curr,
                                         data_dir,annot_dir,
                                         analysis_params=analysis_params,
                                         PLOT_FLAG=False) 
        except: 
            print('failed to analyze')
            continue 

        agree_vol_all = np.append(agree_vol_all,agree_vol)
        agree_overlap_frac_all = np.append(agree_overlap_frac_all,
                                           agree_overlap_frac)
        machine_only_vol_all = np.append(machine_only_vol_all,machine_only_vol)
        user_only_vol_all = np.append(user_only_vol_all,user_only_vol)
        
    #--------------------------------------------------------------------------
    # calculate cost and store in large grid
    cost_val_curr = bout_comp_cost(agree_vol_all,agree_overlap_frac_all,
                                   machine_only_vol_all,user_only_vol_all)
                                   
    cost_grid[ai,bi,ci,di,ei,fi] = cost_val_curr
    
    cc+= 1
    print('Completed {}/{} parameter combinations'.format(cc,A.size))  
    
    if PLOT_FLAG:
        user_only_vol_all = np.asarray(user_only_vol_all)
        machine_only_vol_all = np.asarray(machine_only_vol_all) 
        agree_vol_all = np.asarray(agree_vol_all)               
        agree_overlap_frac_all = np.asarray(agree_overlap_frac_all)
        
        agree_vol_good = agree_vol_all[(agree_overlap_frac_all >= AGREE_FRAC)]
        agree_vol_bad = agree_vol_all[(agree_overlap_frac_all < AGREE_FRAC)] 
                          
        colors = ['blue','grey','red','green']  
        bins = np.arange(0,140,10)              
        
        fig, ax = plt.subplots()
        ax.hist([agree_vol_good,agree_vol_bad,machine_only_vol_all, \
                    user_only_vol_all], bins, histtype='bar',stacked=True,
                    color=colors)
        
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
        if SAVE_PLOT_FLAG:
            hist_fn_str = \
                'bout_comp_a={}_b={}_c={}_d={}_e={}_f={}.png'.format(ai,bi,ci,di,ei,fi)
            hist_save_path = os.path.join(save_path, 'plots',hist_fn_str)
            plt.savefig(hist_save_path)
            plt.close(fig)
      
       
#------------------------------------------------------------------------------
# find minimum of cost function
min_cost = np.nanmin(cost_grid)
min_cost_ind = np.nanargmin(cost_grid)            
min_cost_ind_unravel = np.unravel_index(min_cost_ind,cost_grid.shape)

# find the actual parameters corresponding to this
analysis_params_opt = analysisParams.copy()
    
# reassign values
analysis_params_opt['min_bout_duration'] = min_bout_duration_grid[min_cost_ind_unravel[0]]
analysis_params_opt['min_bout_volume']   = min_bout_volume_grid[min_cost_ind_unravel[1]]
analysis_params_opt['wlevel']            = wlevel_grid[min_cost_ind_unravel[2]]
analysis_params_opt['wtype']             = wtype_grid[min_cost_ind_unravel[3]]
analysis_params_opt['medfilt_window']    = medfilt_window_grid[min_cost_ind_unravel[4]]
analysis_params_opt['mad_thresh']        = mad_thresh_grid[min_cost_ind_unravel[5]]


#------------------------------------------------------------------------------
# look at spread of values


if PLOT_COST_HIST_FLAG:
    fig, ax = plt.subplots()
    nbins = 100
    ax.hist(np.ravel(cost_grid),bins=nbins) ;



#------------------------------------------------------------------------------
# save data?
if SAVE_DATA_FLAG:
    
    # save cost grid
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cost_data_fn = 'cost_matrix_{}.npy'.format(timestr)
    cost_data_fullpath = os.path.join(save_path, cost_data_fn)
    np.save(cost_data_fullpath,cost_grid)
    
    # save the parameter meshgrid that is being searched
    param_grid_fn = 'param_grid_{}'.format(timestr)
    param_grid_fullpath = os.path.join(save_path, param_grid_fn)
    with open(param_grid_fullpath, 'wb') as fp:
        pickle.dump(param_grid_list, fp)
        
    # save the optimal parameters
    opt_params_fn = 'analysis_params_opt_{}'.format(timestr)
    opt_params_fullpath = os.path.join(save_path, opt_params_fn)
    np.save(opt_params_fullpath,analysis_params_opt)