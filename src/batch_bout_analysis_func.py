# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 08:16:39 2017

@author: Fruit Flies
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:55:40 2017

@author: Fruit Flies
"""
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

import os 
#import sys 

import numpy as np
from scipy import interpolate

from load_hdf5_data import load_hdf5
from bout_analysis_func import check_data_set, bout_analysis
from bout_and_vid_analysis import bout_analysis_wTracking

from openpyxl import Workbook
#------------------------------------------------------------------------------

def batch_bout_analysis(channel_entry_list, tmin, tmax, tbin_size, plotFlag=False,
                        combAnalysisFlag=False):

    bouts_list = []
    dset_smooth_list = [] 
    volumes_list = []
    filename_list = []
    filekeyname_list = [] 
    groupkeyname_list = []
    name_list = [] 
    t_global = np.array([])

    for entry in channel_entry_list:
        filepath, filekeyname, groupkeyname = entry.split(', ',2)
        dset, t = load_hdf5(filepath,filekeyname,groupkeyname)        
        
        bad_data_flag, dset, t, frames = check_data_set(dset,t)
        
        # error case for bad dataset
        if bad_data_flag:
            messagestr = "Bad dataset: " + filepath + ", " + filekeyname + ", " + groupkeyname
            print(messagestr)
            continue 
    
        # analyze each channel, with or without video data as aid        
        if combAnalysisFlag:
            dset_smooth, bouts, volumes = bout_analysis_wTracking(filepath,
                                                    filekeyname, groupkeyname)
        else:
            dset_smooth, bouts, volumes, _ = bout_analysis(dset,frames)
        
        # trim analysis output based on time limits
        if not ( np.isnan(tmin) or np.isnan(tmax)):
            dset_smooth, bouts, volumes, t = trim_bout_data(dset_smooth,bouts,
                                                            volumes,t,tmin,tmax)
        
        # find global time (won't be exact, but should only be off by O(ms))
        if t.size > t_global.size:
            t_global = t
        
        # add data to lists 
        bouts_list.append(bouts)
        dset_smooth_list.append(dset_smooth)
        volumes_list.append(volumes)
        
        _, name = os.path.split(filepath)
        filename_list.append(name[0:-5])
        filekeyname_list.append(filekeyname)
        groupkeyname_list.append(groupkeyname)
        
        name_full = name + ", " + filekeyname + ", " + groupkeyname
        name_list.append(name_full)
    
    # limits for time axis. assigned to full t range if no values are entered
    if np.isnan(tmin):
        tmin = np.min(t_global)
    if np.isnan(tmax):
        tmax = np.max(t_global)
    if np.isnan(tbin_size):
        tbin_size = 20     
    
    # calculate raster array 
    idx_min = np.searchsorted(t_global,tmin,side='right')
    idx_max = np.searchsorted(t_global,tmax,side='right')
    idx_bin_size = np.searchsorted(t_global - t_global[0], tbin_size,side='right')
    raster_im = np.zeros([len(bouts_list),idx_max-idx_min])
    for ith, bouts_curr in enumerate(bouts_list):
        bouts_curr = bouts_curr[:,bouts_curr[1,:]<idx_max]
        bouts_curr = bouts_curr[:,bouts_curr[0,:]>idx_min]
        for jth in np.arange(0,bouts_curr.shape[1]):
            raster_im[ith,bouts_curr[0,jth]:bouts_curr[1,jth]] = 1 
    
    # find consumption per unit time
    diff_list = []
    for dset in dset_smooth_list:
        dset_diff = np.diff(dset)
        dset_diff_length = dset_diff.size
        max_ind_curr = np.amin(np.array([dset_diff_length, idx_max-1]))
        num_zeros = (idx_max-1) - max_ind_curr 
        dset_diff = dset_diff[0:max_ind_curr]
        dset_diff = np.append(dset_diff, np.zeros(num_zeros))
        dset_diff = np.insert(dset_diff,0,0)
        diff_list.append(dset_diff)
    
    diff_mat = np.vstack(diff_list)
    diff_mat = -1.0*diff_mat[:,idx_min:]     
    
    
    # get summaries for time bin and fly 
    bin_edges = np.arange(idx_min, idx_max, idx_bin_size)
    total_consumption = np.sum(raster_im*diff_mat,axis=0)
    
    consumption_hist = [np.sum(total_consumption[ind:ind+idx_bin_size]) for ind in bin_edges]
    consumption_per_fly = np.sum(raster_im*diff_mat,axis=1)
    duration_per_fly = np.sum(raster_im,axis=1)
    latency_per_fly = []
    for bouts_temp in bouts_list:
        if bouts_temp.size > 0 :
            latency_per_fly.append(bouts_temp[0,0])
        else:    
            latency_per_fly.append(tmax)
    
    latency_sort_ind = np.argsort(latency_per_fly)
    name_list_sorted = [name_list[sort_ind] for sort_ind in latency_sort_ind]

    if plotFlag:        
        #------------------------------------------------------------------------------
        # make raster plot of bouts             
        
        fig_raster, ax_raster = plt.subplots()
        #ax_raster.imshow(raster_im[latency_sort_ind,:], aspect='auto',cmap='Greys',interpolation='none')
        raster_im_masked = np.ma.masked_array(raster_im, raster_im < 1)
        m = ax_raster.pcolormesh(raster_im_masked[np.flipud(latency_sort_ind),:],vmin=0,vmax=1,cmap='Greys')
        m.set_rasterized(False)
        
        ax_raster.set_yticks(np.arange(0,len(bouts_list)))
        ax_raster.set_yticklabels(reversed(name_list_sorted))
        ax_raster.set_xlabel("Time [s]")
        ax_raster.set_ylabel("Data File")
        ax_raster.set_title("Feeding Bouts")    
        fig_raster.set_tight_layout(True)

        # show raster
        fig_raster.show()

        #NEED TO MAKE AXIS CORRESPOND TO TIME
        #x_tick_labels = t_global[]
            
        #------------------------------------------------------------------------------
        # make histogram of food consumption
        
        fig_hist, ax_hist = plt.subplots()
        ax_hist.fill_between(t_global[bin_edges], 0, consumption_hist, 
                             facecolor='green', alpha=0.5)
        
        ax_hist.set_xlabel("Time [s]")
        ax_hist.set_ylabel("Food consumed [nL]")
        ax_hist.set_title("Total Consumption Histogram")

        # show histogram plot
        fig_hist.show()

        return (bouts_list, name_list, volumes_list, consumption_per_fly, 
                duration_per_fly, latency_per_fly,fig_raster, fig_hist)
    else:
        return (bouts_list, name_list, volumes_list, consumption_per_fly, 
                duration_per_fly, latency_per_fly)        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------    
# make xlsx file for batch 

def save_batch_xlsx(save_name, bouts_list,name_list,volumes_list, 
                    consumption_per_fly, duration_per_fly, latency_per_fly):
    wb = Workbook()    
    
    ws_summary = wb.active
    ws_summary.title = "Summary"
    ws_events = wb.create_sheet("Events")
    
    # summary page
    summary_heading = ["Filename", "XP", "Channel", "Number of Meals", 
                       "Total Volume (nL)", "Total Duration Eating (s)",
                        "Latency to Eat (s)" ]
    ws_summary.append(summary_heading)
    
    for ith, bouts_curr in enumerate(bouts_list):
        name_curr, xp_curr, channel_curr  = name_list[ith].split(', ',2)
        event_num_curr = float(bouts_curr.shape[1])
        total_vol_curr = float(consumption_per_fly[ith])
        duration_curr = float(duration_per_fly[ith])
        latency_curr = float(latency_per_fly[ith])
        
        row_curr = [name_curr, xp_curr, channel_curr, event_num_curr, 
                    total_vol_curr, duration_curr, latency_curr]
        ws_summary.append(row_curr)
    
    #events page
    events_heading = ["Filename", "XP", "Channel", "StartIdx", "EndIdx",  
                      "DurationIdx", "Volume (nL)"]
    ws_events.append(events_heading)
    
    for ith, bouts_curr in enumerate(bouts_list):
        name_curr, xp_curr, channel_curr  = name_list[ith].split(', ',2)
        #ws_events.append([name_curr, xp_curr, channel_curr])
        
        volumes_curr = volumes_list[ith]
        for jth in np.arange(0,bouts_curr.shape[1]):
            bout_start_curr = float(bouts_curr[0,jth])
            bout_end_curr = float(bouts_curr[1,jth])
            volume_curr = float(volumes_curr[jth])
            duration_curr = bout_end_curr - bout_start_curr 
            #row_curr = [" "," "," ",bout_start_curr,bout_end_curr,duration_curr,volume_curr]
            row_curr = [name_curr,xp_curr,channel_curr,bout_start_curr, 
                        bout_end_curr,duration_curr,volume_curr]
            ws_events.append(row_curr) 
            
    wb.save(save_name)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------    
# trim data sets and bout data based on time limits
def trim_bout_data(dset_smooth, bouts, volumes, t, tmin, tmax):
    # first find channel time indices corresponding to max and min t
    idx_min = np.searchsorted(t,tmin,side='right')
    idx_max = np.searchsorted(t,tmax,side='right')
    
    # make sure that these indices don't exceed array bounds?
    #    
    
    # now cut the input variables based on time range
    dset_smooth = dset_smooth[idx_min:idx_max]
    t = t[idx_min:idx_max]
    
    # ...including bout and volume data
    volumes = volumes[np.logical_and(bouts[1,:]<idx_max, bouts[0,:]>idx_min)]
    #volumes = volumes[bouts[0,:]>idx_min]
    #bouts_temp = bouts.copy()
    bouts = bouts[:,bouts[1,:]<idx_max]
    bouts = bouts[:,bouts[0,:]>idx_min]
    
    return (dset_smooth, bouts, volumes, t )