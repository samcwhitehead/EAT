# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:08:13 2018

@author: Fruit Flies
"""
#------------------------------------------------------------------------------
import sys
import os

if sys.version_info[0] < 3:
    from Tkinter import *
    import tkFileDialog
    from ttk import *
    import tkMessageBox
else:
    from tkinter import *
    from tkinter.ttk import *
    from tkinter import filedialog as tkFileDialog
    from tkinter import messagebox as tkMessageBox
    
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.widgets import MultiCursor, RadioButtons
#from pylab import show

import csv

#import h5py
from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis

#------------------------------------------------------------------------------
def annotate_channel_data(data_filename_full, filekeyname, groupkeyname):
    #------------------------------------------------------------------------------
    # print some instructions
    #------------------------------------------------------------------------------
    print(50*'-')
    print("USER ANNOTATION FOR EXPRESSO DATA FILES")
    print(50*'-')
    print('- Double LEFT click to indicate bout START')
    print('- Double RIGHT click to indicate bout END')
    print('- Press Z key to delete previous bout START selection')
    print('- Press X key to delete previous bout END selection')
    print('- Press B key to save results and close plot window')
    print('- Press N key to exit without saving')
    print('')
    print('*Make sure to do all button presses/clicks when plot window is selected')
    print(50*'-')
    
    #------------------------------------------------------------------------------
    # handle file names appropriately
    #------------------------------------------------------------------------------
    if sys.version_info[0] < 3:
        filekeyname = unicode(filekeyname) 
        groupkeyname = unicode(groupkeyname) 
    
    
    data_dir, data_filename = os.path.split(data_filename_full) 
    save_dir = data_dir 
    
    data_filename_no_ext = os.path.splitext(data_filename)[0]
    save_filename =  str(data_filename_no_ext) + '_' +  filekeyname + '_' + \
                     groupkeyname + '_ANNOTATION.csv'
    save_filepath = os.path.join(save_dir, save_filename)
    
    
    #------------------------------------------------------------------------------
    #load and analyze data 
    #------------------------------------------------------------------------------
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
    
    #------------------------------------------------------------------------------          
    # make plots to display results 
    #------------------------------------------------------------------------------  
    bout_start_list = [] 
    bout_end_list = [] 
    
    bout_start_ind_list = [] 
    bout_end_ind_list = [] 
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(17, 7))
            
    ax1.set_ylabel('Liquid [nL]')
    ax2.set_ylabel('Liquid [nL]')
    ax2.set_xlabel('Time [s]')
    ax1.set_title(data_filename + ', ' + filekeyname + ', ' + groupkeyname,
                  fontsize=20)
    ax2.set_title('Smoothed Data')
    
    ax1.plot(t,dset,'k-')
    ax2.plot(t, dset_smooth,'k-')
        
    ax1.set_xlim([t[0],t[-1]])
    ax1.set_ylim([np.amin(dset),np.amax(dset)])
    
    ax1.grid(True)
    ax2.grid(True)
    
    multi = MultiCursor(fig.canvas, (ax1, ax2), color='cyan', lw=1.0, 
                    useblit=True, horizOn=True,  vertOn=True)
    
    #------------------------------------------------------------------------------          
    # handle the indexing of starting/ending bout lines
    #------------------------------------------------------------------------------  
    global start_lines, end_lines, line_counter, N_data_lines 
    start_lines = np.array([],dtype=int)
    end_lines = np.array([],dtype=int)
    line_counter = len(ax1.lines) # this should account for initial data lines
    N_data_lines = len(ax1.lines) 
    
    #------------------------------------------------------------------------------
    # define click events
    #------------------------------------------------------------------------------                
    def onclick(event):
        global start_lines, end_lines, line_counter
        if event.dblclick:
            t_pick = event.xdata 
            t_closest_ind = np.searchsorted(t,t_pick,side='right')
            t_closest = t[t_closest_ind]
            
            # DOUBLE LEFT CLICK TO SELECT BOUT START IND
            if event.button == 1: 
                bout_start_list.append(t_closest)
                bout_start_ind_list.append(t_closest_ind)
                print('Selected bout start:')
                print(t_closest)
                ax1.axvline(x=t_closest,color='g')
                ax2.axvline(x=t_closest,color='g')
                
                fig.canvas.draw()
                
                start_lines = np.append(start_lines,line_counter)
                line_counter += 1
                
            # DOUBLE RIGHT CLICK TO SELECT BOUT END IND    
            elif event.button == 3: 
                bout_end_list.append(t_closest)
                bout_end_ind_list.append(t_closest_ind)
                print('Selected bout end:')
                print(t_closest)
                ax1.axvline(x=t_closest,color='r')
                ax2.axvline(x=t_closest,color='r')
                fig.canvas.draw()
                
                end_lines = np.append(end_lines,line_counter)
                line_counter += 1
                
            else:
                print(event.button)    
              
    def onpress(event):
        global start_lines, end_lines, line_counter
        # Z KEY TO DELETE PREVIOUS START IND SELECTION
        if event.key.lower() == 'z' and len(bout_start_list) > 0 :
            #num_bout_starts = len(bout_start_list)
            del(bout_start_list[-1])
            del(bout_start_ind_list[-1])
            
            curr_line_idx = start_lines[-1]
            
            ax1.lines[curr_line_idx].remove()
            ax2.lines[curr_line_idx].remove()
            
            start_lines = start_lines[start_lines != curr_line_idx]
            start_lines[start_lines > curr_line_idx] += -1 
            end_lines[end_lines > curr_line_idx] += -1 
            
            line_counter += -1 
            
            fig.canvas.draw() 
        # X KEY TO DELETE PREVIOUS END IND SELECTION    
        elif event.key.lower() == 'x' and len(bout_end_list) > 0 :
            del(bout_end_list[-1])
            del(bout_end_ind_list[-1])
            
            curr_line_idx = end_lines[-1]
            
            ax1.lines[curr_line_idx].remove()
            ax2.lines[curr_line_idx].remove()
            
            end_lines = end_lines[end_lines != curr_line_idx]
            start_lines[start_lines > curr_line_idx] += -1 
            end_lines[end_lines > curr_line_idx] += -1 
            
            line_counter += -1 
           
            fig.canvas.draw()
           
        # B KEY TO SAVE RESULTS TO FILE AND EXIT    
        elif event.key.lower() == 'b':
            bout_start_array = np.sort(np.asarray(bout_start_list))
            bout_end_array = np.sort(np.asarray(bout_end_list))
            
            bout_start_ind_array = np.sort(np.asarray(bout_start_ind_list))
            bout_end_ind_array = np.sort(np.asarray(bout_end_ind_list))
            
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
            return
            #quit()
        elif event.key.lower() == 'n': 
            plt.close()
            return
        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)  
    pid = fig.canvas.mpl_connect('key_press_event', onpress)                  
    
    return (multi, cid, pid)