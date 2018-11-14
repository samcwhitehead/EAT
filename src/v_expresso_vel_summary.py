# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 13:52:37 2018

Used to select hdf5 files output by process_visual_expresso and pull out 
average velocity during movement and percent time spent moving 
@author: Fruit Flies
"""
#------------------------------------------------------------------------------
import sys
import os
import numpy as np
import csv
import h5py

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

#------------------------------------------------------------------------------
# request video file(s) for conversion
h5_filenames = tkFileDialog.askopenfilenames(initialdir=sys.path[0],
                                             title='Select files to analyze') 
# print results                                             
if h5_filenames:
    print(h5_filenames)
                                             
# select file to save to 
csv_filename = tkFileDialog.asksaveasfilename(initialdir=sys.path[0],
                                           defaultextension=".csv",
                                           title='Select save filename') 
# headers for csv columns
column_headers = ['Filename', 'Bank', 'Channel', 'Cumulative Dist. (cm)',
                      'Average Speed (cm/s)', 'Fraction Time Moving']

if sys.version_info[0] < 3:
    out_path = open(csv_filename,mode='wb')
else:
    out_path = open(csv_filename, 'w', newline='')
save_writer = csv.writer(out_path)                      
save_writer.writerow(column_headers)                      

    
for h5_fn in h5_filenames:
    filepath = os.path.splitext(h5_fn)[0]  
    
    
    with h5py.File(h5_fn,'r') as f:
        t = f['Time']['t'].value
        N_flies = 1
        for fly_num in np.arange(N_flies):
            
                
            try:
                cum_dist = f['BodyCM']['cum_dist_%02d'%(fly_num)].value 
                vel_mag = f['BodyVel']['vel_mag_%02d'%(fly_num)].value 
                moving_ind = f['BodyVel']['moving_ind_%02d'%(fly_num)].value 
                
                # parameters we want to save
                mean_speed = np.mean(vel_mag[moving_ind])
                cum_dist_max = cum_dist[-1]
                perc_moving = float(np.sum(moving_ind)) / float(moving_ind.size)                
                
                # file identifiers
                filepath_split = filepath.split("/")[-1]
                filename_curr = '_'.join(filepath_split.split('_')[:-5])
                bank_curr = filepath_split.split('_')[-5]
                channel_curr = filepath_split.split('_')[-4] + '_' + \
                                filepath_split.split('_')[-3]
                
                row_mat = np.vstack((filename_curr, bank_curr, channel_curr,
                                     cum_dist_max, mean_speed, perc_moving))
                row_mat = np.transpose(row_mat) 
            except KeyError:
                print('Invalid File Selected:')
                print(h5_fn)
                continue 

            for row in row_mat:
                save_writer.writerow(row)
                
out_path.close()
          