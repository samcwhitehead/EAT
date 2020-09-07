# -*- coding: utf-8 -*-
"""
Created on Sat Jul 04 11:37:30 2020

@author: Fruit Flies
"""

import matplotlib

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os
import sys
import h5py
import numpy as np
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, MultiCursor

from load_hdf5_data import load_hdf5
from bout_analysis_func import check_data_set, plot_channel_bouts, bout_analysis
from batch_bout_analysis_func import batch_bout_analysis, save_batch_xlsx
from v_expresso_gui_params import (initDirectories, guiParams, trackingParams)
from bout_and_vid_analysis import (channel2basic, vid2basic, basic2channel,
                                   basic2vid, flyCombinedData_to_hdf5,
                                   save_comb_time_series, save_comb_summary,
                                   bout_analysis_wTracking, merge_v_expresso_data,
                                   plot_bout_aligned_var, save_comb_summary_hdf5)
from v_expresso_image_lib import (visual_expresso_tracking_main,
                                  process_visual_expresso,
                                  plot_body_cm, plot_body_vel,
                                  plot_body_angle, plot_moving_v_still,
                                  plot_cum_dist, hdf5_to_flyTrackData,
                                  save_vid_time_series, save_vid_summary,
                                  batch_plot_cum_dist, batch_plot_heatmap)

# allows drag and drop functionality. if you don't want this, or are having 
#  trouble with the TkDnD installation, set to false.
try:
    from TkinterDnD2 import *
    from gui_setup_util import (buildButtonListboxPanel, buildBatchPanel, bindToTkDnD, myEntryOptions)

    TKDND_FLAG = True
except ImportError:
    print('Error: could not load TkDnD libraries. Drag/drop disabled')
    from gui_setup_util import buildButtonListboxPanel, buildBatchPanel, myEntryOptions
## ============================================================================

# =============================================================================
""" Top UI frame containing the list of directories to be scanned. """


# =============================================================================
class DirectoryFrame(Frame):
    def __init__(self, parent, col=0, row=0, filedir=None):
        Frame.__init__(self, parent.master)

        # define names/labels for buttons + listboxes
        self.button_names = ['add', 'remove', 'clear_all']
        self.button_labels = ['Add Directory', 'Remove Directory', 'Clear All']
        self.label_str = 'Directory list:'

        self.listbox_name = 'directories'
        # generate general frame with buttons + listbox
        self = buildButtonListboxPanel(self, parent.master, self.button_names,
                                       self.button_labels, self.label_str,
                                       row=row, col=col)

        # assign button callbacks
        self.buttons['add']['command'] = lambda: self.add_items(parent)
        self.buttons['remove']['command'] = self.rm_items
        self.buttons['clear_all']['command'] = self.clear_all

        # switch remove button to 'disabled' until something is selected
        self.buttons['remove']['state'] = DISABLED

        # bind listbox
        self.listbox.bind('<<ListboxSelect>>', self.on_select)
        if TKDND_FLAG:
            self = bindToTkDnD(self, self.listbox_name)

        # make sure listbox retains selection
        self.listbox['exportselection'] = False

    # -----------------------------------------
    # Callback functions
    # ----------------------------------------- 
    # enable/disable buttons upon listbox cursor select
    def on_select(self, selection):
        if self.listbox.curselection():
            self.buttons['remove']['state'] = NORMAL
        else:
            self.buttons['remove']['state'] = DISABLED

    # add user-selected directory to listbox
    def add_items(self, parent):
        current_entries = self.listbox.get(0, END)
        new_entry = Expresso.get_dir(parent)
        if new_entry not in current_entries:
            self.listbox.insert(END, new_entry)

    # remove selected directory from listbox
    def rm_items(self):
        # Reverse sort the selected indexes to ensure all items are removed
        selected = sorted(self.listbox.curselection(), reverse=True)
        for item in selected:
            self.listbox.delete(item)

    # clear all directories from listbox
    def clear_all(self):
        self.listbox.delete(0, END)


# =============================================================================
""" UI frame containing the list of channel data files to analyze  """


# =============================================================================
class FileDataFrame(Frame):
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)

        # define names/labels for buttons + listboxes
        self.button_names = ['add', 'remove', 'clear_all']
        self.button_labels = ['Get HDF5 Files', 'Remove Files', 'Clear All']
        self.label_str = 'Channel data files:'
        self.listbox_name = 'channels'

        # generate general frame with buttons + listbox
        self = buildButtonListboxPanel(self, parent.master, self.button_names,
                                       self.button_labels, self.label_str,
                                       row=row, col=col)

        # assign button callbacks
        self.buttons['add']['command'] = lambda: self.add_items(parent)
        self.buttons['remove']['command'] = self.rm_items
        self.buttons['clear_all']['command'] = self.clear_all

        # switch remove button to 'disabled' until something is selected
        self.buttons['remove']['state'] = DISABLED

        # bind listbox
        self.listbox.bind('<<ListboxSelect>>', self.on_select)
        if TKDND_FLAG:
            self = bindToTkDnD(self, self.listbox_name)

    # ----------------------------------------- 
    # Callback functions
    # ----------------------------------------- 
    # enable/disable buttons upon listbox cursor select
    # enable/disable buttons upon listbox cursor select
    def on_select(self, selection):
        if self.listbox.curselection():
            self.buttons['remove']['state'] = NORMAL
        else:
            self.buttons['remove']['state'] = DISABLED

    # add channel data files (hdf5) from selected directory to listbox
    def add_items(self, parent):
        newfiles = Expresso.scan_dirs(parent, 'channel')
        file_list = self.listbox.get(0, END)
        if len(newfiles) > 0:
            for file in tuple(newfiles):
                if file not in file_list:
                    self.listbox.insert(END, file)

    # remove selected file from listbox
    def rm_items(self):
        selected = sorted(self.listbox.curselection(), reverse=True)
        for item in selected:
            self.listbox.delete(item)

    # clear all files from listbox
    def clear_all(self):
        self.listbox.delete(0, END)


# =============================================================================
""" UI frame containing the bank groups for feeding channel data  """


# =============================================================================
class XPDataFrame(Frame):
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)

        # define names/labels for buttons + listboxes
        self.button_names = ['add', 'remove', 'clear_all']
        self.button_labels = ['Unpack HDF5 Files', 'Remove XP', 'Clear All']
        self.label_str = 'XP list:'

        # generate general frame with buttons + listbox
        self = buildButtonListboxPanel(self, parent.master, self.button_names,
                                       self.button_labels, self.label_str,
                                       row=row, col=col)

        # assign button callbacks
        self.buttons['add']['command'] = lambda: self.add_items(parent)
        self.buttons['remove']['command'] = self.rm_items
        self.buttons['clear_all']['command'] = self.clear_all

        # switch remove button to 'disabled' until something is selected
        self.buttons['remove']['state'] = DISABLED

        # bind listbox
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

    # ----------------------------------------- 
    # Callback functions
    # ----------------------------------------- 
    # enable/disable buttons upon listbox cursor select
    def on_select(self, selection):
        if self.listbox.curselection():
            self.buttons['remove']['state'] = NORMAL
        else:
            self.buttons['remove']['state'] = DISABLED

    # add bank (xp) list from selected feeding data files to listbox
    def add_items(self, parent):
        current_entries = self.listbox.get(0, END)
        new_entries = Expresso.unpack_files(parent)
        for ent in tuple(new_entries):
            if ent not in current_entries:
                self.listbox.insert(END, ent)

    # remove selected bank from listbox
    def rm_items(self):
        # Reverse sort the selected indexes to ensure all items are removed
        selected = sorted(self.listbox.curselection(), reverse=True)
        for item in selected:
            self.listbox.delete(item)

    # clear all banks from listbox
    def clear_all(self):
        self.listbox.delete(0, END)


# =============================================================================
""" UI frame containing channels (subdivision of banks) for feeding data  """


# =============================================================================
class ChannelDataFrame(Frame):
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)

        # define names/labels for buttons + listboxes
        self.button_names = ['add', 'remove', 'clear_all', 'plot', 'save']
        self.button_labels = ['Unpack XP', 'Remove Channels', 'Clear All',
                              'Plot Channel', 'Save CSV']
        self.label_str = 'Channel list:'

        # generate general frame with buttons + listbox
        self = buildButtonListboxPanel(self, parent.master, self.button_names,
                                       self.button_labels, self.label_str,
                                       row=row, col=col)

        # assign button callbacks
        self.buttons['add']['command'] = lambda: self.add_items(parent)
        self.buttons['remove']['command'] = self.rm_items
        self.buttons['clear_all']['command'] = self.clear_all
        self.buttons['plot']['command'] = lambda: self.plot_channel(parent)
        self.buttons['save']['command'] = lambda: self.save_results(parent)

        # switch buttons to 'disabled' until something is selected
        self.buttons['remove']['state'] = DISABLED
        self.buttons['plot']['state'] = DISABLED
        self.buttons['save']['state'] = DISABLED

        # bind listbox
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

    # ----------------------------------------- 
    # Callback functions
    # ----------------------------------------- 
    # enable/disable buttons upon listbox cursor select
    def on_select(self, selection):
        if self.listbox.curselection():
            self.buttons['remove']['state'] = NORMAL
            self.buttons['plot']['state'] = NORMAL
            self.buttons['save']['state'] = NORMAL
        #            self.selection_ind = sorted(self.listbox.curselection(),
        #                                        reverse=False)
        else:
            self.buttons['remove']['state'] = DISABLED
            self.buttons['plot']['state'] = DISABLED
            self.buttons['save']['state'] = DISABLED

    # -------------------------------------------------
    # add channels from selected banks to listbox
    def add_items(self, parent):
        channel_list = self.listbox.get(0, END)
        newchannels = Expresso.unpack_xp(parent)
        for channel in tuple(newchannels):
            if channel not in channel_list:
                self.listbox.insert(END, channel)

    # -------------------------------------------------
    # remove selected channels from listbox
    def rm_items(self):
        selected = sorted(self.listbox.curselection(), reverse=True)
        for item in selected:
            self.listbox.delete(item)

    # -------------------------------------------------      
    # clear all elements of listbox
    def clear_all(self):
        self.listbox.delete(0, END)

    # -------------------------------------------------
    # plot feeding data for current channel
    def plot_channel(self, parent):
        # read in current selection, as well as boolean state vars
        selected_ind = sorted(self.listbox.curselection(), reverse=False)
        menu_debug_flag = parent.debug_bout.get()
        menu_save_flag = parent.save_all_plots.get()

        # loop through selected channels and plot
        for ind in selected_ind:
            channel_entry = self.listbox.get(ind)

            # analyze current channel
            dset, frames, t, dset_smooth, bouts, volumes = Expresso.get_channel_data(parent, channel_entry,
                                                                                     DEBUG_FLAG=menu_debug_flag)

            # if data set is not abnormal, skip
            if dset.size == 0:
                print('Cannot read data from {} -- skipping'.format(channel_entry))
                continue

            # otherwise, make plot
            fig, ax1, ax2 = plot_channel_bouts(dset, dset_smooth, t, bouts)

            # turn on cursor
            multi = MultiCursor(fig.canvas, (ax1, ax2), color='dodgerblue', lw=1.0, useblit=True, horizOn=True,
                                vertOn=True)

            # get file info for title
            filepath, xp_str, channel_str = channel_entry.split(', ', 2)
            dirpath, fn = os.path.split(filepath)
            channel_name_full = ", ".join([fn, xp_str, channel_str])

            fig.canvas.set_window_title(channel_name_full)

            # save this plot if the menu option to save all plots is selected
            if menu_save_flag:
                suffix_str = 'bout_detection.png'
                filename_no_ext = os.path.splitext(fn)[0]
                save_filename = '_'.join((filename_no_ext, xp_str, channel_str, suffix_str))
                savename_full = os.path.join(dirpath, save_filename)
                fig.savefig(savename_full)

    # ------------------------------------------------
    # save summary results in csv format
    def save_results(self, parent):
        selected_ind = sorted(self.listbox.curselection(), reverse=False)
        if len(selected_ind) != 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Please select only one channel for plotting individual traces')
            return

        full_listbox_entry = self.listbox.get(selected_ind[0])

        _, _, self.t, _, self.bouts, self.volumes = Expresso.get_channel_data(parent, full_listbox_entry)

        # full_listbox_entry = self.listbox.get(self.selection_ind[0])
        filepath, filekeyname, groupkeyname = full_listbox_entry.split(', ', 2)
        dirpath, filename = os.path.split(filepath)
        self.channel_name_full = filename + ", " + filekeyname + ", " + groupkeyname

        if self.bouts.size > 0:
            bouts_transpose = np.transpose(self.bouts)
            volumes_col = self.volumes.reshape(self.volumes.size, 1)
            row_mat = np.hstack((bouts_transpose, self.t[bouts_transpose], volumes_col))

            # prompt user to select location to save output
            if sys.version_info[0] < 3:
                save_file = tkFileDialog.asksaveasfile(mode='wb',
                                                       defaultextension=".csv")
                # check that file selection is valid
                if not save_file:
                    tkMessageBox.showinfo(title='Error', message='Invalid filename')
                    return

                # otherwise save results to csv
                save_writer = csv.writer(save_file)
            else:
                save_filename = tkFileDialog.asksaveasfilename(defaultextension=".csv")
                # check that file selection is valid
                if not save_filename:
                    tkMessageBox.showinfo(title='Error', message='Invalid filename')
                    return

                # otherwise save results to csv
                save_file = open(save_filename, 'w', newline='')
                save_writer = csv.writer(save_file)

            save_writer.writerow([self.channel_name_full])
            save_writer.writerow(['Bout Number'] + ['Bout Start [idx]'] + \
                                 ['Bout End [idx]'] + ['Bout Start [s]'] + ['Bout End [s]'] + ['Volume [nL]'])
            cc = 1
            for row in row_mat:
                new_row = np.insert(row, 0, cc)
                save_writer.writerow(new_row)
                cc += 1
        else:
            tkMessageBox.showinfo(title='Error',
                                  message='No feeding bouts to save')


# =============================================================================
""" UI frame containing video files for tracking  """


# =============================================================================
class VideoDataFrame(Frame):
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)

        # define names/labels for buttons + listboxes
        self.button_names = ['add', 'remove', 'clear_all', 'analyze', 'plot']
        self.button_labels = ['Get Video Files', 'Remove Files', 'Clear All',
                              'Analyze Video', 'Plot Results']
        self.label_str = 'Video files:'

        self.listbox_name = 'videos'
        # generate general frame with buttons + listbox
        self = buildButtonListboxPanel(self, parent.master, self.button_names,
                                       self.button_labels, self.label_str,
                                       row=row, col=col)

        # assign button callbacks
        self.buttons['add']['command'] = lambda: self.add_items(parent)
        self.buttons['remove']['command'] = self.rm_items
        self.buttons['clear_all']['command'] = self.clear_all
        self.buttons['analyze']['command'] = lambda: self.analyze_vid(parent)
        self.buttons['plot']['command'] = lambda: self.plot_results(parent)

        # switch buttons to 'disabled' until something is selected
        self.buttons['remove']['state'] = DISABLED
        self.buttons['analyze']['state'] = DISABLED
        self.buttons['plot']['state'] = DISABLED

        # bind listbox
        self.listbox.bind('<<ListboxSelect>>', self.on_select)
        if TKDND_FLAG:
            self = bindToTkDnD(self, self.listbox_name)

    # -----------------------------------------
    # Callback functions
    # ----------------------------------------- 
    # enable/disable buttons upon listbox cursor select
    def on_select(self, selection):
        if self.listbox.curselection():
            self.buttons['remove']['state'] = NORMAL
            self.buttons['analyze']['state'] = NORMAL
            self.buttons['plot']['state'] = NORMAL
        #            self.selection_ind = sorted(self.listbox.curselection(),
        #                                        reverse=False)
        else:
            self.buttons['remove']['state'] = DISABLED
            self.buttons['analyze']['state'] = DISABLED
            self.buttons['plot']['state'] = DISABLED

    # ----------------------------------------------------------
    # scan selected directory for video files to add to listbox
    def add_items(self, parent):
        newfiles = Expresso.scan_dirs(parent, 'video')
        file_list = self.listbox.get(0, END)
        if len(newfiles) > 0:
            for file in tuple(newfiles):
                if file not in file_list:
                    self.listbox.insert(END, file)

    # ----------------------------------------------------------
    # remove selected video from listbox
    def rm_items(self):
        selected = sorted(self.listbox.curselection(), reverse=True)
        for item in selected:
            self.listbox.delete(item)

    # ----------------------------------------------------------
    # clear all videos in listbox
    def clear_all(self):
        self.listbox.delete(0, END)

    # ----------------------------------------------------------
    # perform tracking analysis on selected video
    def analyze_vid(self, parent):
        # get current video selection
        selected_ind = sorted(self.listbox.curselection(), reverse=False)
        if len(selected_ind) != 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Please select only one video file for analysis')
            return

        # determine whether or not we should use debugging options
        menu_debug_flag = parent.debug_tracking.get()
        save_debug_flag = not menu_debug_flag

        # get info from file name
        file_entry = self.listbox.get(selected_ind[0])
        file_path, filename = os.path.split(file_entry)

        # name to save tracking results file to
        filename_prefix = os.path.splitext(filename)[0]
        track_filename = filename_prefix + "_TRACKING.hdf5"
        track_filename_processed = filename_prefix + "_TRACKING_PROCESSED.hdf5"

        # check if we've already analyzed this video
        overWriteFlag = True
        if os.path.exists(os.path.join(file_path, track_filename_processed)) and save_debug_flag:
            yes_no_str = 'Tracking data already exists for {} -- do you wish to overwrite?'
            overWriteFlag = tkMessageBox.askyesno(title='Overwrite?',
                                                  message=yes_no_str.format(filename_prefix))

        # then analyze or not depending on whether or not we want to overwrite (if we're saving)
        if overWriteFlag:
            # get raw center of mass tracking
            visual_expresso_tracking_main(file_path, filename, DEBUG_BG_FLAG=menu_debug_flag,
                                          DEBUG_CM_FLAG=menu_debug_flag, SAVE_DATA_FLAG=save_debug_flag,
                                          ELLIPSE_FIT_FLAG=False, PARAMS=trackingParams)

            # process raw tracking results
            process_visual_expresso(file_path, track_filename, SAVE_DATA_FLAG=save_debug_flag, DEBUG_FLAG=False)

        else:
            print('Not overwriting {} -- exiting'.format(filename_prefix))

    def plot_results(self, parent):
        selected_ind = sorted(self.listbox.curselection(), reverse=False)
        if len(selected_ind) != 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Please select only one video file for plotting')
            return

        file_entry = self.listbox.get(selected_ind[0])
        file_path, filename = os.path.split(file_entry)

        menu_save_flag = parent.save_all_plots.get()

        try:
            savename_prefix = os.path.splitext(filename)[0]
            save_filename = os.path.join(file_path, savename_prefix + \
                                         "_TRACKING_PROCESSED.hdf5")
            save_filename = os.path.abspath(save_filename)
            # save_filename = os.path.normpath(save_filename)

            if os.path.exists(save_filename):
                _, track_filename = os.path.split(save_filename)
                flyTrackData_smooth = hdf5_to_flyTrackData(file_path, track_filename)
                plot_body_cm(flyTrackData_smooth, SAVE_FLAG=menu_save_flag)
                plot_body_vel(flyTrackData_smooth, SAVE_FLAG=menu_save_flag)
                plot_moving_v_still(flyTrackData_smooth, SAVE_FLAG=menu_save_flag)
                plot_cum_dist(flyTrackData_smooth, SAVE_FLAG=menu_save_flag)
            else:
                tkMessageBox.showinfo(title='Error',
                                      message='Please analyze this video first ' + \
                                              'using the <Analyze Video> button')
                return
        except AttributeError:
            tkMessageBox.showinfo(title='Error',
                                  message='Please analyze this video first ' + \
                                          'using the <Analyze Video> button')
            return


# =============================================================================
""" UI frame containing batch analysis for just feeding data  """


# =============================================================================
class BatchChannelFrame(Frame):
    def __init__(self, parent, root, col=0, row=0):
        Frame.__init__(self, parent)

        # define names/labels for buttons + listboxes
        self.button_names = [['add', 'remove', 'clear_all'],
                             ['analyze', 'save_sum', 'save_ts']]
        self.button_labels = [['Add Channel(s) to Batch', 'Remove Selected	',
                               'Clear All'],
                              ['Plot Channel Analysis', 'Save Channel Summary',
                               'Save Time Series']]

        # generate basic panel layout                      
        buildBatchPanel(self, self.button_names, self.button_labels,
                        tboxFlag=True, row=row, col=col)

        # assign button callbacks
        self.buttons['add']['command'] = lambda: self.add_items(root)
        self.buttons['remove']['command'] = self.rm_items
        self.buttons['clear_all']['command'] = self.clear_all
        self.buttons['analyze']['command'] = lambda: self.plot_batch(root)
        self.buttons['save_sum']['command'] = lambda: self.save_batch(root)
        self.buttons['save_ts']['command'] = lambda: self.save_time_series(root)

        # switch buttons to 'disabled' until something is selected
        self.buttons['remove']['state'] = DISABLED

        # bind listbox
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

    # -----------------------------------------------------
    # Callback functions
    # -----------------------------------------------------
    # enable/disable buttons upon listbox cursor select    
    def on_select(self, selection):
        if self.listbox.curselection():
            self.buttons['remove']['state'] = NORMAL
        else:
            self.buttons['remove']['state'] = DISABLED

    # --------------------------------------------------------
    # move channel data from channel listbox -> channel batch
    def add_items(self, root):
        batch_list = self.listbox.get(0, END)
        for_batch = Expresso.fetch_data_for_batch(root, 'channels')
        for_batch = sorted(for_batch, reverse=False)
        for channel in tuple(for_batch):
            if channel not in batch_list:
                self.listbox.insert(END, channel)

    # --------------------------------------------------------
    # remove selected channel data batch listbox
    def rm_items(self):
        selected = sorted(self.listbox.curselection(), reverse=True)
        for item in selected:
            self.listbox.delete(item)

    # --------------------------------------------------------
    # clear all data from batch listbox
    def clear_all(self):
        self.listbox.delete(0, END)

    # --------------------------------------------------------
    # plot summary of batch channel data
    def plot_batch(self, root):
        batch_list = self.listbox.get(0, END)
        comb_analysis_flag = root.comb_analysis_flag.get()
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return

        # get time info
        try:
            tmin = int(self.t_entries['t_min'].get())
            tmax = int(self.t_entries['t_max'].get())
            tbin = int(self.t_entries['t_bin'].get())
        except:
            tkMessageBox.showinfo(title='Error',
                                  message='Set time range and bin size')
            return

        batch_bout_analysis(batch_list, tmin, tmax, tbin, plotFlag=True, combAnalysisFlag=comb_analysis_flag)

    # --------------------------------------------------------
    # save summary of batch channel analysis
    def save_batch(self, root):
        batch_list = self.listbox.get(0, END)
        comb_analysis_flag = root.comb_analysis_flag.get()
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            try:
                tmin = int(self.t_entries['t_min'].get())
                tmax = int(self.t_entries['t_max'].get())
                tbin = int(self.t_entries['t_bin'].get())
            except:
                tkMessageBox.showinfo(title='Error',
                                      message='Set time range and bin size')
                return

            bouts_list, names, vols, consump, dur, latency = batch_bout_analysis(batch_list, tmin, tmax, tbin,
                                                                                 plotFlag=False,
                                                                                 combAnalysisFlag=comb_analysis_flag)

            # request save filename from user
            save_filename = tkFileDialog.asksaveasfilename(defaultextension=".xlsx")

            # check that we got a valid filename
            if not save_filename:
                print('Invalid save name')
                return

            # save batch channel summary
            save_batch_xlsx(save_filename, bouts_list, names, vols, consump, dur, latency)

            print('Completed saving {}'.format(save_filename))

    # --------------------------------------------------------
    # save times series analyses of batch channel data
    def save_time_series(self, root):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            save_dir = Expresso.get_dir(root)

            for entry in batch_list:

                dset, frames, t, dset_smooth, bouts, _ = \
                    Expresso.get_channel_data(root, entry)
                feeding_boolean = np.zeros([1, dset.size])
                for i in np.arange(bouts.shape[1]):
                    feeding_boolean[0, bouts[0, i]:bouts[1, i]] = 1
                row_mat = np.vstack((frames, t, dset, dset_smooth, feeding_boolean))
                row_mat = np.transpose(row_mat)

                filepath, filekeyname, groupkeyname = entry.split(', ', 2)
                dirpath, filename = os.path.split(filepath)
                save_name = filename[:-5] + "_" + filekeyname + "_" + groupkeyname + ".csv"
                save_path = os.path.join(save_dir, save_name)
                if sys.version_info[0] < 3:
                    out_path = open(save_path, mode='wb')
                else:
                    out_path = open(save_path, 'w', newline='')

                save_writer = csv.writer(out_path)

                save_writer.writerow(['Idx'] + ['Time [s]'] + \
                                     ['Data Raw [nL]'] + ['Data Smoothed [nL]'] + ['Feeding [bool]'])
                # cc = 1
                for row in row_mat:
                    # new_row = np.insert(row,0,cc)
                    save_writer.writerow(row)
                    # cc += 1

                out_path.close()


# =============================================================================
""" UI frame containing batch analysis for just video data  """


# =============================================================================
class BatchVidFrame(Frame):
    def __init__(self, parent, root, col=0, row=0):
        Frame.__init__(self, parent)

        # define names/labels for buttons + listboxes
        self.button_names = [['add', 'remove', 'clear_all'],
                             ['analyze', 'save_sum', 'save_ts'],
                             ['plot_heatmap', 'plot_cum_dist']]
        self.button_labels = [['Add Video(s) to Batch',
                               'Remove Selected',
                               'Clear All'],
                              ['Analyze/Save Video(s)',
                               'Save Video Batch Summary',
                               'Save Video Time Series'],
                              ['Plot Heatmap',
                               'Plot Cumulative Distance']]

        # generate basic panel layout                      
        buildBatchPanel(self, self.button_names, self.button_labels,
                        tboxFlag=False, row=row, col=col)

        # assign button callbacks
        self.buttons['add']['command'] = lambda: self.add_items(root)
        self.buttons['remove']['command'] = self.rm_items
        self.buttons['clear_all']['command'] = self.clear_all
        self.buttons['analyze']['command'] = self.analyze_batch
        self.buttons['save_sum']['command'] = self.save_batch
        self.buttons['save_ts']['command'] = self.save_time_series
        self.buttons['plot_heatmap']['command'] = self.plot_heatmap_batch
        self.buttons['plot_cum_dist']['command'] = self.plot_cum_dist_batch

        # switch buttons to 'disabled' until something is selected
        self.buttons['remove']['state'] = DISABLED

        # bind listbox
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

    # -----------------------------------------------------
    # Callback functions
    # -----------------------------------------------------
    # enable/disable buttons upon listbox cursor select    
    def on_select(self, selection):
        if self.listbox.curselection():
            self.buttons['remove']['state'] = NORMAL
        else:
            self.buttons['remove']['state'] = DISABLED

    # --------------------------------------------------------
    # move video data from video listbox -> video batch
    def add_items(self, root):
        batch_list = self.listbox.get(0, END)
        for_batch = Expresso.fetch_data_for_batch(root, 'videos')
        for_batch = sorted(for_batch, reverse=False)
        for vid in tuple(for_batch):
            if vid not in batch_list:
                self.listbox.insert(END, vid)

    # --------------------------------------------------------
    # remove selected video data batch listbox
    def rm_items(self):
        selected = sorted(self.listbox.curselection(), reverse=True)
        for item in selected:
            self.listbox.delete(item)

    # --------------------------------------------------------
    # clear all entries from batch video listbox
    def clear_all(self):
        self.listbox.delete(0, END)

    # --------------------------------------------------------------------
    # analyze and save tracking results for all videos in batch listbox
    def analyze_batch(self):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add video(s) to batch box for batch analysis')
            return
        else:
            # # make sure user wants to analyze data because it could take a while
            # yes_no_str = 'Do you want to continue batch tracking analysis?'
            # overWriteFlag = tkMessageBox.askyesno(title='Continue?', message=yes_no_str)
            for vid_file in batch_list:
                file_path, filename = os.path.split(vid_file)
                try:
                    visual_expresso_tracking_main(file_path, filename, DEBUG_BG_FLAG=False, DEBUG_CM_FLAG=False,
                                                  SAVE_DATA_FLAG=True, ELLIPSE_FIT_FLAG=False, PARAMS=trackingParams)
                    filename_prefix = os.path.splitext(filename)[0]
                    track_filename = filename_prefix + "_TRACKING.hdf5"
                    process_visual_expresso(file_path, track_filename,
                                            SAVE_DATA_FLAG=True, DEBUG_FLAG=False)
                except:
                    e = sys.exc_info()[0]
                    print('Error:')
                    print(e)

    # --------------------------------------------------------------------
    # save summary of all videos in batch listbox
    def save_batch(self):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add videos to batch box for batch analysis')
            return
        else:
            # request summary filename to save to
            xlsx_filename = tkFileDialog.asksaveasfilename(defaultextension=".xlsx",
                                                           title='Select save filename')
            # check that we got a valid filename
            if not xlsx_filename:
                print('Invalid save name')
                return

            # save tracking summary (XLSX)
            save_vid_summary(batch_list, xlsx_filename)

    # --------------------------------------------------------------------
    # save time series data for all videos in batch listbox
    def save_time_series(self):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add videos to batch box for batch analysis')
            return
        else:
            save_vid_time_series(batch_list)

    # --------------------------------------------------------------------
    # plot cumulative distance for all videos in batch listbox
    def plot_cum_dist_batch(self):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            batch_plot_cum_dist(batch_list)

    # --------------------------------------------------------------------
    # plot heatmap for all videos in batch listbox
    def plot_heatmap_batch(self):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            batch_plot_heatmap(batch_list)


# =============================================================================
""" UI frame containing batch analysis for combined feeding and video data  """


# =============================================================================
class BatchCombinedFrame(Frame):
    def __init__(self, parent, root, col=0, row=0):
        Frame.__init__(self, parent)

        # define names/labels for buttons + listboxes
        self.button_names = [['add', 'remove', 'clear_all'],
                             ['analyze', 'comb_data', 'save_sum', 'save_ts'],
                             ['plot_channel', 'plot_heatmap', 'plot_cum_dist', 'plot_xy', 'plot_dist_mag']]
        self.button_labels = [['Add Data to Batch',
                               'Remove Selected',
                               'Clear All'],
                              ['Analyze/Save Data',
                               'Combine Data Types',
                               'Save Combined Summary',
                               'Save Time Series'],
                              ['Plot Channel Analysis',
                               'Plot Heatmap',
                               'Plot Cumulative Distance',
                               'Plot Post-Meal XY',
                               'Plot Post-Meal Radial Dist.']]

        # generate basic panel layout                      
        buildBatchPanel(self, self.button_names, self.button_labels,
                        tboxFlag=True, row=row, col=col)

        # assign button callbacks
        self.buttons['add']['command'] = lambda: self.add_items(root)
        self.buttons['remove']['command'] = self.rm_items
        self.buttons['clear_all']['command'] = self.clear_all
        self.buttons['analyze']['command'] = lambda: self.analyze_batch(root)
        self.buttons['comb_data']['command'] = lambda: self.comb_data_batch(root)
        self.buttons['save_sum']['command'] = self.save_batch_comb
        self.buttons['save_ts']['command'] = self.save_time_series
        self.buttons['plot_channel']['command'] = self.plot_channel_batch
        self.buttons['plot_heatmap']['command'] = self.plot_heatmap_batch
        self.buttons['plot_cum_dist']['command'] = self.plot_cum_dist_batch
        self.buttons['plot_xy']['command'] = lambda: self.plot_post_meal_data_batch(root, varx='xcm_smooth',
                                                                                    vary='ycm_smooth',
                                                                                    data_name='XY',
                                                                                    x_label='X Position (cm)',
                                                                                    y_label='Y Position (cm)',
                                                                                    x_lim=[-0.4, 0.4],
                                                                                    y_lim=[-0.4, 0.4],
                                                                                    init_vals=[1, 0, 10],
                                                                                    axis_equal_flag=True,
                                                                                    axis_tight_flag=False)
        self.buttons['plot_dist_mag']['command'] = lambda: self.plot_post_meal_data_batch(root, varx='t',
                                                                                          vary='dist_mag',
                                                                                          data_name='Radial Dist.',
                                                                                          x_label='Time (s)',
                                                                                          y_label='Radial Dist. (cm)',
                                                                                          x_lim=[],
                                                                                          y_lim=[],
                                                                                          init_vals=[1, 0, 100],
                                                                                          axis_equal_flag=False,
                                                                                          axis_tight_flag=True)

        # switch buttons to 'disabled' until something is selected
        self.buttons['remove']['state'] = DISABLED

        # bind listbox
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

    # -----------------------------------------------------
    # Callback functions
    # -----------------------------------------------------
    # enable/disable buttons upon listbox cursor select    
    def on_select(self, selection):
        if self.listbox.curselection():
            self.buttons['remove']['state'] = NORMAL
        else:
            self.buttons['remove']['state'] = DISABLED

    # --------------------------------------------------------
    # move data (video or channel) -> combined batch
    def add_items(self, root):
        batch_list = self.listbox.get(0, END)
        # grab selections from both video and channel listboxes
        for_batch_ch = Expresso.fetch_data_for_batch(root, 'channels',
                                                     errorFlag=False)
        for_batch_vid = Expresso.fetch_data_for_batch(root, 'videos',
                                                      errorFlag=False)

        # convert both filetypes to common format
        ch_ent_basic = [channel2basic(c_fn) for c_fn in for_batch_ch]
        vid_ent_basic = [vid2basic(v_fn) for v_fn in for_batch_vid]

        # combine lists and remove duplicates
        union_set = set.union(set(vid_ent_basic), set(ch_ent_basic))
        union_list = list(union_set)
        for_batch = union_list

        # sort final list
        for_batch = sorted(for_batch, reverse=False)
        for ent in tuple(for_batch):
            if ent not in batch_list:
                self.listbox.insert(END, ent)

    # --------------------------------------------------------
    # remove selected data from combined batch listbox
    def rm_items(self):
        selected = sorted(self.listbox.curselection(), reverse=True)
        for item in selected:
            self.listbox.delete(item)

    # --------------------------------------------------------
    # clear all data from combined batch listbox
    def clear_all(self):
        self.listbox.delete(0, END)

    # ---------------------------------------------------------
    # perform analysis with combined video and channel data
    # ---------------------------------------------------------
    def analyze_batch(self, root):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            for data_file in batch_list:
                # perform tracking analysis
                file_path, filename_no_ext = os.path.split(data_file)
                vid_filename = filename_no_ext + '.avi'
                visual_expresso_tracking_main(file_path, vid_filename,
                                              DEBUG_BG_FLAG=False, DEBUG_CM_FLAG=False,
                                              SAVE_DATA_FLAG=True, ELLIPSE_FIT_FLAG=False,
                                              PARAMS=trackingParams)
                filename_prefix = os.path.splitext(vid_filename)[0]
                track_filename = filename_prefix + "_TRACKING.hdf5"
                flyTrackData = process_visual_expresso(file_path, track_filename,
                                                       SAVE_DATA_FLAG=True,
                                                       DEBUG_FLAG=False)

                # perform feeding analysis
                channel_entry = basic2channel(data_file)
                dset, frames, channel_t, dset_smooth, bouts, volumes = \
                    Expresso.get_channel_data(root, channel_entry,
                                              DEBUG_FLAG=False,
                                              combFlagArg=True)

                # merge data into one dict structure
                flyCombinedData = merge_v_expresso_data(dset, dset_smooth,
                                                        channel_t, frames, bouts,
                                                        volumes, flyTrackData)
                flyCombinedData_to_hdf5(flyCombinedData)

        self.comb_file_list = batch_list

    #
    # -----------------------------------------------------
    # save batch video summary
    # -----------------------------------------------------
    def save_batch_comb(self):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            # request filename for summary
            xlsx_filename = tkFileDialog.asksaveasfilename(defaultextension=".xlsx",
                                                           title='Select save filename for COMBINED summary')
            # check that we got a valid filename
            if not xlsx_filename:
                print('Invalid save name')
                return

            # save video summary (XLSX)
            save_comb_summary(batch_list, xlsx_filename)

            # also save to hdf5
            save_filename = os.path.splitext(xlsx_filename)[0]
            hdf5_filename = save_filename + '.hdf5'
            save_comb_summary_hdf5(batch_list, hdf5_filename)

    # ---------------------------------------------------------------------
    # save channel and video data as one csv file (at each time point)
    # ---------------------------------------------------------------------
    def save_time_series(self):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            data_suffix = '_COMBINED_DATA.hdf5'
            data_filenames = [ent + data_suffix for ent in batch_list]
            initdir, _ = os.path.split(data_filenames[0])
            savedir = tkFileDialog.askdirectory(initialdir=initdir)
            save_comb_time_series(data_filenames, savedir)

    # ---------------------------------------------------------------------
    # for each list element, combine channel and video data into one file
    # ---------------------------------------------------------------------
    def comb_data_batch(self, root):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            for data_file in batch_list:

                # LOAD tracking analysis
                file_path, filename_no_ext = os.path.split(data_file)
                tracking_filename = filename_no_ext + "_TRACKING_PROCESSED.hdf5"
                track_path_full = os.path.join(file_path, tracking_filename)
                if os.path.exists(track_path_full):
                    flyTrackData = hdf5_to_flyTrackData(file_path,
                                                        tracking_filename)
                else:
                    print('Cannot find file: {}'.format(track_path_full))
                    continue

                # perform feeding analysis
                channel_entry = basic2channel(data_file)
                dset, frames, channel_t, dset_smooth, bouts, volumes = Expresso.get_channel_data(root, channel_entry,
                                                                                                 DEBUG_FLAG=False,
                                                                                                 combFlagArg=True)

                # merge data into one dict structure
                flyCombinedData = merge_v_expresso_data(dset, dset_smooth,
                                                        channel_t, frames, bouts,
                                                        volumes, flyTrackData)
                flyCombinedData_to_hdf5(flyCombinedData)

    # ---------------------------------------------------------------------
    # plot channel (feeding) data summary
    # ---------------------------------------------------------------------
    def plot_channel_batch(self):
        batch_list = self.listbox.get(0, END)
        # comb_analysis_flag = root.comb_analysis_flag.get()
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            try:
                tmin = int(self.t_entries['t_min'].get())
                tmax = int(self.t_entries['t_max'].get())
                tbin = int(self.t_entries['t_bin'].get())
            except (NameError, ValueError):
                tkMessageBox.showinfo(title='Error',
                                      message='Set time range and bin size')
                return

            # run and save feeding analysis (a little redundant)
            batch_list_ch = [basic2channel(ent) for ent in batch_list]
            batch_bout_analysis(batch_list_ch, tmin, tmax, tbin, plotFlag=True, combAnalysisFlag=True)

    # ---------------------------------------------------------------------
    # plot cumulative distance
    # ---------------------------------------------------------------------                               
    def plot_cum_dist_batch(self):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            # try to grab time limits
            try:
                tmin = int(self.t_entries['t_min'].get())
                tmax = int(self.t_entries['t_max'].get())
                # tbin = int(self.t_entries['t_bin'].get())
            except (NameError, ValueError):
                tkMessageBox.showinfo(title='Error',
                                      message='Set time range and bin size')
                return
            # if we have time range info, make plots    
            batch_list_vid = [basic2vid(ent) for ent in batch_list]
            batch_plot_cum_dist(batch_list_vid, t_lim=[tmin, tmax])

    # ---------------------------------------------------------------------
    # plot position heatmap
    # ---------------------------------------------------------------------   
    def plot_heatmap_batch(self):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Add data to batch box for batch analysis')
            return
        else:
            # try to grab time limits
            try:
                tmin = int(self.t_entries['t_min'].get())
                tmax = int(self.t_entries['t_max'].get())
                # tbin = int(self.t_entries['t_bin'].get())
            except (NameError, ValueError):
                tkMessageBox.showinfo(title='Error',
                                      message='Set time range and bin size')
                return
            # if we have time range info, make plots
            batch_list_vid = [basic2vid(ent) for ent in batch_list]
            batch_plot_heatmap(batch_list_vid, t_lim=[tmin, tmax], SAVE_FLAG=False)

    # ---------------------------------------------------------------------
    # plot XY trajectory for flies after a given meal
    # ---------------------------------------------------------------------
    def plot_post_meal_data_batch(self, root, varx='xcm_smooth', vary='ycm_smooth', data_name='XY',
                                  x_label='X Position (cm)', y_label='Y Position (cm)', x_lim=[-0.4, 0.4],
                                  y_lim=[-0.4, 0.4], init_vals=[1, 0, 10], axis_equal_flag=True,
                                  axis_tight_flag=False):
        batch_list = self.listbox.get(0, END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error', message='Add data to batch box for batch analysis')
            return
        else:
            # --------------------------------------------------------------------------
            # get plot options from user using pop-up window with entry boxes
            options_entry_list = ['Meal Number', 'Time before meal end (s)', 'Time after meal end (s)']
            options_init_vals = init_vals
            options_chkbtn_list = ['Save data output?']
            options_popup = myEntryOptions(root.master, root, entry_list=options_entry_list,
                                           title_str='Post-meal {} Plot Options'.format(data_name),
                                           init_vals=options_init_vals,
                                           chkbtn_list=options_chkbtn_list)

            # wait for user input before preceding
            options_popup.wait_window()

            # ------------------------------------------------------------------------------------
            # extract param values from pop-up window (which should have been sent to "root")
            try:
                meal_num = int(root.popup_entry_values[options_entry_list[0]])
                window_left_sec = float(root.popup_entry_values[options_entry_list[1]])
                window_right_sec = float(root.popup_entry_values[options_entry_list[2]])
                options_save_flag = root.popup_chkbtn_values[options_chkbtn_list[0]]
            except (AttributeError, KeyError):
                tkMessageBox.showinfo(title='Error', message='No values selected for plot params')
                return

            # --------------------------------------------------------------------------------------
            # if we're saving results, request filename from user
            if options_save_flag:
                prompt_str = "Select save filename for meal-aligned radial distance"
                xlsx_filename = tkFileDialog.asksaveasfilename(defaultextension=".xlsx", title=prompt_str)

                # check that we got a valid filename
                if not xlsx_filename:
                    print('Invalid save name')
                    options_save_flag = False
            else:
                xlsx_filename = None

            # --------------------------------------------------------------------------------------
            # make meal indexing more intuitive
            if meal_num > 0:
                meal_num = meal_num - 1

            # run plot function
            fig, ax = plot_bout_aligned_var(batch_list, varx=varx, vary=vary, window_left_sec=window_left_sec,
                                            window_right_sec=window_right_sec, meal_num=meal_num,
                                            save_flag=options_save_flag, save_filename=xlsx_filename,
                                            varx_name=x_label, vary_name=y_label)

            # -----------------------------------
            # modify axis
            # square axis (good for XY)?
            if axis_equal_flag:
                ax.set_aspect('equal', 'box')

            # tight axes?
            if axis_tight_flag:
                ax.autoscale(enable=True, axis='both', tight=True)

            # apply use-supplied x and y limits if provided
            if len(x_lim) > 1:
                ax.set_xlim(x_lim)
            if len(y_lim) > 1:
                ax.set_ylim(y_lim)

            # try to keep labels from being cut off
            fig.tight_layout()

# =============================================================================
# Main class for GUI
# =============================================================================
class Expresso:
    """The GUI and functions."""

    def __init__(self, master):
        # Tk.__init__(self)
        self.master = master

        # style
        # ???

        # initialize important fields for retaining where we are in data space

        datapath = []
        self.datadir_curr = datapath

        filename = []
        self.filename_curr = filename

        filekeyname = []
        self.filekeyname_curr = filekeyname

        grpnum = []
        self.grpnum_curr = grpnum

        grp = []
        self.grp_curr = grp

        # --------------------------------
        # debugging boolean variables
        # --------------------------------
        self.debug_tracking = BooleanVar()
        self.debug_tracking.set(False)

        self.debug_bout = BooleanVar()
        self.debug_bout.set(False)

        self.save_all_plots = BooleanVar()
        self.save_all_plots.set(False)

        # --------------------------------------
        # combined analysis boolean variable(s)
        # --------------------------------------
        self.comb_analysis_flag = BooleanVar()
        self.comb_analysis_flag.set(True)

        # --------------------------------------
        # run gui presets. may be unecessary
        # --------------------------------------
        self.init_gui()

        # ----------------------------------------------
        # initialize instances of frames defined above
        # ----------------------------------------------
        # data navigation and single-file analysis frames:
        self.dirframe = DirectoryFrame(self, col=0, row=0)
        self.fdata_frame = FileDataFrame(self, col=0, row=1)
        self.xpdata_frame = XPDataFrame(self, col=0, row=2)
        self.channeldata_frame = ChannelDataFrame(self, col=0, row=3)
        self.viddata_frame = VideoDataFrame(self, col=0, row=4)

        # ttk notebook for batch analysis:
        nb_r = 1  # 1
        nb_c = 3  # 3
        nb_rspan = 4
        nb_cspan = 3

        # configure weights for notebook grid area
        for rowCurr in range(nb_r, nb_r + nb_rspan):
            Grid.rowconfigure(self.master, rowCurr, weight=1)
        for colCurr in range(nb_c, nb_c + nb_cspan):
            Grid.columnconfigure(self.master, colCurr, weight=1)

        self.batch_nb = Notebook(self.master)
        self.batch_nb.grid(row=nb_r, column=nb_c, rowspan=nb_rspan,
                           columnspan=nb_cspan, sticky=NSEW)

        # batch channel analysis
        self.batchdata_frame = BatchChannelFrame(self.batch_nb, self)
        self.batchvid_frame = BatchVidFrame(self.batch_nb, self)
        self.batchcomb_frame = BatchCombinedFrame(self.batch_nb, self)

        self.batch_nb.add(self.batchdata_frame, text='Batch Channel Analysis',
                          sticky='NSEW')
        self.batch_nb.add(self.batchvid_frame, text='Batch Video Analysis',
                          sticky='NSEW')
        self.batch_nb.add(self.batchcomb_frame, text='Batch Combined Analysis',
                          sticky='NSEW')

        # insert logo image!
        # get path to image (should be in src directory)
        self.img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'expresso_alpha.gif')

        # configure weights for image grid area
        Grid.rowconfigure(self.master, 0, weight=1)
        # Grid.columnconfigure(self.master,)
        # im = Image.open(im_path)
        # ph = ImageTk.PhotoImage(im)
        self.img = PhotoImage(file=self.img_path, master=self.master)
        self.im_label = Label(self.master, image=self.img,
                              background=guiParams['bgcolor'])
        self.im_label.img = self.img
        self.im_label.grid(column=3, row=0, columnspan=nb_cspan, padx=10,
                           pady=2, sticky=NSEW)

        # ----------------------------------------------------
        # populate directory list with initial directories
        for init_dir in initDirectories:
            if os.path.exists(init_dir) and os.path.isdir(init_dir):
                init_dir_norm = os.path.normpath(init_dir)
                self.dirframe.listbox.insert(END, init_dir_norm)

        # -----------------------------------------------------
        # define variable to store info on user-searched paths
        # (want to keep it updated so we don't always default to some random spot)
        # start_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # self.search_path = start_path

        # -----------------------------
        # define quit command
        self.master.protocol("WM_DELETE_WINDOW", self.on_quit)

    # ===================================================================
    def on_quit(self):
        """Exits program."""
        if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
            self.master.destroy()
            self.master.quit()

    # ===================================================================
    def on_open_crop_gui(self):
        """Exits program."""
        if tkMessageBox.askokcancel("Open crop and save GUI",
                                    "Do you want to quit and open cropping GUI?"):
            self.master.destroy()
            self.master.quit()
            # run script
            os.system("python crop_and_save_gui.py")

    # ===================================================================
    def on_open_pp_gui(self):
        """Exits program."""
        if tkMessageBox.askokcancel("Open post-processing GUI",
                                    "Do you want to quit and open post processing GUI?"):
            self.master.destroy()
            self.master.quit()
            # run script
            os.system("python post_processing_gui.py")

    # ===================================================================
    def toggle_track_debug(self):
        """Turns on or off the tracking debug flags"""
        curr_val = self.debug_tracking.get()
        if curr_val:
            print('In tracking debug mode')
        else:
            print('NOT in tracking debug mode')

    # ===================================================================
    def toggle_bout_debug(self):
        """Turns on or off the bout debug flags"""
        curr_val = self.debug_bout.get()
        if curr_val:
            print('In feeding bout detection debug mode')
        else:
            print('NOT in feeding bout detection debug mode')

    # ===================================================================
    def toggle_save_all(self):
        """Turns on or off the save plots flags"""
        curr_val = self.save_all_plots.get()
        if curr_val:
            print('In save all plots mode')
        else:
            print('NOT in save all plots mode')

    # ===================================================================
    def toggle_comb_analysis(self):
        """Turns on or off the synchronized selection flags"""
        curr_val = self.comb_analysis_flag.get()
        if curr_val:
            print('Using tracking data to validate meal bout detection')
        else:
            print('NOT using tracking data to validate meal bout detection')

    # ===================================================================
    def sync_listboxes_intersect(self):
        """Repopulate the video and channel listboxes so that they match"""
        N_ch_entries = self.channeldata_frame.listbox.size()
        channel_entries = self.channeldata_frame.listbox.get(0, N_ch_entries)
        N_vid_entries = self.viddata_frame.listbox.size()
        vid_entries = self.viddata_frame.listbox.get(0, N_vid_entries)

        # reformat entry types to facilitate comparison
        vid_entries_refrm = [vid2basic(v_en) for v_en in vid_entries]
        channel_entries_refrm = [channel2basic(ch) for ch in channel_entries]

        # find set intersection of listbox entries
        vid_entries_set = set(vid_entries_refrm)
        channel_entries_set = set(channel_entries_refrm)
        entry_intersect = vid_entries_set.intersection(channel_entries_set)
        entry_intersect = sorted(list(entry_intersect), reverse=False)

        # delete old entries
        self.channeldata_frame.listbox.delete(0, N_ch_entries)
        self.viddata_frame.listbox.delete(0, N_vid_entries)

        # switch back to format for the listboxes and add 
        for ent in entry_intersect:
            ent_vid = basic2vid(ent)
            self.viddata_frame.listbox.insert(END, ent_vid)

            ent_ch = basic2channel(ent)
            self.channeldata_frame.listbox.insert(END, ent_ch)

    # ===================================================================
    def sync_listboxes_union(self):
        """Repopulate the video and channel listboxes so that they match"""
        N_ch_entries = self.channeldata_frame.listbox.size()
        channel_entries = self.channeldata_frame.listbox.get(0, N_ch_entries)
        N_vid_entries = self.viddata_frame.listbox.size()
        vid_entries = self.viddata_frame.listbox.get(0, N_vid_entries)

        # reformat entry types to facilitate comparison
        vid_entries_refrm = [vid2basic(v_en) for v_en in vid_entries]
        channel_entries_refrm = [channel2basic(ch) for ch in channel_entries]

        # find set intersection of listbox entries
        vid_entries_set = set(vid_entries_refrm)
        channel_entries_set = set(channel_entries_refrm)
        entry_union = vid_entries_set.union(channel_entries_set)
        entry_union = sorted(list(entry_union), reverse=False)

        # delete old entries
        self.channeldata_frame.listbox.delete(0, N_ch_entries)
        self.viddata_frame.listbox.delete(0, N_vid_entries)

        # switch back to format for the listboxes and add 
        for ent in entry_union:
            ent_vid = basic2vid(ent)
            self.viddata_frame.listbox.insert(END, ent_vid)

            ent_ch = basic2channel(ent)
            self.channeldata_frame.listbox.insert(END, ent_ch)

    # ===================================================================
    def sync_select(self):
        """Synchronizes selection for channel and video listboxes"""
        # selected_vid_ind = self.viddata_frame.selection_ind
        # selected_channel_ind = self.channeldata_frame.selection_ind
        selected_vid_ind = self.viddata_frame.listbox.curselection()
        selected_channel_ind = self.channeldata_frame.listbox.curselection()

        if (len(selected_vid_ind) > 0) and (len(selected_channel_ind) < 1):
            new_channel_ind = []
            vid_entries = [self.viddata_frame.listbox.get(ind) for ind in \
                           selected_vid_ind]
            basic_entries = [vid2basic(v_ent) for v_ent in vid_entries]
            for b_ent in basic_entries:
                ch_ent = basic2channel(b_ent)
                if ch_ent not in self.channeldata_frame.listbox.get(0, END):
                    self.channeldata_frame.listbox.insert(END, ch_ent)
                ch_ind = self.channeldata_frame.listbox.get(0, END).index(ch_ent)
                new_channel_ind.append(ch_ind)
                # self.channeldata_frame.listbox.selection_set(ch_ind)
            # self.channeldata_frame.selection_ind = new_channel_ind

        elif (len(selected_vid_ind) < 1) and (len(selected_channel_ind) > 0):
            new_vid_ind = []
            ch_entries = [self.channeldata_frame.listbox.get(ind) for \
                          ind in selected_channel_ind]
            basic_entries = [channel2basic(ch_ent) for ch_ent in ch_entries]
            for b_ent in basic_entries:
                vid_ent = basic2vid(b_ent)
                if vid_ent not in self.viddata_frame.listbox.get(0, END):
                    self.viddata_frame.listbox.insert(END, vid_ent)
                vid_ind = self.viddata_frame.listbox.get(0, END).index(vid_ent)
                new_vid_ind.append(vid_ind)
                # self.viddata_frame.listbox.selection_set(vid_ind)
            # self.viddata_frame.selection_ind = new_vid_ind

        else:
            print('Error--need to fix this')

    # ===================================================================
    def make_topmost(self):
        """Makes this window the topmost window"""
        self.master.lift()
        self.master.attributes("-topmost", 1)
        self.master.attributes("-topmost", 0)

    # ===================================================================

    def init_gui(self):
        """Label for GUI"""

        self.master.title('Expresso Analysis Toolbox (EAT)')

        """ Menu bar """
        self.master.option_add('*tearOff', 'FALSE')
        self.master.menubar = Menu(self.master)

        # file menu
        self.master.menu_file = Menu(self.master.menubar)
        self.master.menu_file.add_command(label='Open PRE-process GUI',
                                          command=self.on_open_crop_gui)
        self.master.menu_file.add_command(label='Open POST-process GUI',
                                          command=self.on_open_pp_gui)
        self.master.menu_file.add_command(label='Exit', command=self.on_quit)

        # debug menu
        self.master.menu_debug = Menu(self.master.menubar)
        self.master.menu_debug.add_checkbutton(label='Save All Plots [toggle]',
                                               variable=self.save_all_plots,
                                               command=self.toggle_save_all)
        self.master.menu_debug.add_checkbutton(label='Bout Detection Debug [toggle]',
                                               variable=self.debug_bout,
                                               command=self.toggle_bout_debug)
        self.master.menu_debug.add_checkbutton(label='Tracking Debug [toggle]',
                                               variable=self.debug_tracking,
                                               command=self.toggle_track_debug)

        # combined analysis
        self.master.menu_comb = Menu(self.master.menubar)
        self.master.menu_comb.add_command(label='Synchronize data lists (intersection)',
                                          command=self.sync_listboxes_intersect)
        self.master.menu_comb.add_command(label='Synchronize data lists (union)',
                                          command=self.sync_listboxes_union)
        self.master.menu_comb.add_command(label='Synchronize selection',
                                          command=self.sync_select)
        self.master.menu_comb.add_checkbutton(label='Combine Data Types [toggle]',
                                              variable=self.comb_analysis_flag,
                                              command=self.toggle_comb_analysis)

        # add these bits to menu bar
        self.master.menubar.add_cascade(menu=self.master.menu_file, label='File')
        self.master.menubar.add_cascade(menu=self.master.menu_debug,
                                        label='Debugging Options')
        self.master.menubar.add_cascade(menu=self.master.menu_comb,
                                        label='Combined Analysis Tools')

        self.master.config(menu=self.master.menubar)

        # self.master.config(background='white')

    # ===================================================================
    @staticmethod
    def get_dir(self):
        """ Method to return the directory selected by the user which should
            be scanned by the application. """

        # get user specified directory and normalize path
        seldir = tkFileDialog.askdirectory()
        if seldir:
            seldir = os.path.abspath(seldir)
            self.datadir_curr = seldir
            return seldir

    # ===================================================================
    # perform checks to make sure a given file in search directory is valid
    # (i.e. should be added to appropriate listbox). Also returns standardized 
    # form for filename
    def is_valid_dir_file(self, f, file_type):
        # ----------------------------------------------
        # tests if selected file type is channel data
        if file_type == 'channel':
            valid_ext = (".hdf5")
            invalid_end = ('VID_INFO.hdf5', 'TRACKING.hdf5',
                           'TRACKING_PROCESSED.hdf5', 'COMBINED_DATA.hdf5')
            is_valid = f.endswith(valid_ext) and not f.endswith(invalid_end)

            # for channel data files, format should already be good
            f_out = f

        # ----------------------------------------------
        # tests if selected data is a video file
        elif file_type == 'video':
            valid_ext = (".avi", ".mov", ".mp4", ".mpg", ".mpeg", ".rm",
                         ".swf", ".vob", ".wmv")
            processed_sfx = ("_TRACKING_PROCESSED.hdf5", "_COMBINED_DATA.hdf5")
            is_valid = f.endswith(valid_ext) and ('channel' in f) and ('XP' in f)
            processed_chk = any(sfx in f for sfx in processed_sfx)
            is_valid = is_valid or processed_chk

            # since we're allowing analyzed hdf5 files as well, need to make 
            # sure file formats are all the same
            if is_valid and processed_chk:
                f_split = f.split('_')
                f_out = '_'.join(f_split[:-2])
                f_out = f_out + ".avi"
            else:
                f_out = f
        # -------------------------------------------------------------------
        # file should be either video or channel -- otherwise return nothing     
        else:
            print('Error: invalid file_type')
            is_valid = False
            f_out = None

        return is_valid, f_out

    # ===================================================================
    @staticmethod
    def scan_dirs(self, fileType):
        # initialize list of files to send to file listbox
        files = []

        # current directory listbox contents
        temp_dirlist = list(self.dirframe.listbox.get(0, END))
        # selected directories
        selected_ind = sorted(self.dirframe.listbox.curselection(),
                              reverse=False)

        # throw exception if no directory is selected
        if len(selected_ind) < 1:
            tkMessageBox.showinfo(title='Error',
                                  message='Please select directory' +
                                          'from which to grab hdf5 files')
            return files

        # search each selected directory and add valid files to list
        for ind in selected_ind:
            temp_dir = temp_dirlist[ind]
            for f in os.listdir(temp_dir):
                # check for validity and generate standardized file format
                validFlag, f_new = self.is_valid_dir_file(f, fileType)
                # if valid, add file
                if validFlag:
                    files.append(os.path.join(temp_dir, f_new))

        # make sure we don't pull duplicates
        files = list(set(files))

        # normalize paths to current OS
        files = [os.path.normpath(f) for f in files]

        # sort files
        files.sort()

        # save current directory to frame structure      
        self.datadir_curr = temp_dir

        # return files to populate listbox or let user know if they don't exist
        if len(files) > 0:
            return files
        else:
            tkMessageBox.showinfo(title='Error',
                                  message='No {} files found.'.format(fileType))
            files = []
            return files

    # ===================================================================
    @staticmethod
    def unpack_files(self):
        selected_ind = self.fdata_frame.listbox.curselection()
        # print(selected_ind)
        selected = []
        for ind in selected_ind:
            selected.append(self.fdata_frame.listbox.get(ind))

        # temp_dirlist = list(self.dirframe.dirlist.get(0, END))
        # for dir in temp_dirlist:
        fileKeyNames = []
        for filename in selected:
            # filename = os.path.join(dir,selected[0])
            # print(filename)
            if os.path.isfile(filename):
                self.filename_curr = filename
                f = h5py.File(filename, 'r')
                for key in list(f.keys()):
                    if key.startswith('XP'):
                        fileKeyNames.append(filename + ", " + key)

        return fileKeyNames

    # ===================================================================
    @staticmethod
    def unpack_xp(self):
        selected_ind = self.xpdata_frame.listbox.curselection()
        groupKeyNames = []
        for ind in selected_ind:
            xp_entry = self.xpdata_frame.listbox.get(ind)
            filename, filekeyname = xp_entry.split(', ', 1)
            f = h5py.File(filename, 'r')
            # fileKeyNames = list(f.keys())
            grp = f[filekeyname]
            for key in list(grp.keys()):
                dset, _ = load_hdf5(filename, filekeyname, key)
                dset_check = (dset != -1)
                if np.sum(dset_check) > 0:
                    groupKeyNames.append(filename + ', ' + filekeyname +
                                         ', ' + key)
        return groupKeyNames

    # ===================================================================
    # @staticmethod
    def get_channel_data(self, channel_entry, DEBUG_FLAG=False, combFlagArg=False):
        filename, filekeyname, groupkeyname = channel_entry.split(', ', 2)
        comb_analysis_flag = self.comb_analysis_flag.get()
        comb_analysis_flag = comb_analysis_flag or combFlagArg

        # load data        
        dset, t = load_hdf5(filename, filekeyname, groupkeyname)

        bad_data_flag, dset, t, frames = check_data_set(dset, t)

        if not bad_data_flag:
            if comb_analysis_flag:
                dset_smooth, bouts, volumes = bout_analysis_wTracking(filename, filekeyname, groupkeyname,
                                                                      debugBoutFlag=DEBUG_FLAG)
            else:
                dset_smooth, bouts, volumes, _ = bout_analysis(dset, frames, debug_mode=DEBUG_FLAG)
        else:
            dset_smooth = np.array([])
            bouts = np.array([])
            volumes = np.array([])
            print('Problem with loading data set--invalid name')

        return dset, frames, t, dset_smooth, bouts, volumes

    # ===================================================================
    @staticmethod
    def fetch_data_for_batch(self, listboxSourceType, errorFlag=True):
        # initialize list for sending to batch frame
        for_batch = []

        # choose appropriate box
        if listboxSourceType == 'channels':
            listbox_source = self.channeldata_frame.listbox
        elif listboxSourceType == 'videos':
            listbox_source = self.viddata_frame.listbox
        else:
            print('Error: invalid source selection')
            return

        selected_ind = listbox_source.curselection()

        if errorFlag and (len(selected_ind) < 1):
            err_str = 'Please select {} to move to batch'.format(listboxSourceType)
            tkMessageBox.showinfo(title='Error', message=err_str)
            return for_batch

        for ind in selected_ind:
            for_batch.append(listbox_source.get(ind))

        return for_batch


# def main():
#    root = Tk()
#    root.geometry("300x280+300+300")
#    app = Expresso(root)
#    root.mainloop()
# root = Toplevel()
# root.pack(fill="both", expand=True)
# root.grid_rowconfigure(0, weight=1)
# root.grid_columnconfigure(0, weight=1)

""" Run main loop """
if __name__ == '__main__':
    if TKDND_FLAG:
        root = TkinterDnD.Tk()
    else:
        root = Tk()
    Expresso(root)
    root.mainloop()
    # main()
