# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 17:11:07 2016

@author: Fruit Flies
"""

import matplotlib
matplotlib.use('TkAgg')

import os 
import sys 

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

import h5py

import numpy as np
from scipy import interpolate

#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, MultiCursor
#from matplotlib.figure import Figure

from load_hdf5_data import load_hdf5
from bout_analysis_func import check_data_set, plot_channel_bouts, bout_analysis
from batch_bout_analysis_func import batch_bout_analysis, save_batch_xlsx
from v_expresso_gui_params import (initDirectories, guiParams, trackingParams)
from bout_and_vid_analysis import (channel2basic, vid2basic, basic2channel, 
                                   basic2vid, flyCombinedData_to_hdf5,
                                   save_comb_time_series, hdf5_to_flyCombinedData,
                                   bout_analysis_wTracking, merge_v_expresso_data,
                                   plot_bout_aligned_var)
from v_expresso_image_lib import (visual_expresso_main, 
                                    process_visual_expresso, 
                                    plot_body_cm, plot_body_vel, 
                                    plot_body_angle, plot_moving_v_still, 
                                    plot_cum_dist, hdf5_to_flyTrackData,
                                    save_vid_time_series, save_vid_summary, 
                                    batch_plot_cum_dist, batch_plot_heatmap)

#from PIL import ImageTk, Image
import csv

# allows drag and drop functionality. if you don't want this, or are having 
#  trouble with the TkDnD installation, set to false.
TKDND_FLAG = True
if TKDND_FLAG:
    from TkinterDnD2 import *
#------------------------------------------------------------------------------

class DirectoryFrame(Frame):
    """ Top UI frame containing the list of directories to be scanned. """

    def __init__(self, parent, col=0, row=0, filedir=None):
        
        Frame.__init__(self, parent.master)
        
        self.btnframe = Frame(parent.master)
        self.btnframe.grid(column=col, row=row, sticky=(N, S, E, W))
        #self.btnframe.config(bg='white')
        
        self.lib_label = Label(self.btnframe, text='Directory list:',
                               font=guiParams['labelfontstr'])
                               #background=guiParams['bgcolor'])
                               #foreground=guiParams['textcolor'], 
                               
        self.lib_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)
        #self.btnframe.config(background=guiParams['bgcolor'])
        
        self.lib_addbutton =Button(self.btnframe, text='Add Directory',
                                   command= lambda: self.add_library(parent))
        self.lib_addbutton.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)
        
        self.lib_delbutton = Button(self.btnframe, text='Remove Directory',
                                  command=self.rm_library, state=DISABLED)
                                   
        self.lib_delbutton.grid(column=col, row=row+2, padx=10, pady=2,
                                sticky=NW)
        
        self.lib_clearbutton = Button(self.btnframe, text='Clear All',
                                  command=self.clear_library)
                                   
        self.lib_clearbutton.grid(column=col, row=row+3, padx=10, pady=2,
                                sticky=NW)                        

        
         # placement
        self.dirlistframe = Frame(parent.master)
        self.dirlistframe.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        # listbox containing all selected additional directories to scan
        self.dirlist = Listbox(self.dirlistframe, width=64, height=8,
                               selectmode=EXTENDED, exportselection=False)
                               #foreground=guiParams['textcolor'], 
                               #background=guiParams['listbgcolor'])
        
        # set as a target for drag and drop
        if TKDND_FLAG:                        
            self.dirlist.drop_target_register(DND_FILES, DND_TEXT)
        
        # set binding functions
        self.dirlist.bind('<<ListboxSelect>>', self.on_select)
        
        if TKDND_FLAG:   
            self.dirlist.dnd_bind('<<DropEnter>>', Expresso.drop_enter)
            self.dirlist.dnd_bind('<<DropPosition>>', Expresso.drop_position)
            self.dirlist.dnd_bind('<<DropLeave>>', Expresso.drop_leave)
            self.dirlist.dnd_bind('<<Drop>>', Expresso.dir_drop)
            self.dirlist.dnd_bind('<<Drop:DND_Files>>', Expresso.dir_drop)
            self.dirlist.dnd_bind('<<Drop:DND_Text>>', Expresso.dir_drop)
            
            self.dirlist.drag_source_register(1, DND_TEXT, DND_FILES)
            #text.drag_source_register(3, DND_TEXT)
        
            self.dirlist.dnd_bind('<<DragInitCmd>>', Expresso.drag_init_listbox)
            self.dirlist.dnd_bind('<<DragEndCmd>>', Expresso.drag_end)
        
        # scroll bars
        self.hscroll = Scrollbar(self.dirlistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.dirlistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.dirlist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        self.dirlist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.dirlist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.dirlist.yview)
        
       

       
    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.dirlist.curselection():
            self.lib_delbutton.configure(state=NORMAL)
        else:
            self.lib_delbutton.configure(state=DISABLED)

    def add_library(self,parent):
        """ Insert every selected directory chosen from the dialog.
            Prevent duplicate directories by checking existing items. """

        dirlist = self.dirlist.get(0, END)
        newdir = Expresso.get_dir(parent)
        if newdir not in dirlist:
            self.dirlist.insert(END, newdir)

    def rm_library(self):
        """ Remove selected items from listbox when button in remove mode. """

        # Reverse sort the selected indexes to ensure all items are removed
        selected = sorted(self.dirlist.curselection(), reverse=True)
        for item in selected:
            self.dirlist.delete(item)
    
    def clear_library(self):
        """ Remove all items from listbox when button pressed """
        self.dirlist.delete(0,END)        

#------------------------------------------------------------------------------

class FileDataFrame(Frame):
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)
        
        self.btnframe = Frame(parent.master)
        self.btnframe.grid(column=col, row=row, sticky=NW)
        
        self.list_label = Label(self.btnframe, text='Data files:', 
                                font = guiParams['labelfontstr'])
        self.list_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)
        
        # button used to initiate the scan of the above directories
        self.scan_btn =Button(self.btnframe, text='Get HDF5 Files',
                                        command= lambda: self.add_files(parent))
        #self.scan_btn['state'] = 'disabled'                                
        self.scan_btn.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Files')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_files
        self.remove_button.grid(column=col, row=row+2, padx=10, pady=2,
                                sticky=NW)
                                
        self.clear_button = Button(self.btnframe, text='Clear All')
        self.clear_button['command'] = self.clear_files
        self.clear_button.grid(column=col, row=row+3, padx=10, pady=2,
                                sticky=NW)                        

        

        self.filelistframe = Frame(parent.master) 
        self.filelistframe.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.filelist = Listbox(self.filelistframe,  width=64, height=8, 
                                selectmode=EXTENDED)
        
        #now make the Listbox and Text drop targets
        if TKDND_FLAG:   
            self.filelist.drop_target_register(DND_FILES, DND_TEXT)
        
        self.filelist.bind('<<ListboxSelect>>', self.on_select)
        
        if TKDND_FLAG:
            self.filelist.dnd_bind('<<DropEnter>>', Expresso.drop_enter)
            self.filelist.dnd_bind('<<DropPosition>>', Expresso.drop_position)
            self.filelist.dnd_bind('<<DropLeave>>', Expresso.drop_leave)
            self.filelist.dnd_bind('<<Drop>>', Expresso.file_drop)
            self.filelist.dnd_bind('<<Drop:DND_Files>>', Expresso.file_drop)
            self.filelist.dnd_bind('<<Drop:DND_Text>>', Expresso.file_drop)
            
            self.filelist.drag_source_register(1, DND_TEXT, DND_FILES)
            #text.drag_source_register(3, DND_TEXT)
        
            self.filelist.dnd_bind('<<DragInitCmd>>', Expresso.drag_init_listbox)
            self.filelist.dnd_bind('<<DragEndCmd>>', Expresso.drag_end)
            #text.dnd_bind('<<DragInitCmd>>', drag_init_text)    
    
    
        #self.filelist.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        # scroll bars
        self.hscroll = Scrollbar(self.filelistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.filelistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.filelist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        self.filelist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.filelist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.filelist.yview)


    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.filelist.curselection():
            self.remove_button.configure(state=NORMAL)
        else:
            self.remove_button.configure(state=DISABLED)
            
    def add_files(self,parent):
        newfiles = Expresso.scan_dirs(parent)
        file_list = self.filelist.get(0,END)
        if len(newfiles) > 0:
            #file_list = self.filelist.get(0,END)
            #Expresso.clear_xplist(parent)
            #Expresso.clear_channellist(parent)
            for file in tuple(newfiles):
                if file not in file_list:
                    self.filelist.insert(END,file)
    
    def rm_files(self):
        selected = sorted(self.filelist.curselection(), reverse=True)
        for item in selected:
            self.filelist.delete(item)
    
    def clear_files(self):
        self.filelist.delete(0,END)        
        
        
#------------------------------------------------------------------------------
        
class XPDataFrame(Frame):        
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)

        self.btnframe = Frame(parent.master)
        self.btnframe.grid(column=col, row=row, sticky=NW)
        
        self.list_label = Label(self.btnframe, text='XP list:', 
                                font=guiParams['labelfontstr'])
        self.list_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)
        
        # button used to initiate the scan of the above directories
        self.unpack_btn =Button(self.btnframe, text='Unpack HDF5 Files',
                                        command= lambda: self.add_xp(parent))
        self.unpack_btn.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Files')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_xp
        self.remove_button.grid(column=col, row=row+2, padx=10, pady=2,
                                sticky=NW)
         
        self.clear_button = Button(self.btnframe, text='Clear All')
        self.clear_button['command'] = self.clear_xp
        self.clear_button.grid(column=col, row=row+3, padx=10, pady=2,
                                sticky=NW)                         
        # label to show total files found and their size
        # this label is blank to hide it until required to be shown
        #self.total_label = Label(parent)
        #self.total_label.grid(column=col+1, row=row+2, padx=10, pady=2,
        #                      sticky=E)
                                
        self.xplistframe = Frame(parent.master) 
        self.xplistframe.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.xplist = Listbox(self.xplistframe,  width=64, height=8, 
                              selectmode=EXTENDED)
        
        self.xplist.bind('<<ListboxSelect>>', self.on_select)
        #self.filelist.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.hscroll = Scrollbar(self.xplistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.xplistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.xplist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        self.xplist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.xplist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.xplist.yview)

    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.xplist.curselection():
            self.remove_button.configure(state=NORMAL)
        else:
            self.remove_button.configure(state=DISABLED)
            
    def add_xp(self,parent):
        xp_list = self.xplist.get(0,END)
        #Expresso.clear_channellist(parent)
        newxp = Expresso.unpack_files(parent)
        
        for xp in tuple(newxp):
            if xp not in xp_list:
                self.xplist.insert(END,xp)
    
    def rm_xp(self):
        selected = sorted(self.xplist.curselection(), reverse=True)
        for item in selected:
            self.xplist.delete(item)
        
    def clear_xp(self):
        self.xplist.delete(0,END)
    
#------------------------------------------------------------------------------
    
class ChannelDataFrame(Frame):        
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)

        self.btnframe = Frame(parent.master)
        self.btnframe.grid(column=col, row=row, sticky=(N,S,E,W))
        
        self.list_label = Label(self.btnframe, text='Channel list:', 
                                font=guiParams['labelfontstr'])
        self.list_label.grid(column=col, row=row, padx=10, pady=2, sticky=N)
        
        # button used to initiate the scan of the above directories
        self.unpack_btn =Button(self.btnframe, text='Unpack XP',
                                        command= lambda: self.add_channels(parent))
        self.unpack_btn.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Files')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_channel
        self.remove_button.grid(column=col, row=row+2, padx=10, pady=2,
                                sticky=NW)
                   
        self.clear_button = Button(self.btnframe, text='Clear All')
        self.clear_button['command'] = self.clear_channel
        self.clear_button.grid(column=col, row=row+3, padx=10, pady=2,
                                sticky=NW)                  
                                
        self.plot_button = Button(self.btnframe, text='Plot Channel')
        self.plot_button['state'] = 'disabled'
        self.plot_button['command'] = lambda: self.plot_channel(parent)
        self.plot_button.grid(column=col, row=row+4, padx=10, pady=2,
                                sticky=SW) 
        
        self.save_button = Button(self.btnframe, text='Save CSV')
        self.save_button['state'] = 'disabled'
        self.save_button['command'] = lambda: self.save_results(parent)
        self.save_button.grid(column=col, row=row+5, padx=10, pady=2,
                                sticky=SW)                        
                                
                                
        self.channellistframe = Frame(parent.master) 
        self.channellistframe.grid(column=col+1, row=row, padx=10, pady=2, 
                                   sticky=(N,S,E,W))
        
        self.channellist = Listbox(self.channellistframe,  width=64, height=10,
                                   selectmode=EXTENDED)
        
        self.channellist.bind('<<ListboxSelect>>', self.on_select)
        #self.filelist.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.hscroll = Scrollbar(self.channellistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.channellistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.channellist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        self.channellist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.channellist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.channellist.yview)

        
        self.selection_ind = []                        

    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.channellist.curselection():
            self.remove_button.configure(state=NORMAL)
            self.plot_button.configure(state=NORMAL)
            self.save_button.configure(state=NORMAL)
            self.selection_ind = sorted(self.channellist.curselection(), 
                                        reverse=False)
            #print(self.selection_ind)
        else:
            self.remove_button.configure(state=DISABLED)
            self.plot_button.configure(state=DISABLED)
            self.save_button.configure(state=DISABLED)
            
            
    def add_channels(self,parent):
        channel_list = self.channellist.get(0,END)
        newchannels = Expresso.unpack_xp(parent)
        for channel in tuple(newchannels):
            if channel not in channel_list:
                self.channellist.insert(END,channel)
    
    def rm_channel(self):
        selected = sorted(self.channellist.curselection(), reverse=True)
        for item in selected:
            self.channellist.delete(item)
    
    def clear_channel(self):
        self.channellist.delete(0,END)        
    
    def plot_channel(self,parent):
        selected_ind = self.selection_ind
        if len(selected_ind) != 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select only one channel for plotting individual traces')
            return 
        
        menu_debug_flag = parent.debug_bout.get()
        channel_entry = self.channellist.get(selected_ind[0])
        dset, frames, t, dset_smooth, bouts, volumes = \
                            Expresso.get_channel_data(parent,channel_entry,
                                                    DEBUG_FLAG=menu_debug_flag) 
        
        if dset.size != 0:   

            self.fig, self.ax1, self.ax2 = plot_channel_bouts(dset,dset_smooth,
                                                              t,bouts)
            
            self.multi = MultiCursor(self.fig.canvas, (self.ax1, self.ax2),
                                     color='dodgerblue', lw=1.0, useblit=True,
                                     horizOn=True, vertOn=True)
                                    
            
            # get file info for title
            full_channellist_entry = self.channellist.get(self.selection_ind[0])
            filepath, filekeyname, groupkeyname = full_channellist_entry.split(', ',2)
            dirpath, filename = os.path.split(filepath) 
            self.channel_name_full = filename + ", " + filekeyname + ", " + groupkeyname
            
            self.fig.canvas.set_window_title(self.channel_name_full)    
            
            menu_save_flag = parent.save_all_plots.get()
            if menu_save_flag:
                filename_no_ext = os.path.splitext(filename)
                save_filename = filename_no_ext + '_' + filekeyname + "_" + \
                                    groupkeyname + '_bout_detection.png'
                savename_full = os.path.join(dirpath,save_filename)
                self.fig.savefig(savename_full)
            #self.save_button['state'] = 'normal'
            #self.cursor = matplotlib.widgets.MultiCursor(self.fig.canvas, (self.ax1, self.ax2), 
            #                                        color='black', linewidth=1, 
            #                                        horizOn=False,vertOn=True)
            #plt.show()                                        
            #plt.show(self.fig)
            
        else:
            tkMessageBox.showinfo(title='Error',
                                message='Invalid channel selection--no data in channel')
    
    def save_results(self,parent):
        selected_ind = self.selection_ind
        if len(selected_ind) != 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select only one channel for plotting individual traces')
            return 
        
        full_channellist_entry = self.channellist.get(selected_ind[0])
        
        _, _, self.t, _, self.bouts, self.volumes = \
                    Expresso.get_channel_data(parent,full_channellist_entry)
        
        #full_channellist_entry = self.channellist.get(self.selection_ind[0])
        filepath, filekeyname, groupkeyname = full_channellist_entry.split(', ',2)
        dirpath, filename = os.path.split(filepath) 
        self.channel_name_full = filename + ", " + filekeyname + ", " + groupkeyname
        
        if self.bouts.size > 0 :
            bouts_transpose = np.transpose(self.bouts)
            volumes_col = self.volumes.reshape(self.volumes.size,1)
            row_mat = np.hstack((bouts_transpose,self.t[bouts_transpose],volumes_col))
            
            if sys.version_info[0] < 3:
                save_file = tkFileDialog.asksaveasfile(mode='wb', 
                                defaultextension=".csv")
                save_writer = csv.writer(save_file)
            else:
                save_filename = tkFileDialog.asksaveasfilename(defaultextension=".csv")
                save_file = open(save_filename, 'w', newline='')
                save_writer = csv.writer(save_file)
            
            save_writer.writerow([self.channel_name_full])
            save_writer.writerow(['Bout Number'] + ['Bout Start [idx]'] + \
                ['Bout End [idx]'] + ['Bout Start [s]'] + ['Bout End [s]']+ ['Volume [nL]'])
            cc = 1            
            for row in row_mat:
                new_row = np.insert(row,0,cc)
                save_writer.writerow(new_row)
                cc += 1
        else:
            tkMessageBox.showinfo(title='Error',
                                message='No feeding bouts to save')  

#------------------------------------------------------------------------------
              
class VideoDataFrame(Frame):
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)
        
        self.btnframe = Frame(parent.master)
        self.btnframe.grid(column=col, row=row, sticky=(N, S, E, W))
        
        self.list_label = Label(self.btnframe, text='Video files:',
                                font = guiParams['labelfontstr'])
        self.list_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)
        
        # button used to initiate the scan of the above directories
        self.scan_btn =Button(self.btnframe, text='Get Video Files',
                                        command= lambda: self.add_files(parent))
        #self.scan_btn['state'] = 'disabled'                                
        self.scan_btn.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Files')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_files
        self.remove_button.grid(column=col, row=row+2, padx=10, pady=2,
                                sticky=NW)
                                
        self.clear_button = Button(self.btnframe, text='Clear All')
        self.clear_button['command'] = self.clear_files
        self.clear_button.grid(column=col, row=row+3, padx=10, pady=2,
                                sticky=NW)                  
       
        self.analyze_button = Button(self.btnframe, text='Analyze Video')
        self.analyze_button['state'] = 'disabled'
        self.analyze_button['command'] = lambda: self.analyze_vid(parent)
        self.analyze_button.grid(column=col, row=row+4, padx=10, pady=2,
                                sticky=SW)
                                
        self.plot_button = Button(self.btnframe, text='Plot Results')
        self.plot_button['state'] = 'disabled'
        self.plot_button['command'] = lambda: self.plot_tracking_results(parent)
        self.plot_button.grid(column=col, row=row+5, padx=10, pady=2,
                                sticky=SW)
        #----------------------------------------------------------------------
                                
        self.filelistframe = Frame(parent.master) 
        self.filelistframe.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.filelist = Listbox(self.filelistframe,  width=64, height=10, 
                                selectmode=EXTENDED)
        
        #now make the Listbox and Text drop targets
        if TKDND_FLAG:
            self.filelist.drop_target_register(DND_FILES, DND_TEXT)
        
        self.filelist.bind('<<ListboxSelect>>', self.on_select)
        
        if TKDND_FLAG:
            self.filelist.dnd_bind('<<DropEnter>>', Expresso.drop_enter)
            self.filelist.dnd_bind('<<DropPosition>>', Expresso.drop_position)
            self.filelist.dnd_bind('<<DropLeave>>', Expresso.drop_leave)
            self.filelist.dnd_bind('<<Drop>>', Expresso.file_drop)
            self.filelist.dnd_bind('<<Drop:DND_Files>>', Expresso.vid_file_drop)
            self.filelist.dnd_bind('<<Drop:DND_Text>>', Expresso.vid_file_drop)
            
            self.filelist.drag_source_register(1, DND_TEXT, DND_FILES)
            #text.drag_source_register(3, DND_TEXT)
        
            self.filelist.dnd_bind('<<DragInitCmd>>', Expresso.drag_init_listbox)
            self.filelist.dnd_bind('<<DragEndCmd>>', Expresso.drag_end)
            #text.dnd_bind('<<DragInitCmd>>', drag_init_text)    
        
    
        #self.filelist.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        # scroll bars
        self.hscroll = Scrollbar(self.filelistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.filelistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.filelist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        self.filelist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.filelist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.filelist.yview)


        self.selection_ind = [] 
        self.vid_filepath = '' 
        
    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """
        if self.filelist.curselection():
            self.remove_button.configure(state=NORMAL)
            self.analyze_button.configure(state=NORMAL)
            self.plot_button.configure(state=NORMAL)
            self.selection_ind = sorted(self.filelist.curselection(), 
                                        reverse=False)
        else:
            self.remove_button.configure(state=DISABLED)
            self.analyze_button.configure(state=DISABLED)
            self.plot_button.configure(state=DISABLED)
            
    def add_files(self,parent):
        newfiles = Expresso.scan_dirs_vid(parent)
        file_list = self.filelist.get(0,END)
        if len(newfiles) > 0:
            for file in tuple(newfiles):
                if file not in file_list:
                    self.filelist.insert(END,file)
    
    def rm_files(self):
        selected = sorted(self.filelist.curselection(), reverse=True)
        for item in selected:
            self.filelist.delete(item)
    
    def clear_files(self):
        self.filelist.delete(0,END)      
        
    def analyze_vid(self,parent):
        selected_ind = self.selection_ind
        if len(selected_ind) != 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select only one video file for analysis')
            return 
        
        menu_debug_flag = parent.debug_tracking.get()
        save_debug_flag = not(menu_debug_flag)
        file_entry = self.filelist.get(selected_ind[0])
        file_path, filename = os.path.split(file_entry)
        _ = visual_expresso_main(file_path, filename, 
                            DEBUG_BG_FLAG=menu_debug_flag, 
                            DEBUG_CM_FLAG=menu_debug_flag, 
                            SAVE_DATA_FLAG=save_debug_flag, 
                            ELLIPSE_FIT_FLAG = False, 
                            PARAMS=trackingParams)
                            
        filename_prefix = os.path.splitext(filename)[0]
        track_filename = filename_prefix + "_TRACKING.hdf5"  
                  
        _ = process_visual_expresso(file_path, track_filename,
                            SAVE_DATA_FLAG=save_debug_flag, DEBUG_FLAG=False)
                            
        self.vid_filepath = file_entry
    
    def plot_tracking_results(self,parent):
        selected_ind = self.selection_ind
        if len(selected_ind) != 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select only one video file for plotting')
            return 
        
        file_entry = self.filelist.get(selected_ind[0])
        file_path, filename = os.path.split(file_entry)
        
        menu_save_flag = parent.save_all_plots.get()
        
        try:
            savename_prefix = os.path.splitext(filename)[0]
            save_filename = os.path.join(file_path, savename_prefix + \
                                                    "_TRACKING_PROCESSED.hdf5")
            save_filename = os.path.abspath(save_filename)
            #save_filename = os.path.normpath(save_filename)
            
#            if self.vid_filepath == file_entry:
#                plot_body_cm(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag)
#                plot_body_vel(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag) 
#                plot_moving_v_still(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag)
#                plot_cum_dist(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag)
            if os.path.exists(save_filename):
                self.vid_filepath = file_entry
                _, track_filename = os.path.split(save_filename)
                self.flyTrackData_smooth = \
                                hdf5_to_flyTrackData(file_path, track_filename)
                plot_body_cm(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag)
                plot_body_vel(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag) 
                plot_moving_v_still(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag)
                plot_cum_dist(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag)
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
                        
#------------------------------------------------------------------------------

class BatchFrame(Frame):
    def __init__(self, parent, root, col=0,row=0):
        Frame.__init__(self, parent)
        
#        self.list_label = Label(parent, text='Batch analyze list:',
#                                font = guiParams['labelfontstr'])
#        self.list_label.grid(column=col, row=row, padx=10, pady=10, sticky=NW)
        
        self.batchlistframe = Frame(self) 
        self.batchlistframe.grid(column=col, row=row, columnspan = 2,
                                 rowspan = 2, padx=10, pady=35,  
                                 sticky=(N, S, E, W))
        
        self.batchlist = Listbox(self.batchlistframe,  width=90, height=16,
                                   selectmode=EXTENDED)
        
        self.batchlist.bind('<<ListboxSelect>>', self.on_select)
        
        self.hscroll = Scrollbar(self.batchlistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.batchlistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.batchlist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        
        self.batchlist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.batchlist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.batchlist.yview)
        
        self.entryframe = Frame(self)
        self.entryframe.grid(column=col, row=row+2,columnspan = 1, rowspan = 1,
                             pady=4, padx=4, sticky='NESW')
        
        self.tmin_entry_label = Label(self.entryframe, text='t_min')
        self.tmin_entry_label.grid(column=col, row=row, padx=10, pady=2, sticky=N)
        self.tmin_entry = Entry(self.entryframe, width=8)
        self.tmin_entry.insert(END,'0')
        self.tmin_entry.grid(column=col+1, row=row,padx=10, pady=2, sticky=N)
        
        self.tmax_entry_label = Label(self.entryframe, text='t_max')
        self.tmax_entry_label.grid(column=col, row=row+1, padx=10, pady=2, sticky=N)
        self.tmax_entry = Entry(self.entryframe, width=8)
        self.tmax_entry.insert(END,'2000')
        self.tmax_entry.grid(column=col+1, row=row+1,padx=10, pady=2, sticky=N)
        
        self.tbin_entry_label = Label(self.entryframe, text='t_bin')
        self.tbin_entry_label.grid(column=col, row=row+2, padx=10, pady=2, sticky=N)
        self.tbin_entry = Entry(self.entryframe, width=8)
        self.tbin_entry.insert(END,'20')
        self.tbin_entry.grid(column=col+1, row=row+2,padx=10, pady=2, sticky=N)                           
        
        
#        self.btnframe = Frame(parent.master)
#        self.btnframe.grid(column=col+1, row=row+2, columnspan = 2, rowspan = 1,
#                           pady = 0, sticky = W)
        
        # button used to initiate the scan of the above directories
        self.add_btn =Button(self.entryframe, text='Add Channel(s) to Batch',
                                        command= lambda: self.add_to_batch(root))
        self.add_btn.grid(column=col+2, row=row, padx=10, pady=2,
                                sticky=N)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.entryframe, text='Remove Selected')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_channel
        self.remove_button.grid(column=col+2, row=row+1, padx=10, pady=2,
                                sticky=N)
        
        # button used to clear all batch files
        self.clear_button = Button(self.entryframe, text='Clear All')
        self.clear_button['command'] = self.clear_batch
        self.clear_button.grid(column=col+2, row=row+2, padx=10, pady=2,
                                sticky=N)
                                
        self.plot_button = Button(self.entryframe, text='Plot Batch Analysis')
        #self.plot_button['state'] = 'disabled'
        self.plot_button['command'] = self.plot_batch
        self.plot_button.grid(column=col+3, row=row, padx=10, pady=2,
                                sticky=S) 
        
        self.save_button = Button(self.entryframe, text='Save Batch Results')
        #self.save_button['state'] = 'disabled'
        self.save_button['command'] = self.save_batch
        self.save_button.grid(column=col+3, row=row+1, padx=10, pady=2,
                                sticky=S)                        
        
        self.save_ts_button = Button(self.entryframe, text='Save Time Series')
        #self.save_button['state'] = 'disabled'
        self.save_ts_button['command'] = lambda: self.save_time_series(root)
        self.save_ts_button.grid(column=col+3, row=row+2, padx=10, pady=2,
                                sticky=S)         
                                
        self.selection_ind = []                        

    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.batchlist.curselection():
            self.remove_button.configure(state=NORMAL)
            #self.plot_button.configure(state=NORMAL)
            self.selection_ind = sorted(self.batchlist.curselection(), reverse=True)
            #print(self.selection_ind)
        else:
            self.remove_button.configure(state=DISABLED)
            #self.plot_button.configure(state=DISABLED)
            
    def add_to_batch(self,root):
        batch_list = self.batchlist.get(0,END)
        for_batch = Expresso.fetch_channels_for_batch(root)
        for_batch = sorted(for_batch, reverse=False)
        for channel in tuple(for_batch):
            if channel not in batch_list:
                self.batchlist.insert(END,channel)
    
    def rm_channel(self):
        selected = sorted(self.batchlist.curselection(), reverse=True)
        for item in selected:
            self.batchlist.delete(item)
    
    def clear_batch(self):
        self.batchlist.delete(0,END)         
        
    def plot_batch(self):
        batch_list = self.batchlist.get(0,END)
        comb_analysis_flag = parent.comb_analysis_flag.get()
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            try:
                tmin = int(self.tmin_entry.get())
                tmax = int(self.tmax_entry.get())
                tbin = int(self.tbin_entry.get())
            except:
                tkMessageBox.showinfo(title='Error',
                                message='Set time range and bin size')
                return                
            
            (self.bouts_list, self.name_list, self.volumes_list, self.consumption_per_fly, 
             self.duration_per_fly, self.latency_per_fly, self.fig_raster, 
             self.fig_hist) = batch_bout_analysis(batch_list, tmin, tmax, tbin,
                            plotFlag=True,combAnalysisFlag=comb_analysis_flag)
             
             #self.save_button['state'] = 'enabled'   
    
    def save_batch(self):
        batch_list = self.batchlist.get(0,END)
        comb_analysis_flag = parent.comb_analysis_flag.get()
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            try:
                tmin = int(self.tmin_entry.get())
                tmax = int(self.tmax_entry.get())
                tbin = int(self.tbin_entry.get())
            except:
                tkMessageBox.showinfo(title='Error',
                                message='Set time range and bin size')
                return                
            
            (self.bouts_list, self.name_list, self.volumes_list, self.consumption_per_fly, 
             self.duration_per_fly, self.latency_per_fly) = \
                 batch_bout_analysis(batch_list, tmin, tmax, tbin,plotFlag=False, 
                                     combAnalysisFlag=comb_analysis_flag)
            
            save_filename = tkFileDialog.asksaveasfilename(defaultextension=".xlsx")
            save_batch_xlsx(save_filename, self.bouts_list,self.name_list,
                        self.volumes_list,self.consumption_per_fly, 
                        self.duration_per_fly, self.latency_per_fly)

    def save_time_series(self,root):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            save_dir = Expresso.get_dir(parent)
            
            for entry in batch_list:
                                
                dset, frames, t, dset_smooth, bouts, _ = \
                                        Expresso.get_channel_data(root,entry) 
                feeding_boolean = np.zeros([1,dset.size])
                for i in np.arange(bouts.shape[1]):
                    feeding_boolean[0,bouts[0,i]:bouts[1,i]] = 1
                row_mat = np.vstack((frames, t, dset, dset_smooth, feeding_boolean))
                row_mat = np.transpose(row_mat)
                
                filepath, filekeyname, groupkeyname = entry.split(', ',2)
                dirpath, filename = os.path.split(filepath) 
                save_name = filename[:-5] + "_" + filekeyname + "_" + groupkeyname + ".csv"
                save_path = os.path.join(save_dir,save_name)
                if sys.version_info[0] < 3:
                    out_path = open(save_path,mode='wb')
                else:
                    out_path = open(save_path, 'w', newline='')
                    
                save_writer = csv.writer(out_path)
                
                save_writer.writerow(['Idx'] + ['Time [s]'] + \
                    ['Data Raw [nL]'] + ['Data Smoothed [nL]'] + ['Feeding [bool]'])
                #cc = 1            
                for row in row_mat:
                    #new_row = np.insert(row,0,cc)
                    save_writer.writerow(row)
                    #cc += 1
                    
                out_path.close()

#------------------------------------------------------------------------------

class BatchVidFrame(Frame):
    def __init__(self, parent, root, col=0,row=0):
        Frame.__init__(self, parent)
        
#        self.list_label = Label(parent, text='Batch analyze list:',
#                                font = guiParams['labelfontstr'])
#        self.list_label.grid(column=col, row=row, padx=10, pady=10, sticky=NW)
        
        self.batchlistframe = Frame(self) 
        self.batchlistframe.grid(column=col, row=row, columnspan = 2,
                                 rowspan = 2, padx=10, pady=35,  
                                 sticky=(N, S, E, W))
        
        self.batchlist = Listbox(self.batchlistframe,  width=90, height=16,
                                   selectmode=EXTENDED)
        
        self.batchlist.bind('<<ListboxSelect>>', self.on_select)
        
        self.hscroll = Scrollbar(self.batchlistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.batchlistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.batchlist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        
        self.batchlist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.batchlist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.batchlist.yview)
        
        self.entryframe = Frame(self)
        self.entryframe.grid(column=col, row=row+2,columnspan = 4, rowspan = 1,
                             pady=4, padx=4, sticky='NESW')
              

#        self.btnframe = Frame(parent.master)
#        self.btnframe.grid(column=col+1, row=row+2, columnspan = 2, rowspan = 1,
#                           pady = 0, sticky = W)
        
        # button used to initiate the scan of the above directories
        self.add_btn =Button(self.entryframe, text='Add Video(s) to Batch',
                                        command= lambda: self.add_to_batch(root))
        self.add_btn.grid(column=col+2, row=row, padx=10, pady=2,
                                sticky=N)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.entryframe, text='Remove Selected')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_channel
        self.remove_button.grid(column=col+2, row=row+1, padx=10, pady=2,
                                sticky=N)
        
        # button used to clear all batch files
        self.clear_button = Button(self.entryframe, text='Clear All')
        self.clear_button['command'] = self.clear_batch
        self.clear_button.grid(column=col+2, row=row+2, padx=10, pady=2,
                                sticky=N)
        
        # button used to analyze movies and save results in hdf5 files                        
        self.analyze_button = Button(self.entryframe, text='Analyze/Save Video(s)')
        #self.plot_button['state'] = 'disabled'
        self.analyze_button['command'] = self.analyze_batch
        self.analyze_button.grid(column=col+3, row=row, padx=10, pady=2,
                                sticky=S) 
        
        # button to save CSV file of video summary
        self.save_button = Button(self.entryframe, text='Save Video Batch Summary')
        #self.save_button['state'] = 'disabled'
        self.save_button['command'] = self.save_batch
        #self.save_button['command'] = self.plot_cum_dist_batch
        self.save_button.grid(column=col+3, row=row+1, padx=10, pady=2,
                                sticky=S)                        
        
        # button to save CSV file of video time series
        self.save_ts_button = Button(self.entryframe, text='Save Video Time Series')
        #self.save_button['state'] = 'disabled'
        self.save_ts_button['command'] = lambda: self.save_time_series(root)
        self.save_ts_button.grid(column=col+3, row=row+2, padx=10, pady=2,
                                sticky=S)         
        
        # button to plot cumulative distance traveled                       
        self.plot_heatmap_button = Button(self.entryframe, text='Plot Heatmap')
        #self.save_button['state'] = 'disabled'
        self.plot_heatmap_button['command'] = lambda: self.plot_heatmap_batch(root)
        self.plot_heatmap_button.grid(column=col+4, row=row, padx=10, pady=2,
                                sticky=S)         
                                
        # button to plot cumulative distance traveled                       
        self.plot_cum_dist_button = Button(self.entryframe, text='Plot Cumulative Distance')
        #self.save_button['state'] = 'disabled'
        self.plot_cum_dist_button['command'] = lambda: self.plot_cum_dist_batch(root)
        self.plot_cum_dist_button.grid(column=col+4, row=row+1, padx=10, pady=2,
                                sticky=S)         
                                
        self.selection_ind = []                        

    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.batchlist.curselection():
            self.remove_button.configure(state=NORMAL)
            #self.plot_button.configure(state=NORMAL)
            self.selection_ind = sorted(self.batchlist.curselection(), reverse=True)
            #print(self.selection_ind)
        else:
            self.remove_button.configure(state=DISABLED)
            #self.plot_button.configure(state=DISABLED)
            
    def add_to_batch(self,root):
        batch_list = self.batchlist.get(0,END)
        for_batch = Expresso.fetch_videos_for_batch(root)
        for_batch = sorted(for_batch, reverse=False)
        for vid in tuple(for_batch):
            if vid not in batch_list:
                self.batchlist.insert(END,vid)
    
    def rm_channel(self):
        selected = sorted(self.batchlist.curselection(), reverse=True)
        for item in selected:
            self.batchlist.delete(item)
    
    def clear_batch(self):
        self.batchlist.delete(0,END)         
        
    def analyze_batch(self):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add video(s) to batch box for batch analysis')
            return 
        else:
           for vid_file in batch_list:
               file_path, filename = os.path.split(vid_file)
               try:
                   visual_expresso_main(file_path, filename, 
                                    DEBUG_BG_FLAG=False, DEBUG_CM_FLAG=False, 
                                    SAVE_DATA_FLAG=True,ELLIPSE_FIT_FLAG = False, 
                                    PARAMS=trackingParams)
                   filename_prefix = os.path.splitext(filename)[0]
                   track_filename = filename_prefix + "_TRACKING.hdf5"
                   process_visual_expresso(file_path, track_filename,
                                    SAVE_DATA_FLAG = True, DEBUG_FLAG = False)
               except:
                   e = sys.exc_info()[0]
                   print('Error:')
                   print(e)
        self.vid_file_list = batch_list
               
    def save_batch(self):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add videos to batch box for batch analysis')
            return 
        else:
            csv_filename = tkFileDialog.asksaveasfilename(initialdir=sys.path[0],
                                           defaultextension=".csv",
                                           title='Select save filename') 
            save_vid_summary(batch_list, csv_filename)
            
    def save_time_series(self,root):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add videos to batch box for batch analysis')
            return 
        else:
            save_vid_time_series(batch_list)
            
    def plot_cum_dist_batch(self,root):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            batch_plot_cum_dist(batch_list)
    
    def plot_heatmap_batch(self,root):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            batch_plot_heatmap(batch_list)
           
#------------------------------------------------------------------------------

class BatchCombinedFrame(Frame):
    def __init__(self, parent, root, col=0,row=0):
        Frame.__init__(self, parent)
        
#        self.list_label = Label(parent, text='Batch analyze list:',
#                                font = guiParams['labelfontstr'])
#        self.list_label.grid(column=col, row=row, padx=10, pady=10, sticky=NW)
        
        self.batchlistframe = Frame(self) 
        self.batchlistframe.grid(column=col, row=row, columnspan = 2,
                                 rowspan = 2, padx=10, pady=35,  
                                 sticky=(N, S, E, W))
        
        self.batchlist = Listbox(self.batchlistframe,  width=90, height=16,
                                   selectmode=EXTENDED)
        
        self.batchlist.bind('<<ListboxSelect>>', self.on_select)
        
        self.hscroll = Scrollbar(self.batchlistframe)
        self.hscroll.pack(side=BOTTOM, fill=X)

        self.vscroll = Scrollbar(self.batchlistframe)
        self.vscroll.pack(side=RIGHT, fill=Y)
        
        self.batchlist.config(xscrollcommand=self.hscroll.set,
                               yscrollcommand=self.vscroll.set)
        
        self.batchlist.pack(side=TOP, fill=BOTH)
        
        self.hscroll.configure(orient=HORIZONTAL,
                               command=self.batchlist.xview)
        self.vscroll.configure(orient=VERTICAL,
                               command=self.batchlist.yview)
        
        self.entryframe = Frame(self)
        self.entryframe.grid(column=col, row=row+2,columnspan = 3, rowspan = 1,
                             pady=4, padx=4, sticky='NESW')
              

#        self.btnframe = Frame(parent.master)
#        self.btnframe.grid(column=col+1, row=row+2, columnspan = 2, rowspan = 1,
#                           pady = 0, sticky = W)
        
        # button used to initiate the scan of the above directories
        self.add_btn =Button(self.entryframe, text='Add Data to Batch',
                                        command= lambda: self.add_to_batch(root))
        self.add_btn.grid(column=col+2, row=row, padx=10, pady=2,
                                sticky=N)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.entryframe, text='Remove Selected')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_channel
        self.remove_button.grid(column=col+2, row=row+1, padx=10, pady=2,
                                sticky=N)
        
        # button used to clear all batch files
        self.clear_button = Button(self.entryframe, text='Clear All')
        self.clear_button['command'] = self.clear_batch
        self.clear_button.grid(column=col+2, row=row+2, padx=10, pady=2,
                                sticky=N)
                                
        self.analyze_button = Button(self.entryframe, text='Analyze/Save Data')
        #self.plot_button['state'] = 'disabled'
        self.analyze_button['command'] = lambda: self.analyze_batch(root)
        self.analyze_button.grid(column=col+3, row=row, padx=10, pady=2,
                                sticky=S) 
        
        self.save_button = Button(self.entryframe, text='Save Batch Summary')
        #self.save_button['state'] = 'disabled'
        self.save_button['command'] = self.save_batch
        self.save_button.grid(column=col+3, row=row+1, padx=10, pady=2,
                                sticky=S)                        
        
        self.save_ts_button = Button(self.entryframe, text='Save Time Series')
        #self.save_button['state'] = 'disabled'
        self.save_ts_button['command'] = lambda: self.save_time_series(root)
        self.save_ts_button.grid(column=col+3, row=row+2, padx=10, pady=2,
                                sticky=S)         
        
        # button to plot meal-aligned distance from cap tip traveled                       
        self.plot_dist_button = Button(self.entryframe, text='Plot Meal-Aligned Dist')
        #self.save_button['state'] = 'disabled'
        self.plot_dist_button['command'] = lambda: self.plot_meal_aligned_dist(root)
        self.plot_dist_button.grid(column=col+4, row=row, padx=10, pady=2,
                                sticky=S)         
                                
        # button to plot meal-aligned speed                            
        self.plot_vel_button = Button(self.entryframe, text='Plot Meal-Aligned Speed')
        #self.save_button['state'] = 'disabled'
        self.plot_vel_button['command'] = lambda: self.plot_meal_aligned_vel(root)
        self.plot_vel_button.grid(column=col+4, row=row+1, padx=10, pady=2,
                                sticky=S)         
                                
        self.selection_ind = []                        

    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.batchlist.curselection():
            self.remove_button.configure(state=NORMAL)
            #self.plot_button.configure(state=NORMAL)
            self.selection_ind = sorted(self.batchlist.curselection(), reverse=True)
            #print(self.selection_ind)
        else:
            self.remove_button.configure(state=DISABLED)
            #self.plot_button.configure(state=DISABLED)
            
    def add_to_batch(self,root):
        batch_list = self.batchlist.get(0,END)
        for_batch = Expresso.fetch_data_for_batch(root)
        for_batch = sorted(for_batch, reverse=False)
        for ent in tuple(for_batch):
            if ent not in batch_list:
                self.batchlist.insert(END,ent)
    
    def rm_channel(self):
        selected = sorted(self.batchlist.curselection(), reverse=True)
        for item in selected:
            self.batchlist.delete(item)
    
    def clear_batch(self):
        self.batchlist.delete(0,END)         
        
    def analyze_batch(self,root):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
           for data_file in batch_list:
               
               # perform tracking analysis
               file_path, filename_no_ext = os.path.split(data_file)
               vid_filename = filename_no_ext + '.avi' 
               visual_expresso_main(file_path, vid_filename, 
                                DEBUG_BG_FLAG=False, DEBUG_CM_FLAG=False, 
                                SAVE_DATA_FLAG=True, ELLIPSE_FIT_FLAG = False, 
                                PARAMS=trackingParams)
               filename_prefix = os.path.splitext(vid_filename)[0]
               track_filename = filename_prefix + "_TRACKING.hdf5"
               flyTrackData = process_visual_expresso(file_path, track_filename,
                                SAVE_DATA_FLAG = True, DEBUG_FLAG = False)
               
               # perform feeding analysis
               channel_entry = basic2channel(data_file)
               dset, frames, channel_t, dset_smooth, bouts, volumes = \
                            Expresso.get_channel_data(root,channel_entry,
                                                    DEBUG_FLAG=False,
                                                    combFlagArg=True) 
               
               # merge data into one dict structure                                     
               flyCombinedData = merge_v_expresso_data(dset,dset_smooth,
                                                       channel_t,frames,bouts, 
                                                       volumes, flyTrackData) 
               flyCombinedData_to_hdf5(flyCombinedData)
               
        self.comb_file_list = batch_list
               
    def save_batch(self):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            # save video summary
            csv_filename_vid = tkFileDialog.asksaveasfilename(initialdir=sys.path[0],
                                           defaultextension=".csv",
                                           title='Select save filename for VIDEO summary') 
            batch_list_vid = [basic2vid(ent) for ent in batch_list]
            save_vid_summary(batch_list_vid, csv_filename_vid)
            
            # run and save feeding analysis (a little redundant)
            batch_list_ch = [basic2channel(ent) for ent in batch_list]
                    
            tmin = np.nan # this tells the code to just take limits from data
            tmax = np.nan
            tbin = np.nan
            
            (bouts_list, name_list, volumes_list, consumption_per_fly, 
             duration_per_fly, latency_per_fly) = \
                 batch_bout_analysis(batch_list_ch,tmin,tmax,tbin,plotFlag=False, 
                                     combAnalysisFlag=True)
            
            savename_ch = tkFileDialog.asksaveasfilename(initialdir=sys.path[0],
                                           defaultextension=".xlsx",
                                           title='Select save filename for CHANNEL summary') 
            save_batch_xlsx(savename_ch, bouts_list, name_list, volumes_list,
                            consumption_per_fly,duration_per_fly,latency_per_fly)
            
    def save_time_series(self,root):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            data_suffix = '_COMBINED_DATA.hdf5'
            data_filenames = [ent + data_suffix for ent in batch_list]
            save_comb_time_series(data_filenames)
    
    def plot_meal_aligned_dist(self,root):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            fig = plot_bout_aligned_var(batch_list, var='dist_mag')
    
    def plot_meal_aligned_vel(self,root):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            fig = plot_bout_aligned_var(batch_list, var='vel_mag')

#==============================================================================
# Main class for GUI
#==============================================================================
class Expresso:
    """The GUI and functions."""
    def __init__(self, master):
        #Tk.__init__(self)
        self.master = master        
        
        # style
        #???

        # initialize important fields for retaining where we are in data space
        self.initdirs = [] 
        init_dirs = initDirectories 
        
        for init_dir in init_dirs:  
            if os.path.exists(init_dir):
                #print(init_dir)
                self.initdirs.append(init_dir)
        
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
        
        #--------------------------------
        # debugging boolean variables
        #--------------------------------
        self.debug_tracking= IntVar()
        self.debug_tracking.set(0)
        
        self.debug_bout = IntVar()
        self.debug_bout.set(0)
        
        self.save_all_plots = IntVar()
        self.save_all_plots.set(0)
        
        #--------------------------------------
        # combined analysis boolean variable(s)
        #--------------------------------------
        self.comb_analysis_flag = IntVar()
        self.comb_analysis_flag.set(0)
        
        # run gui presets. may be unecessary
        self.init_gui()
        
        # initialize instances of frames created above
        self.dirframe = DirectoryFrame(self, col=0, row=0)
        self.fdata_frame = FileDataFrame(self, col=0, row=1)
        self.xpdata_frame = XPDataFrame(self, col=0, row=2)
        self.channeldata_frame = ChannelDataFrame(self, col=0, row=3)
        #self.batchdata_frame = BatchFrame(self,col=3,row=1)
        self.viddata_frame = VideoDataFrame(self,col=0,row=4)
        #self.extrabtns_frame = ExtraButtonsFrame(self,col=5,row=5)
        
        # ttk notebook for batch analysis
        self.batch_nb = Notebook(self.master)
        self.batch_nb.grid(row=1,column=3,rowspan=3, columnspan = 3, sticky='NESW')
        
        # batch channel analysis
        self.batchdata_frame = BatchFrame(self.batch_nb,self)
        self.batchvid_frame =  BatchVidFrame(self.batch_nb,self)
        self.batchcomb_frame = BatchCombinedFrame(self.batch_nb, self)
        
        self.batch_nb.add(self.batchdata_frame,text='Batch Channel Analysis')
        self.batch_nb.add(self.batchvid_frame,text='Batch Video Analysis')
        self.batch_nb.add(self.batchcomb_frame,text='Batch Combined Analysis')
        
         
        # insert logo image!
        self.img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        'expresso_alpha.gif')
        #im = Image.open(im_path)
        #ph = ImageTk.PhotoImage(im)
        self.img = PhotoImage(file=self.img_path,master=self.master) 
        self.im_label = Label(self.master,image=self.img,
                              background=guiParams['bgcolor'])
        self.im_label.img = self.img
        self.im_label.grid(column=3, row=0, padx=10, pady=2, sticky=S)
        
        for datadir in self.initdirs:
            self.dirframe.dirlist.insert(END, datadir)
        
        #self.rawdata_plot = FigureFrame(self, col=4, row=0)
        
        #self.make_topmost()
        self.master.protocol("WM_DELETE_WINDOW", self.on_quit)
    
    #===================================================================    
    def on_quit(self):
        """Exits program."""
        if tkMessageBox.askokcancel("Quit","Do you want to quit?"):
            self.master.destroy()
            self.master.quit()
    
    #===================================================================
    def toggle_track_debug(self):
        """Turns on or off the tracking debug flags"""
        curr_val = self.debug_tracking.get()
        if curr_val == True:
            self.debug_tracking.set(0)
        elif curr_val == False:
            self.debug_tracking.set(1)
    
    #===================================================================
    def toggle_bout_debug(self):
        """Turns on or off the bout debug flags"""
        curr_val = self.debug_bout.get()
        if curr_val == True:
            self.debug_bout.set(0)
        elif curr_val == False:
            self.debug_bout.set(1)
    
    #===================================================================
    def toggle_save_all(self):
        """Turns on or off the save plots flags"""
        curr_val = self.save_all_plots.get()
        if curr_val == True:
            self.save_all_plots.set(0)
        elif curr_val == False:
            self.save_all_plots.set(1)
    
    #===================================================================
    def toggle_comb_analysis(self):
        """Turns on or off the synchronized selection flags"""
        curr_val = self.comb_analysis_flag.get()
        if curr_val == True:
            self.comb_analysis_flag.set(0)
        elif curr_val == False:
            self.comb_analysis_flag.set(1)
    
    #===================================================================     
    def sync_listboxes_intersect(self):
        """Repopulate the video and channel listboxes so that they match"""
        N_ch_entries = self.channeldata_frame.channellist.size()
        channel_entries = self.channeldata_frame.channellist.get(0,N_ch_entries)
        N_vid_entries = self.viddata_frame.filelist.size()
        vid_entries = self.viddata_frame.filelist.get(0,N_vid_entries)
        
        # reformat entry types to facilitate comparison
        vid_entries_refrm = [vid2basic(v_en) for v_en in vid_entries]
        channel_entries_refrm = [channel2basic(ch) for ch in channel_entries]
        
        # find set intersection of listbox entries
        vid_entries_set = set(vid_entries_refrm)
        channel_entries_set = set(channel_entries_refrm)
        entry_intersect = vid_entries_set.intersection(channel_entries_set)
        entry_intersect = sorted(list(entry_intersect),reverse=False)
        
        # delete old entries
        self.channeldata_frame.channellist.delete(0,N_ch_entries) 
        self.viddata_frame.filelist.delete(0,N_vid_entries)
        
        # switch back to format for the listboxes and add 
        for ent in entry_intersect:
            ent_vid = basic2vid(ent)
            self.viddata_frame.filelist.insert(END,ent_vid)
            
            ent_ch = basic2channel(ent)
            self.channeldata_frame.channellist.insert(END,ent_ch)
    
    #===================================================================     
    def sync_listboxes_union(self):
        """Repopulate the video and channel listboxes so that they match"""
        N_ch_entries = self.channeldata_frame.channellist.size()
        channel_entries = self.channeldata_frame.channellist.get(0,N_ch_entries)
        N_vid_entries = self.viddata_frame.filelist.size()
        vid_entries = self.viddata_frame.filelist.get(0,N_vid_entries)
        
        # reformat entry types to facilitate comparison
        vid_entries_refrm = [vid2basic(v_en) for v_en in vid_entries]
        channel_entries_refrm = [channel2basic(ch) for ch in channel_entries]
        
        # find set intersection of listbox entries
        vid_entries_set = set(vid_entries_refrm)
        channel_entries_set = set(channel_entries_refrm)
        entry_union = vid_entries_set.union(channel_entries_set)
        entry_union = sorted(list(entry_union),reverse=False)
        
        # delete old entries
        self.channeldata_frame.channellist.delete(0,N_ch_entries) 
        self.viddata_frame.filelist.delete(0,N_vid_entries)
        
        # switch back to format for the listboxes and add 
        for ent in entry_union:
            ent_vid = basic2vid(ent)
            self.viddata_frame.filelist.insert(END,ent_vid)
            
            ent_ch = basic2channel(ent)
            self.channeldata_frame.channellist.insert(END,ent_ch)
            
    #===================================================================
    def sync_select(self):
        """Synchronizes selection for channel and video listboxes"""
        #selected_vid_ind = self.viddata_frame.selection_ind
        #selected_channel_ind = self.channeldata_frame.selection_ind
        selected_vid_ind = self.viddata_frame.filelist.curselection()
        selected_channel_ind = self.channeldata_frame.channellist.curselection()
        
        if (len(selected_vid_ind) > 0) and (len(selected_channel_ind) < 1):
            new_channel_ind = [] 
            vid_entries = [self.viddata_frame.filelist.get(ind) for ind in \
                            selected_vid_ind] 
            basic_entries = [vid2basic(v_ent) for v_ent in vid_entries]
            for b_ent in basic_entries:
                ch_ent = basic2channel(b_ent)
                if ch_ent not in self.channeldata_frame.channellist.get(0,END):
                    self.channeldata_frame.channellist.insert(END,ch_ent)
                ch_ind = self.channeldata_frame.channellist.get(0,END).index(ch_ent)
                new_channel_ind.append(ch_ind)
                #self.channeldata_frame.channellist.selection_set(ch_ind)
            #self.channeldata_frame.selection_ind = new_channel_ind
        
        elif (len(selected_vid_ind) < 1) and (len(selected_channel_ind) > 0):
            new_vid_ind = [] 
            ch_entries = [self.channeldata_frame.channellist.get(ind) for \
                                ind in selected_channel_ind] 
            basic_entries = [channel2basic(ch_ent) for ch_ent in ch_entries]
            for b_ent in basic_entries:
                vid_ent = basic2vid(b_ent)
                if vid_ent not in self.viddata_frame.filelist.get(0,END):
                    self.viddata_frame.filelist.insert(END,vid_ent)
                vid_ind = self.viddata_frame.filelist.get(0,END).index(vid_ent)
                new_vid_ind.append(vid_ind)
                #self.viddata_frame.filelist.selection_set(vid_ind)
            #self.viddata_frame.selection_ind = new_vid_ind
            
        else:
            print('Error--need to fix this')
            
    #===================================================================     
    def make_topmost(self):
        """Makes this window the topmost window"""
        self.master.lift()
        self.master.attributes("-topmost", 1)
        self.master.attributes("-topmost", 0) 
        
    #===================================================================    
    def init_gui(self):
        """Label for GUI"""
        
        self.master.title('Visual Expresso Data Analysis')
        
        """ Menu bar """
        self.master.option_add('*tearOff', 'FALSE')
        self.master.menubar = Menu(self.master)
        
        # file menu
        self.master.menu_file = Menu(self.master.menubar)
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
                                               command = self.toggle_track_debug)
        
        # combined analysis
        self.master.menu_comb = Menu(self.master.menubar)
        self.master.menu_comb.add_command(label='Synchronize data lists (intersection)',
                                          command=self.sync_listboxes_intersect)
        self.master.menu_comb.add_command(label='Synchronize data lists (union)',
                                          command=self.sync_listboxes_union)
        self.master.menu_comb.add_command(label='Synchronize selection',
                                              command = self.sync_select)
        self.master.menu_comb.add_checkbutton(label='Combine Data Types [toggle]',
                                              variable=self.comb_analysis_flag,
                                              command = self.toggle_comb_analysis)
        
        # add these bits to menu bar
        self.master.menubar.add_cascade(menu=self.master.menu_file, label='File')
        self.master.menubar.add_cascade(menu=self.master.menu_debug, 
                                        label='Debugging Options')
        self.master.menubar.add_cascade(menu=self.master.menu_comb, 
                                        label='Combined Analysis Tools')
 
        self.master.config(menu=self.master.menubar)
        
        #self.master.config(background='white')
        
    
    #===================================================================    
    @staticmethod
    def get_dir(self):
        """ Method to return the directory selected by the user which should
            be scanned by the application. """

        # get user specified directory and normalize path
        start_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        #seldir = tkFileDialog.askdirectory(initialdir=sys.path[0])
        seldir = tkFileDialog.askdirectory(initialdir=start_path)
        if seldir:
            seldir = os.path.abspath(seldir)
            self.datadir_curr = seldir
            return seldir
    
    #===================================================================
    @staticmethod        
    def scan_dirs(self):
        # build list of detected files from selected paths
        files = [] 
        temp_dirlist = list(self.dirframe.dirlist.get(0, END))
        selected_ind = sorted(self.dirframe.dirlist.curselection(), reverse=False)
        
        if len(selected_ind) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select directory from which to grab hdf5 files')
            return files                    
        
        invalid_end = ('VID_INFO.hdf5','TRACKING.hdf5', \
                        'TRACKING_PROCESSED.hdf5','COMBINED_DATA.hdf5')
        for ind in selected_ind:
            temp_dir = temp_dirlist[ind]
            for file in os.listdir(temp_dir):
                if file.endswith(".hdf5") and not file.endswith(invalid_end):
                    files.append(os.path.join(temp_dir,file))
                    
        self.datadir_curr = temp_dir
        
        if len(files) > 0:
            return files
        else:
            tkMessageBox.showinfo(title='Error',
                                message='No HDF5 files found.')
            files = []
            return files 
    
    #===================================================================
    @staticmethod        
    def scan_dirs_vid(self):
        # build list of detected files from selected paths
        files = [] 
        temp_dirlist = list(self.dirframe.dirlist.get(0, END))
        selected_ind = sorted(self.dirframe.dirlist.curselection(), reverse=False)
        
        valid_ext = [".avi", ".mov", ".mp4", ".mpg", ".mpeg", \
                        ".rm", ".swf", ".vob", ".wmv"]
        if len(selected_ind) < 1:
            tkMessageBox.showinfo(title='Error',
                message='Please select directory from which to grab video files')
            return files                    
        
        for ind in selected_ind:
            temp_dir = temp_dirlist[ind]
            for file in os.listdir(temp_dir):
                if file.endswith(tuple(valid_ext)):
                    files.append(os.path.join(temp_dir,file))
                    
        self.datadir_curr = temp_dir
        
        if len(files) > 0:
            return files
        else:
            tkMessageBox.showinfo(title='Error',
                                message='No video files found.')
            files = []
            return files                                   
    
    #===================================================================        
    @staticmethod 
    def unpack_files(self):
        selected_ind = sorted(self.fdata_frame.filelist.curselection(), reverse=False)
        #print(selected_ind)
        selected = [] 
        for ind in selected_ind: 
            selected.append(self.fdata_frame.filelist.get(ind))
        
        #temp_dirlist = list(self.dirframe.dirlist.get(0, END))
        #for dir in temp_dirlist:
        fileKeyNames = []
        for filename in selected:
            #filename = os.path.join(dir,selected[0])
            #print(filename)
            if os.path.isfile(filename):
                self.filename_curr = filename 
                f = h5py.File(filename,'r')
                for key in list(f.keys()):
                    if key.startswith('XP'):
                        fileKeyNames.append(filename + ", " + key)
        
        return fileKeyNames    
    
    #===================================================================
    @staticmethod 
    def unpack_xp(self):
        selected_ind = sorted(self.xpdata_frame.xplist.curselection(), reverse=False)
        groupKeyNames = []
        for ind in selected_ind:
            xp_entry = self.xpdata_frame.xplist.get(ind)
            filename, filekeyname = xp_entry.split(', ', 1)
            f = h5py.File(filename,'r')
            #fileKeyNames = list(f.keys())
            grp = f[filekeyname]
            for key in list(grp.keys()):
                dset, _ = load_hdf5(filename,filekeyname,key)        
                dset_check = (dset != -1)
                if (np.sum(dset_check) > 0):
                    groupKeyNames.append(filename + ', ' + filekeyname + 
                                        ', ' + key) 
        return groupKeyNames
    
    #===================================================================
    @staticmethod
    def clear_xplist(self):
        self.xpdata_frame.xplist.delete(0,END)
    
    #===================================================================
    @staticmethod
    def clear_channellist(self):
        self.channeldata_frame.channellist.delete(0,END)
    
    #===================================================================         
    @staticmethod
    def get_channel_data(self,channel_entry,DEBUG_FLAG=False,combFlagArg=False):
        filename, filekeyname, groupkeyname = channel_entry.split(', ',2)
        comb_analysis_flag = self.comb_analysis_flag.get()
        comb_analysis_flag = comb_analysis_flag or combFlagArg
        
        # load data        
        dset, t = load_hdf5(filename,filekeyname,groupkeyname)        
        
        bad_data_flag, dset, t, frames = check_data_set(dset,t)
        
        if not bad_data_flag:
            if comb_analysis_flag:
                dset_smooth, bouts, volumes = bout_analysis_wTracking(filename,
                                                    filekeyname, groupkeyname,
                                                    debugBoutFlag=DEBUG_FLAG)
            else:
                dset_smooth, bouts, volumes = bout_analysis(dset,frames,
                                                        debug_mode=DEBUG_FLAG)
        else:
            dset_smooth = np.array([])
            bouts = np.array([])
            volumes = np.array([])
            print('Problem with loading data set--invalid name')
        
        return (dset, frames, t, dset_smooth, bouts, volumes)
    #===================================================================
    @staticmethod
    def fetch_channels_for_batch(self):
        selected_ind = self.channeldata_frame.selection_ind
        for_batch = []
        if len(selected_ind) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select channels to move to batch')
            return for_batch
        
        for ind in selected_ind: 
            for_batch.append(self.channeldata_frame.channellist.get(ind))
        
        return for_batch
    
    #=================================================================== 
    @staticmethod
    def fetch_videos_for_batch(self):
        selected_ind = self.viddata_frame.selection_ind
        for_batch = []
        if len(selected_ind) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select videos to move to batch')
            return for_batch
        
        for ind in selected_ind: 
            for_batch.append(self.viddata_frame.filelist.get(ind))
        
        return for_batch
    
    #===================================================================    
    @staticmethod
    def fetch_data_for_batch(self):
        selected_vid_ind = self.viddata_frame.selection_ind
        selected_channel_ind = self.channeldata_frame.selection_ind
        for_batch = []
        if (len(selected_vid_ind) < 1) and (len(selected_channel_ind) < 1):
            tkMessageBox.showinfo(title='Error',
                                message='Please select videos or channels to move to batch')
            return for_batch
        
        vid_filenames_full = [self.viddata_frame.filelist.get(ind) for ind in \
                                    selected_vid_ind] 
        channel_filenames_full = [self.channeldata_frame.channellist.get(ind) \
                                    for ind in selected_channel_ind] 
        
        vid_ent_basic = [vid2basic(v_fn) for v_fn in vid_filenames_full]
        ch_ent_basic = [channel2basic(c_fn) for c_fn in channel_filenames_full]
        union_set = set.union(set(vid_ent_basic), set(ch_ent_basic))
        union_list = list(union_set)
        for_batch = union_list           
        
        return for_batch
    #--------------------------------------------------------------------------    
    # define drag and drop functions
    if TKDND_FLAG:
        @staticmethod
        def drop_enter(event):
            event.widget.focus_force()
            #print('Entering widget: %s' % event.widget)
            #print_event_info(event)
            return event.action
        
        @staticmethod
        def drop_position(event):
            #print('Position: x %d, y %d' %(event.x_root, event.y_root))
            #print_event_info(event)
            return event.action
        
        @staticmethod
        def drop_leave(event):
            #print('Leaving %s' % event.widget)
            #print_event_info(event)
            return event.action
        
        # define drag callbacks
        
        @staticmethod
        def drag_init_listbox(event):
            #print_event_info(event)
            # use a tuple as file list, this should hopefully be handled gracefully
            # by tkdnd and the drop targets like file managers or text editors
            data = ()
            if listbox.curselection():
                data = tuple([listbox.get(i) for i in listbox.curselection()])
                print('Dragging :', data)
            # tuples can also be used to specify possible alternatives for
            # action type and DnD type:
            return ((ASK, COPY), (DND_FILES, DND_TEXT), data)
        
        @staticmethod
        def drag_init_text(event):
            #print_event_info(event)
            # use a string if there is only a single text string to be dragged
            data = ''
            sel = text.tag_nextrange(SEL, '1.0')
            if sel:
                data = text.get(*sel)
                print('Dragging :\n', data)
            # if there is only one possible alternative for action and DnD type
            # we can also use strings here
            return (COPY, DND_TEXT, data)
        
        @staticmethod
        def drag_end(event):
            #print_event_info(event)
            # this callback is not really necessary if it doesn't do anything useful
            print('Drag ended for widget:', event.widget)
        
        
        # specific functions for different listboxes
        @staticmethod 
        def file_drop(event):
            if event.data:
                #print('Dropped data:\n', event.data)
                #print_event_info(event)
                
                files = event.widget.tk.splitlist(event.data)
                for f in files:
                    if os.path.exists(f) and f.endswith(".hdf5"):
                        #print('Dropped file: "%s"' % f)
                        event.widget.insert('end', f)
                    else:
                        print('Not dropping file "%s": file does not exist or is invalid.' % f)
                
            return event.action
        
        @staticmethod 
        def vid_file_drop(event):
            if event.data:
                valid_ext = [".avi", ".mov", ".mp4", ".mpg", ".mpeg", \
                            ".rm", ".swf", ".vob", ".wmv"]
                            
                files = event.widget.tk.splitlist(event.data)
                for f in files:
                    if os.path.exists(f) and f.endswith(tuple(valid_ext)):
                        #print('Dropped file: "%s"' % f)
                        event.widget.insert('end', f)
                    else:
                        print('Not dropping file "%s": file does not exist or is invalid.' % f)
                        
        @staticmethod 
        def dir_drop(event):
            if event.data:
                #print('Dropped data:\n', event.data)
                #print_event_info(event)
                
                dirs = event.widget.tk.splitlist(event.data)
                for d in dirs:
                    if os.path.isdir(d):
                        #print('Dropped folder: "%s"' % d)
                        event.widget.insert('end', d)
                    else:
                        print('Not dropping folder "%s": folder does not exist or is invalid.' % d)
                
            return event.action    
    #----------------------------------------------------------------------
    
    #@staticmethod
    #def get_batch_data(self):
#def main():
#    root = Tk()
#    root.geometry("300x280+300+300")
#    app = Expresso(root)
#    root.mainloop()
        
""" Run main loop """
if __name__ == '__main__':
    if TKDND_FLAG:
        root = TkinterDnD.Tk()
    else:
        root = Tk()
    #root = Toplevel()
    Expresso(root)
    root.mainloop()
    #main()
    
"""   
    root = Tk()
    Expresso(root)
    root.mainloop()
"""
