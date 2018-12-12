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
from bout_analysis_func import bout_analysis
from batch_bout_analysis_func import batch_bout_analysis, save_batch_xlsx
from v_expresso_gui_params import (initDirectories, guiParams, trackingParams)
from v_expresso_image_lib_mk2 import (visual_expresso_main, 
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
            (dset,frames)
            self.bouts = bouts
            self.dset_smooth = dset_smooth
            self.volumes  = volumes
            self.t = t 
            #fig_window = Toplevel()
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, sharex=True, 
                                                sharey=True,figsize=(12, 7))
            
            self.ax1.set_ylabel('Liquid [nL]')
            self.ax2.set_ylabel('Liquid [nL]')
            self.ax2.set_xlabel('Time [s]')
            self.ax1.set_title('Raw Data')
            self.ax2.set_title('Smoothed Data')
            
            self.ax1.plot(t,dset)
            self.ax2.plot(t, dset_smooth)
            for i in np.arange(bouts.shape[1]):
                self.ax2.plot(t[bouts[0,i]:bouts[1,i]], dset_smooth[bouts[0,i]:bouts[1,i]],'r-')
                self.ax2.axvspan(t[bouts[0,i]],t[bouts[1,i]-1], 
                                 facecolor='grey', edgecolor='none', alpha=0.3)
                self.ax1.axvspan(t[bouts[0,i]],t[bouts[1,i]-1], 
                                 facecolor='grey', edgecolor='none', alpha=0.3)
                
            self.ax1.set_xlim([t[0],t[-1]])
            self.ax1.set_ylim([np.amin(dset),np.amax(dset)])    
                
            #self.fig.set_tight_layout(True)
               
            plt.subplots_adjust(bottom=0.2)
            self.ax_xrange = plt.axes([0.25, 0.1, 0.65, 0.03])
            self.ax_xmid = plt.axes([0.25, 0.06, 0.65, 0.03])
            
            self.multi = MultiCursor(self.fig.canvas, (self.ax1, self.ax2),
                                     color='dodgerblue', lw=1.0, useblit=True,
                                     horizOn=True, vertOn=True)
                        
            self.slider_xrange = Slider(self.ax_xrange, 't range', 
                                        -1.0*np.amax(self.t), -1.0*3.0, 
                                        valinit=-1.0*np.amax(self.t))
            self.slider_xmid = Slider(self.ax_xmid, 't mid', 0.0, 
                                        np.amax(self.t), 
                                        valinit=np.amax(self.t)/2,
                                        facecolor='white')
            
            def update(val):
                xrange_val = -1.0*self.slider_xrange.val
                xmid_val = self.slider_xmid.val
                xmin = int(np.rint(np.amax([0, xmid_val - xrange_val/2])))
                xmax = int(np.rint(np.amin([self.dset_smooth.size, xmid_val + xrange_val/2])))
                xlim = [xmin, xmax]
                ymin = self.dset_smooth[xmin]+1
                ymax = self.dset_smooth[xmax-1]-1
                ylim = np.sort([ymin, ymax])
                self.ax2.set_xlim(xlim)
                self.ax2.set_ylim(ylim)
                #ax2_lim = ((xmin, np.amin(self.dset_smooth)), (xmax,np.amax(self.dset_smooth)))
                #self.ax2.update_datalim(ax2_lim)
                #self.ax2.autoscale()
                self.fig.canvas.draw_idle()
            self.slider_xrange.on_changed(update)
            self.slider_xmid.on_changed(update)
            
            
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
        file_entry = self.filelist.get(selected_ind[0])
        file_path, filename = os.path.split(file_entry)
        self.flyTrackData = visual_expresso_main(file_path, filename, 
                            DEBUG_BG_FLAG=menu_debug_flag, 
                            DEBUG_CM_FLAG=menu_debug_flag, 
                            SAVE_DATA_FLAG=True, ELLIPSE_FIT_FLAG = False, 
                            PARAMS=trackingParams)
                            
        filename_prefix = os.path.splitext(filename)[0]
        track_filename = filename_prefix + "_TRACKING.hdf5"  
                  
        self.flyTrackData_smooth = process_visual_expresso(file_path, track_filename,
                            SAVE_DATA_FLAG = True, DEBUG_FLAG = False)
                            
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
            
            if self.vid_filepath == file_entry:
                plot_body_cm(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag)
                plot_body_vel(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag) 
                plot_moving_v_still(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag)
                plot_cum_dist(self.flyTrackData_smooth,SAVE_FLAG=menu_save_flag)
            elif os.path.exists(save_filename):
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
             self.fig_hist) = batch_bout_analysis(batch_list, tmin, tmax, tbin,True)
             
             #self.save_button['state'] = 'enabled'   
    
    def save_batch(self):
        batch_list = self.batchlist.get(0,END)
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
             batch_bout_analysis(batch_list, tmin, tmax, tbin,False)
            
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
                                message='Add data to batch box for batch analysis')
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
                                message='Add data to batch box for batch analysis')
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
                                
        self.analyze_button = Button(self.entryframe, text='Analyze/Save Video(s)')
        #self.plot_button['state'] = 'disabled'
        self.analyze_button['command'] = self.analyze_batch
        self.analyze_button.grid(column=col+3, row=row, padx=10, pady=2,
                                sticky=S) 
        
        self.save_button = Button(self.entryframe, text='Save Video Batch Summary')
        #self.save_button['state'] = 'disabled'
        self.save_button['command'] = self.save_batch
        self.save_button.grid(column=col+3, row=row+1, padx=10, pady=2,
                                sticky=S)                        
        
        self.save_ts_button = Button(self.entryframe, text='Save Video Time Series')
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
        for_batch = Expresso.fetch_data_for_batch(root)
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
               visual_expresso_main(file_path, filename, 
                                DEBUG_BG_FLAG=False, DEBUG_CM_FLAG=False, 
                                SAVE_DATA_FLAG=True, ELLIPSE_FIT_FLAG = False, 
                                PARAMS=trackingParams)
               filename_prefix = os.path.splitext(filename)[0]
               track_filename = filename_prefix + "_TRACKING.hdf5"
               process_visual_expresso(file_path, track_filename,
                                SAVE_DATA_FLAG = True, DEBUG_FLAG = False)
        self.vid_file_list = batch_list
               
    def save_batch(self):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
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
                                message='Add data to batch box for batch analysis')
            return 
        else:
            save_vid_time_series(batch_list)

#------------------------------------------------------------------------------

class ExtraButtonsFrame(Frame):
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)
        
        self.btnframe = Frame(parent.master)    
        self.sync_select_var = IntVar()
        self.sync_select_checkbox = Checkbutton(self.btnframe, text='Synchronize Selection',
                                                variable = self.sync_select_var)
                                                
        self.sync_select_checkbox.grid(column=col, row=row, padx=10, pady=2,
                                sticky=NE)   
                                
        self.scan_btn =Button(self.btnframe, text='TEST',
                                        command= lambda: self.temp_callback())
        #self.scan_btn['state'] = 'disabled'                                
        self.scan_btn.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)
                                
    def temp_callback(self):
        print('click!')
#------------------------------------------------------------------------------

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
        
        # debugging boolean variables
        self.debug_tracking= IntVar()
        self.debug_tracking.set(0)
        
        self.debug_bout = IntVar()
        self.debug_bout.set(0)
        
        self.save_all_plots = IntVar()
        self.save_all_plots.set(0)
        
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
        """Turns on or off the tracking debug flags"""
        curr_val = self.debug_bout.get()
        if curr_val == True:
            self.debug_bout.set(0)
        elif curr_val == False:
            self.debug_bout.set(1)
    
    #===================================================================
    def toggle_save_all(self):
        """Turns on or off the tracking debug flags"""
        curr_val = self.save_all_plots.get()
        if curr_val == True:
            self.save_all_plots.set(0)
        elif curr_val == False:
            self.save_all_plots.set(1)
    
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
        
        self.master.menu_file = Menu(self.master.menubar)
        self.master.menu_file.add_command(label='Exit', command=self.on_quit)
 
        self.master.menu_debug = Menu(self.master.menubar)
        self.master.menu_debug.add_checkbutton(label='Save All Plots', 
                                               variable=self.save_all_plots,
                                               command=self.toggle_save_all)
        self.master.menu_debug.add_checkbutton(label='Bout Detection Debug', 
                                               variable=self.debug_bout,
                                               command=self.toggle_bout_debug)
        self.master.menu_debug.add_checkbutton(label='Tracking Debug', 
                                               variable=self.debug_tracking,
                                               command = self.toggle_track_debug)
        
        self.master.menubar.add_cascade(menu=self.master.menu_file, label='File')
        self.master.menubar.add_cascade(menu=self.master.menu_debug, 
                                        label='Debugging Options')
 
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
        
        for ind in selected_ind:
            temp_dir = temp_dirlist[ind]
            for file in os.listdir(temp_dir):
                if file.endswith(".hdf5"):
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
    def get_channel_data(self,channel_entry,DEBUG_FLAG=False):
        filename, filekeyname, groupkeyname = channel_entry.split(', ',2)
        dset, t = load_hdf5(filename,filekeyname,groupkeyname)        
        
        dset_check = (dset != -1)
        if (np.sum(dset_check) == 0):
            dset = np.array([])
            frames = np.array([])
            t = np.array([])
            dset_smooth = np.array([])
            bouts = np.array([])
            volumes = np.array([])
            print('Problem with loading data - invalid data set')
            return (dset, frames, t, dset_smooth, bouts, volumes)    
            
        frames = np.arange(0,dset.size)
        
        dset = dset[dset_check]
        frames = frames[np.squeeze(dset_check)]
        t = t[dset_check]
        
        new_frames = np.arange(0,np.max(frames)+1)
        sp_raw = interpolate.InterpolatedUnivariateSpline(frames, dset)
        sp_t = interpolate.InterpolatedUnivariateSpline(frames, t)
        dset = sp_raw(new_frames)
        t = sp_t(new_frames)
        frames = new_frames
        
        try:
            dset_smooth, bouts, volumes = bout_analysis(dset,frames,
                                                        debug_mode=DEBUG_FLAG)
            return (dset, frames, t, dset_smooth, bouts, volumes)
        except NameError:
            dset = np.array([])
            frames = np.array([])
            t = np.array([])
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
                                    for ind in selected_vid_ind] 
        
        vid_files = [os.path.splitext(v_fn)[0] for v_fn in vid_filenames_full]
        channel_files = [os.path.splitext(c_fn)[0] for c_fn in channel_filenames_full]
        intersect_files_set = set.intersection(set(vid_files), set(channel_files))
        intersect_files_list = list(intersect_files_set)
        for_batch = intersect_files_list
#            vid_filename_full = self.viddata_frame.filelist.get(ind)
#            vid_file_path, vid_filename = os.path.split(vid_filename_full)
#            vid_data_prefix = os.path.splitext(vid_filename)[0]
#            for_batch.append(self.viddata_frame.filelist.get(ind))
        
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
