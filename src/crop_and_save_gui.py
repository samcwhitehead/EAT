# -*- coding: utf-8 -*-
"""
GUI interface for cropping videos from the visual expresso system. videos are 
cropped to ROIs containing a single channel to reduce computation time later in
the analysis pipeline. This GUI also prompts the user to manually select the 
location of the capillary tip for each channel--the tip becomes the (0,0) 
position for the tracked coordinates

Created on Mon May 07 20:19:33 2018

@author: Fruit Flies

GUI for (hopefully) more convenient pre-processing of video data
"""

import matplotlib
matplotlib.use('TkAgg')

import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
import csv
import re

from v_expresso_gui_params import initDirectories 
from v_expresso_pre_process import roi_and_cap_tip_single_video
from v_expresso_pre_process import crop_and_save_single_video

from refine_tip_estimation import refine_tip 

# allows drag and drop functionality. if you don't want this, or are having
#  trouble with the TkDnD installation, set to false.
try:
    from TkinterDnD2 import *
    from gui_setup_util import (buildButtonListboxPanel, buildBatchPanel, bindToTkDnD, myEntryOptions)

    TKDND_FLAG = True
except ImportError:
    print('Error: could not load TkDnD libraries. Drag/drop disabled')
    from gui_setup_util import buildButtonListboxPanel, buildBatchPanel, myEntryOptions
#------------------------------------------------------------------------------

class DirectoryFrame(Frame):
    """ Top UI frame containing the list of directories to be scanned. """

    def __init__(self, parent, col=0, row=0, filedir=None):
        Frame.__init__(self, parent.master)
                           
        self.lib_label = Label(parent.master, text='Directory list:') 
                               #foreground=guiParams['textcolor'], 
                               #background=guiParams['bgcolor'])
        self.lib_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)

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
            self.dirlist.dnd_bind('<<DropEnter>>', VExpressoPreProcessing.drop_enter)
            self.dirlist.dnd_bind('<<DropPosition>>', VExpressoPreProcessing.drop_position)
            self.dirlist.dnd_bind('<<DropLeave>>', VExpressoPreProcessing.drop_leave)
            self.dirlist.dnd_bind('<<Drop>>', VExpressoPreProcessing.dir_drop)
            self.dirlist.dnd_bind('<<Drop:DND_Files>>', VExpressoPreProcessing.dir_drop)
            self.dirlist.dnd_bind('<<Drop:DND_Text>>', VExpressoPreProcessing.dir_drop)
            
            self.dirlist.drag_source_register(1, DND_TEXT, DND_FILES)
            #text.drag_source_register(3, DND_TEXT)
        
            self.dirlist.dnd_bind('<<DragInitCmd>>', VExpressoPreProcessing.drag_init_listbox)
            self.dirlist.dnd_bind('<<DragEndCmd>>', VExpressoPreProcessing.drag_end)
        
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
        
       

        self.btnframe = Frame(parent.master)
        self.btnframe.grid(column=col+2, row=row, sticky=NW)
        #self.btnframe.config(background=guiParams['bgcolor'])
        
        self.lib_addbutton =Button(self.btnframe, text='Add Directory',
                                   command= lambda: self.add_library(parent))
        self.lib_addbutton.grid(column=col, row=row, padx=10, pady=2,
                                sticky=NW)
        
        self.lib_delbutton = Button(self.btnframe, text='Remove Directory',
                                  command=self.rm_library, state=DISABLED)
                                   
        self.lib_delbutton.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)
        
        self.lib_clearbutton = Button(self.btnframe, text='Clear All',
                                  command=self.clear_library)
                                   
        self.lib_clearbutton.grid(column=col, row=row+2, padx=10, pady=2,
                                sticky=NW)                        

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
        newdir = VExpressoPreProcessing.get_dir(parent)
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
        
        

        self.list_label = Label(parent.master, text='Detected files:')
        self.list_label.grid(column=col, row=row, padx=10, pady=2, sticky=NW)

        self.filelistframe = Frame(parent.master) 
        self.filelistframe.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
        self.filelist = Listbox(self.filelistframe,  width=64, height=8, 
                                selectmode=EXTENDED)
        
        #now make the Listbox and Text drop targets
        if TKDND_FLAG:
            self.filelist.drop_target_register(DND_FILES, DND_TEXT)
        
        self.filelist.bind('<<ListboxSelect>>', self.on_select)
        
        if TKDND_FLAG:
            self.filelist.dnd_bind('<<DropEnter>>', VExpressoPreProcessing.drop_enter)
            self.filelist.dnd_bind('<<DropPosition>>', VExpressoPreProcessing.drop_position)
            self.filelist.dnd_bind('<<DropLeave>>', VExpressoPreProcessing.drop_leave)
            self.filelist.dnd_bind('<<Drop>>', VExpressoPreProcessing.file_drop)
            self.filelist.dnd_bind('<<Drop:DND_Files>>', VExpressoPreProcessing.file_drop)
            self.filelist.dnd_bind('<<Drop:DND_Text>>', VExpressoPreProcessing.file_drop)
            
            self.filelist.drag_source_register(1, DND_TEXT, DND_FILES)
            #text.drag_source_register(3, DND_TEXT)
        
            self.filelist.dnd_bind('<<DragInitCmd>>', VExpressoPreProcessing.drag_init_listbox)
            self.filelist.dnd_bind('<<DragEndCmd>>', VExpressoPreProcessing.drag_end)
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

        self.btnframe = Frame(parent.master)
        self.btnframe.grid(column=col+2, row=row, sticky=NW)
        
        # button used to initiate the scan of the above directories
        self.scan_btn =Button(self.btnframe, text='Get Video Files',
                                        command= lambda: self.add_files(parent))
        #self.scan_btn['state'] = 'disabled'                                
        self.scan_btn.grid(column=col, row=row, padx=10, pady=2,
                                sticky=NW)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Files')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_files
        self.remove_button.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NW)
                                
        self.clear_button = Button(self.btnframe, text='Clear All')
        self.clear_button['command'] = self.clear_files
        self.clear_button.grid(column=col, row=row+2, padx=10, pady=2,
                                sticky=NW)                        
        
        self.selection_ind = [] 
        
    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.filelist.curselection():
            self.remove_button.configure(state=NORMAL)
            self.selection_ind = sorted(self.filelist.curselection(), 
                                        reverse=False)
        else:
            self.remove_button.configure(state=DISABLED)
            
    def add_files(self,parent):
        newfiles = VExpressoPreProcessing.scan_dirs(parent)
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

#------------------------------------------------------------------------------


class BatchFrame(Frame):
    def __init__(self, parent,col=0,row=0):
        Frame.__init__(self, parent.master)
        
        self.list_label = Label(parent.master, text='Batch analyze list:')
        self.list_label.grid(column=col, row=row, padx=10, pady=5, sticky=NW)
        
        self.batchlistframe = Frame(parent.master) 
        self.batchlistframe.grid(column=col, row=row, columnspan = 2, rowspan = 3, padx=10, pady=30, sticky=N)
        
        self.batchlist = Listbox(self.batchlistframe,  width=64, height=12,
                                   selectmode=EXTENDED)
        
        self.batchlist.bind('<<ListboxSelect>>', self.on_select)
        #self.filelist.grid(column=col+1, row=row, padx=10, pady=2, sticky=W)
        
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
        
        self.entryframe = Frame(parent.master)
        self.entryframe.grid(column=col, row=row+1,columnspan = 2, rowspan = 3, 
                             pady=100, sticky=W)
        
        self.bank1_entry_label = Label(self.entryframe, text='Bank 1 Name:')
        self.bank1_entry_label.grid(column=col, row=0, padx=10, pady=2, sticky=N)
        self.bank1_entry = Entry(self.entryframe, width=8)
        self.bank1_entry.insert(END,'XP04')
        self.bank1_entry.grid(column=col, row=1,padx=10, pady=2, sticky=S)
        
        self.bank2_entry_label = Label(self.entryframe, text='Bank 2 Name:')
        self.bank2_entry_label.grid(column=col+1, row=0, padx=10, pady=2, sticky=N)
        self.bank2_entry = Entry(self.entryframe, width=8)
        self.bank2_entry.insert(END,'XP05')
        self.bank2_entry.grid(column=col+1, row=1,padx=10, pady=2, sticky=S)
        
        self.bank1_channel_entry_label = Label(self.entryframe, text='Bank 1 Channels:')
        self.bank1_channel_entry_label.grid(column=col, row=2, padx=10, pady=2, sticky=N)
        self.bank1_channel_entry = Entry(self.entryframe, width=8)
        self.bank1_channel_entry.insert(END,'1,2,3,4,5')
        self.bank1_channel_entry.grid(column=col, row=3,padx=10, pady=2, sticky=S)
        
        self.bank2_channel_entry_label = Label(self.entryframe, text='Bank 2 Channels:')
        self.bank2_channel_entry_label.grid(column=col+1, row=2, padx=10, pady=2, sticky=N)
        self.bank2_channel_entry = Entry(self.entryframe, width=8)
        self.bank2_channel_entry.insert(END,'1,2,3,4,5')
        self.bank2_channel_entry.grid(column=col+1, row=3,padx=10, pady=2, sticky=S)                    
        
        
        self.btnframe = Frame(parent.master)
        self.btnframe.grid(column=col+1, row=row+1, columnspan = 1, rowspan = 3, 
                           pady=100, sticky=N)
        
        # button used to initiate the scan of the above directories
        self.add_btn =Button(self.btnframe, text='Add Video(s) to Batch',
                                        command= lambda: self.add_to_batch(parent))
        self.add_btn.grid(column=col, row=row, padx=10, pady=2,
                                sticky=N)
                                
        # button used to remove the detected data
        self.remove_button = Button(self.btnframe, text='Remove Selected')
        self.remove_button['state'] = 'disabled'
        self.remove_button['command'] = self.rm_channel
        self.remove_button.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=N)
        
        # button used to clear all batch files
        self.clear_button = Button(self.btnframe, text='Clear All')
        self.clear_button['command'] = self.clear_batch
        self.clear_button.grid(column=col, row=row+2, padx=10, pady=2,
                                sticky=N)
        
        # button used to process videos                       
        self.process_button = Button(self.btnframe, text='Process Videos')
        #self.plot_button['state'] = 'disabled'
        self.process_button['command'] = self.process_batch
        self.process_button.grid(column=col, row=row+3, padx=10, pady=2,
                                sticky=S) 
                                
        # button used to refine tip estimate           
        self.retip_button = Button(self.btnframe, text='Refine Tip Estimate')
        #self.plot_button['state'] = 'disabled'
        self.retip_button['command'] = self.redo_tip
        self.retip_button.grid(column=col, row=row+4, padx=10, pady=5,
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
            
    def add_to_batch(self,parent):
        batch_list = self.batchlist.get(0,END)
        for_batch = VExpressoPreProcessing.fetch_videos_for_batch(parent)
        for channel in tuple(for_batch):
            if channel not in batch_list:
                self.batchlist.insert(END,channel)
    
    def rm_channel(self):
        selected = sorted(self.batchlist.curselection(), reverse=True)
        for item in selected:
            self.batchlist.delete(item)
    
    def clear_batch(self):
        self.batchlist.delete(0,END)         
        
    def process_batch(self):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            try:
                bank1_name = str(self.bank1_entry.get())
                bank2_name = str(self.bank2_entry.get())
                bank1_channels = list(map(int, re.findall(r'\d+', 
                                        str(self.bank1_channel_entry.get()))))
                bank2_channels = list(map(int, re.findall(r'\d+', 
                                        str(self.bank2_channel_entry.get()))))
                
                bank_names = [bank1_name,bank2_name]
                channel_numbers = [bank1_channels,bank2_channels]
                self.bank_names = bank_names 
                self.channel_numbers = channel_numbers 
                
            except:
                tkMessageBox.showinfo(title='Error',
                    message='Specify bank names and channel numbers for each bank')
                return                
            
            for fn in batch_list:
                dirpath_curr, filename_curr = os.path.split(fn)
                data_prefix_curr = os.path.splitext(filename_curr)[0]
                vid_info_filename = os.path.join(dirpath_curr,
                                                 data_prefix_curr + "_VID_INFO.hdf5")
                
                if os.path.exists(vid_info_filename):
                    print('Already obtained video info for:')
                    print(vid_info_filename)
                    continue
                else:
                    roi_and_cap_tip_single_video(fn, bank_names, channel_numbers)
            
            for fn in batch_list:
                crop_and_save_single_video(fn, bank_names, channel_numbers)
            
            print('Processing completed!')
             #self.save_button['state'] = 'enabled'   
    def redo_tip(self):
        batch_list = self.batchlist.get(0,END)
        if len(batch_list) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Add data to batch box for batch analysis')
            return 
        else:
            for fn in batch_list:
                # convert avi filename into processed hdf5 name
                dirpath_curr, filename_curr = os.path.split(fn)
                data_prefix_curr = os.path.splitext(filename_curr)[0]
                hdf5_name = data_prefix_curr + "_TRACKING_PROCESSED.hdf5"
                vid_data_filename = os.path.join(dirpath_curr,hdf5_name)
                
                # check if analysis exists, if so refine tip
                if os.path.exists(vid_data_filename):
                    refine_tip(vid_data_filename)
                else:
                    print('Need to run analysis on {}'.format(vid_data_filename))
                    continue
        
    
    
#------------------------------------------------------------------------------
                
class VExpressoPreProcessing:
    """The GUI and functions."""
    def __init__(self, master):
        #Tk.__init__(self)
        self.master = master        
        
        # initialize important fields for retaining where we are in data space
        self.initdirs = [] 
        init_dirs = initDirectories 
        
        for init_dir in init_dirs:        
            if os.path.exists(init_dir):
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
        
        # run gui presets. may be unecessary
        self.init_gui()
        
        # initialize instances of frames created above
        self.dirframe = DirectoryFrame(self, col=0, row=0)
        self.fdata_frame = FileDataFrame(self, col=0, row=1)
       
        self.batchdata_frame = BatchFrame(self,col=3,row=0)
        #self.logo_frame = LogoFrame(self,col=0,row=0)
        
        for datadir in self.initdirs:
            self.dirframe.dirlist.insert(END, datadir)
        
        #self.rawdata_plot = FigureFrame(self, col=4, row=0)
        
        self.make_topmost()
        self.master.protocol("WM_DELETE_WINDOW", self.on_quit)
        
    def on_quit(self):
        """Exits program."""
        if tkMessageBox.askokcancel("Quit","Do you want to quit?"):
            self.master.destroy()
            self.master.quit()
    
    def on_open_new_gui(self):
        """Opens visual_expresso_gui_main.py' and exits program."""
        if tkMessageBox.askokcancel("Open main GUI","Do you want to quit?"):
            self.master.destroy()
            self.master.quit()
            # run script
            os.system("python visual_expresso_gui_main.py") 
            
    
    def make_topmost(self):
        """Makes this window the topmost window"""
        self.master.lift()
        self.master.attributes("-topmost", 1)
        self.master.attributes("-topmost", 0) 
        
    def init_gui(self):
        """Label for GUI"""
        
        self.master.title('Visual Expresso video pre-processing')
        
        """ Menu bar """
        self.master.option_add('*tearOff', 'FALSE')
        self.master.menubar = Menu(self.master)
        
        self.master.menu_file = Menu(self.master.menubar)
        self.master.menu_file.add_command(label='Open Main GUI', 
                                          command=self.on_open_new_gui)
        self.master.menu_file.add_command(label='Exit', command=self.on_quit)
 
        self.master.menu_edit = Menu(self.master.menubar)
 
        self.master.menubar.add_cascade(menu=self.master.menu_file, label='File')
        self.master.menubar.add_cascade(menu=self.master.menu_edit, label='Edit')
 
        self.master.config(menu=self.master.menubar)
        
        """ 
        parent.title('Expresso Data Analysis (rough version)')
        
        parent.option_add('*tearOff', 'FALSE')
        parent.menubar = Menu(parent)
 
        parent.menu_file = Menu(parent.menubar)
        parent.menu_file.add_command(label='Exit', command=self.on_quit)
 
        parent.menu_edit = Menu(parent.menubar)
 
        parent.menubar.add_cascade(menu=parent.menu_file, label='File')
        parent.menubar.add_cascade(menu=parent.menu_edit, label='Edit')
 
        parent.config(menu=parent.menubar)
        #self.configure(background='dim gray')
        #self.tk_setPalette(background=guiParams['bgcolor'],
        #                   foreground=guiParams['textcolor']) 
        """                   
        
    @staticmethod
    def get_dir(self):
        """ Method to return the directory selected by the user which should
            be scanned by the application. """

        # get user specified directory and normalize path
        seldir = tkFileDialog.askdirectory(initialdir=sys.path[0])
        if seldir:
            seldir = os.path.abspath(seldir)
            self.datadir_curr = seldir
            return seldir
    
    @staticmethod        
    def scan_dirs(self):
        # build list of detected files from selected paths
        files = [] 
        temp_dirlist = list(self.dirframe.dirlist.get(0, END))
        selected_ind = sorted(self.dirframe.dirlist.curselection(), reverse=False)
        
        valid_ext = [".avi", ".mov", ".mp4", ".mpg", ".mpeg", \
                        ".rm", ".swf", ".vob", ".wmv"]
        if len(selected_ind) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select directory from which to grab hdf5 files')
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
            
    
    @staticmethod
    def fetch_videos_for_batch(self):
        selected_ind = self.fdata_frame.selection_ind
        for_batch = []
        if len(selected_ind) < 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select videos to move to batch')
            return for_batch
        
        for ind in selected_ind: 
            for_batch.append(self.fdata_frame.filelist.get(ind))
        
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
            print_event_info(event)
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
            print_event_info(event)
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
                
                valid_ext = [".avi", ".mov", ".mp4", ".mpg", ".mpeg", \
                            ".rm", ".swf", ".vob", ".wmv"]
                            
                files = event.widget.tk.splitlist(event.data)
                for f in files:
                    if os.path.exists(f) and f.endswith(tuple(valid_ext)):
                        #print('Dropped file: "%s"' % f)
                        event.widget.insert('end', f)
                    else:
                        print('Not dropping file "%s": file does not exist or is invalid.' % f)
                
            return event.action
            
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
    root = TkinterDnD.Tk()
    Expresso_PreProc = VExpressoPreProcessing(root)
    root.mainloop()
    #main()       
        