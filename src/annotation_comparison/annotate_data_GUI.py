# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:11:42 2018

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
#from scipy import interpolate

#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider
#from matplotlib.widgets import MultiCursor
#from matplotlib.figure import Figure

from load_hdf5_data import load_hdf5
from bout_analysis_func import bout_analysis
from v_expresso_gui_params import (initDirectories, guiParams, trackingParams)
from annotate_data_func import annotate_channel_data

#from PIL import ImageTk, Image

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
            self.dirlist.dnd_bind('<<DropEnter>>', Expresso_Annotation.drop_enter)
            self.dirlist.dnd_bind('<<DropPosition>>', Expresso_Annotation.drop_position)
            self.dirlist.dnd_bind('<<DropLeave>>', Expresso_Annotation.drop_leave)
            self.dirlist.dnd_bind('<<Drop>>', Expresso_Annotation.dir_drop)
            self.dirlist.dnd_bind('<<Drop:DND_Files>>', Expresso_Annotation.dir_drop)
            self.dirlist.dnd_bind('<<Drop:DND_Text>>', Expresso_Annotation.dir_drop)
            
            self.dirlist.drag_source_register(1, DND_TEXT, DND_FILES)
            #text.drag_source_register(3, DND_TEXT)
    
        self.dirlist.dnd_bind('<<DragInitCmd>>', Expresso_Annotation.drag_init_listbox)
        self.dirlist.dnd_bind('<<DragEndCmd>>', Expresso_Annotation.drag_end)
        
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
        newdir = Expresso_Annotation.get_dir(parent)
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
            self.filelist.dnd_bind('<<DropEnter>>', Expresso_Annotation.drop_enter)
            self.filelist.dnd_bind('<<DropPosition>>', Expresso_Annotation.drop_position)
            self.filelist.dnd_bind('<<DropLeave>>', Expresso_Annotation.drop_leave)
            self.filelist.dnd_bind('<<Drop>>', Expresso_Annotation.file_drop)
            self.filelist.dnd_bind('<<Drop:DND_Files>>', Expresso_Annotation.file_drop)
            self.filelist.dnd_bind('<<Drop:DND_Text>>', Expresso_Annotation.file_drop)
            
            self.filelist.drag_source_register(1, DND_TEXT, DND_FILES)
            #text.drag_source_register(3, DND_TEXT)
        
            self.filelist.dnd_bind('<<DragInitCmd>>', Expresso_Annotation.drag_init_listbox)
            self.filelist.dnd_bind('<<DragEndCmd>>', Expresso_Annotation.drag_end)
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
        newfiles = Expresso_Annotation.scan_dirs(parent)
        file_list = self.filelist.get(0,END)
        if len(newfiles) > 0:
            #file_list = self.filelist.get(0,END)
            #Expresso_Annotation.clear_xplist(parent)
            #Expresso_Annotation.clear_channellist(parent)
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
        newxp = Expresso_Annotation.unpack_files(parent)
        
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
                                
        self.plot_button = Button(self.btnframe, text='Annotate Channel')
        self.plot_button['state'] = 'disabled'
        self.plot_button['command'] = lambda: self.annotate_channel(parent)
        self.plot_button.grid(column=col, row=row+4, padx=10, pady=2,
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
            #self.save_button.configure(state=NORMAL)
            self.selection_ind = sorted(self.channellist.curselection(), 
                                        reverse=False)
            #print(self.selection_ind)
        else:
            self.remove_button.configure(state=DISABLED)
            self.plot_button.configure(state=DISABLED)
            #self.save_button.configure(state=DISABLED)
            
            
    def add_channels(self,parent):
        channel_list = self.channellist.get(0,END)
        newchannels = Expresso_Annotation.unpack_xp(parent)
        for channel in tuple(newchannels):
            if channel not in channel_list:
                self.channellist.insert(END,channel)
    
    def rm_channel(self):
        selected = sorted(self.channellist.curselection(), reverse=True)
        for item in selected:
            self.channellist.delete(item)
    
    def clear_channel(self):
        self.channellist.delete(0,END)        
    
    def annotate_channel(self,parent):
        selected_ind = self.selection_ind
        if len(selected_ind) != 1:
            tkMessageBox.showinfo(title='Error',
                                message='Please select only one channel for plotting individual traces')
            return 
        
        channel_entry = self.channellist.get(selected_ind[0])
        data_filename, bank_name, channel_name = Expresso_Annotation.get_channel_info(parent,channel_entry) 
        
        self.multi, self.cid, self.pid = annotate_channel_data(data_filename, 
                                                       bank_name, channel_name)
#        self.multi = MultiCursor(self.fig.canvas, (self.ax1, self.ax2), color='cyan', lw=1.0, useblit=True,
#                        horizOn=True,  vertOn=True)
#        self.multi.active = True
        
#------------------------------------------------------------------------------

class Expresso_Annotation:
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
        
        # run gui presets. may be unecessary
        self.init_gui()
        
        # initialize instances of frames created above
        self.dirframe = DirectoryFrame(self, col=0, row=0)
        self.fdata_frame = FileDataFrame(self, col=0, row=1)
        self.xpdata_frame = XPDataFrame(self, col=0, row=2)
        self.channeldata_frame = ChannelDataFrame(self, col=0, row=3)

        
        for datadir in self.initdirs:
            self.dirframe.dirlist.insert(END, datadir)
        
        #self.rawdata_plot = FigureFrame(self, col=4, row=0)
        
        #self.make_topmost()
        self.master.protocol("WM_DELETE_WINDOW", self.on_quit)
        
    def on_quit(self):
        """Exits program."""
        if tkMessageBox.askokcancel("Quit","Do you want to quit?"):
            self.master.destroy()
            self.master.quit()
    
    def make_topmost(self):
        """Makes this window the topmost window"""
        self.master.lift()
        self.master.attributes("-topmost", 1)
        self.master.attributes("-topmost", 0) 
        
    def init_gui(self):
        """Label for GUI"""
        
        self.master.title('Expresso Data Annotation')
        
        """ Menu bar """
        self.master.option_add('*tearOff', 'FALSE')
        self.master.menubar = Menu(self.master)
 
        self.master.menu_file = Menu(self.master.menubar)
        self.master.menu_file.add_command(label='Exit', command=self.on_quit)
 
        self.master.menu_edit = Menu(self.master.menubar)
 
        self.master.menubar.add_cascade(menu=self.master.menu_file, label='File')
        self.master.menubar.add_cascade(menu=self.master.menu_edit, label='Edit')
 
        self.master.config(menu=self.master.menubar)
        
        #self.master.config(background='white')
        
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
            return sorted(files)
        else:
            tkMessageBox.showinfo(title='Error',
                                message='No HDF5 files found.')
            files = []
            return files 

            
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
    
    @staticmethod
    def clear_xplist(self):
        self.xpdata_frame.xplist.delete(0,END)
    
    @staticmethod
    def clear_channellist(self):
        self.channeldata_frame.channellist.delete(0,END)
             
    @staticmethod
    def get_channel_info(self,channel_entry):
        filename, filekeyname, groupkeyname = channel_entry.split(', ',2)
        return (filename, filekeyname, groupkeyname)       
        

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
                print('Dropped data:\n', event.data)
                #print_event_info(event)
                
                files = event.widget.tk.splitlist(event.data)
                for f in files:
                    if os.path.exists(f) and f.endswith(".hdf5"):
                        print('Dropped file: "%s"' % f)
                        event.widget.insert('end', f)
                    else:
                        print('Not dropping file "%s": file does not exist or is invalid.' % f)
                
            return event.action
        
                        
        @staticmethod 
        def dir_drop(event):
            if event.data:
                print('Dropped data:\n', event.data)
                #print_event_info(event)
                
                dirs = event.widget.tk.splitlist(event.data)
                for d in dirs:
                    if os.path.isdir(d):
                        print('Dropped folder: "%s"' % d)
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
    #root = Toplevel()
    Expresso_Annotation(root)
    root.mainloop()
    #main()
    
"""   
    root = Tk()
    Expresso(root)
    root.mainloop()
"""

