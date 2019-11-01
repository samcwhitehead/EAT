# -*- coding: utf-8 -*-
"""
GUI to use for post-processing Visual Expresso data

TO DO:
    -Expand functionality to include data from both tracking and channel data 
     (current version just does channel data). To do this, it would be nice to 
     add a function to the main gui that saves a comprehensive summary file
    -Expand list of stats tests and plot types (latter is related to above)
    -Need to (overall) switch everything to csv. this csv and xlsx thing is bad
    -general aesthetic update
    -add in ways to save plots and stats analyses (should the whole package 
    just switch to pandas?)
    -Fix TkDnD for this GUI

Created on Wed Sep 18 10:15:58 2019

@author: Fruit Flies
"""
# -----------------------------------------------------------------------------
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
#import h5py
import numpy as np
import csv
#from statsmodels.stats.multicomp import (MultiComparison, pairwise_tukeyhsd)
import scikit_posthocs as sp
from openpyxl import load_workbook
from scipy.stats import mstats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.version_info[0] < 3:
    from Tkinter import *
    import tkFileDialog
    from ttk import *
    import tkMessageBox
    import tkSimpleDialog
else:
    from tkinter import *
    from tkinter.ttk import *
    from tkinter import filedialog as tkFileDialog
    from tkinter import messagebox as tkMessageBox
    from tkinter import simplediaglog as tkSimpleDialog

#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, MultiCursor

from v_expresso_gui_params import (initDirectories, guiParams)
from v_expresso_utils import get_curr_screen_geometry
# allows drag and drop functionality. if you don't want this, or are having 
#  trouble with the TkDnD installation, set to false.
TKDND_FLAG = True
if TKDND_FLAG:
    from TkinterDnD2 import *

# ===================================================================
"""

General utility functions

"""
def convert_data_for_stats(data_list, data_labels):
    # initialize output array
    data_array = np.array([])
    label_list = []
    
    # loop through list elements and put data into one array + store labels
    for ith, data_curr in enumerate(data_list):
        data_array = np.append(data_array, data_curr)
        label_list.append(len(data_curr)*data_labels[ith])
    
    return (data_array, label_list) 
# =============================================================================
"""

Define frame for loading data. 

This should allow the user to select a summary analysis file (from the main 
visual expresso gui) and load it for plotting/comparison to other groups
 
"""  
# =============================================================================

class DataLoader(Frame):
    """ Frame containg buttons to search for data"""
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)
        
        # ------------------------------------------------------
        # buttons to allow data loading/clearing
        self.btnframe = Frame(parent.master)
        self.btnframe.grid(column=col, row=row, columnspan=2, sticky=(N, S, E, W))
        #self.btnframe.config(bg='white')
        
        # section label
        self.frame_label = Label(self.btnframe, text='Select Data:',
                               font=guiParams['labelfontstr'])
                               #background=guiParams['bgcolor'])
                               #foreground=guiParams['textcolor'], 
                               
        self.frame_label.grid(column=col, row=row, columnspan=3, padx=10, pady=2, 
                            sticky=(N, S, E, W))
        #self.btnframe.config(background=guiParams['bgcolor'])
        
        # -------------------------------------------------------------
        # define load data button
        self.add_button =Button(self.btnframe, text='Load Data',
                                   command= lambda: self.add_data(parent))
        self.add_button.grid(column=col, row=row+1, padx=10, pady=2,
                                sticky=NSEW)
        
        # -------------------------------------------------------------
        # define remove current data selection button
        self.rm_button = Button(self.btnframe, text='Remove Data',
                                  command= lambda: self.rm_data(parent),
                                  state=DISABLED)
                                   
        self.rm_button.grid(column=col+1, row=row+1, padx=10, pady=2,
                                sticky=NSEW)
        
        # -------------------------------------------------------------
        # define clear all data files button
        self.clear_button = Button(self.btnframe, text='Clear Data',
                                  command= lambda: self.clear_data(parent))
                                   
        self.clear_button.grid(column=col+2, row=row+1, padx=10, pady=2,
                                sticky=NSEW)
        # ---------------------------------------------------------------------
        # create listbox to house the different files for potential analysis
        # ---------------------------------------------------------------------        
        self.filelistframe = Frame(parent.master)
        self.filelistframe.grid(column=col, row=row+1, columnspan=2, padx=10, 
                                pady=2, sticky=W)

        self.filelist = Listbox(self.filelistframe, width=64, height=10,
                                selectmode=EXTENDED)
        
        # --------------------------------------------------------------------                 
        # now make the Listbox and Text drop targets
        if TKDND_FLAG:
            self.filelist.drop_target_register(DND_FILES, DND_TEXT)

        self.filelist.bind('<<ListboxSelect>>', self.on_select)

        if TKDND_FLAG:
            self.filelist.dnd_bind('<<DropEnter>>', postProcess.drop_enter)
            self.filelist.dnd_bind('<<DropPosition>>', postProcess.drop_position)
            self.filelist.dnd_bind('<<DropLeave>>', postProcess.drop_leave)
            self.filelist.dnd_bind('<<Drop>>', postProcess.file_drop)
            self.filelist.dnd_bind('<<Drop:DND_Files>>', postProcess.file_drop)
            self.filelist.dnd_bind('<<Drop:DND_Text>>', postProcess.file_drop)

            self.filelist.drag_source_register(1, DND_TEXT, DND_FILES)
            # text.drag_source_register(3, DND_TEXT)

            self.filelist.dnd_bind('<<DragInitCmd>>', postProcess.drag_init_listbox)
            self.filelist.dnd_bind('<<DragEndCmd>>', postProcess.drag_end)
            # text.dnd_bind('<<DragInitCmd>>', drag_init_text)

        
        # ------------------------------------------------------------
        # scroll bars for listbox
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
        
        # index of selected file
        self.selection_ind = []
                        
    # -------------------------------------------------------------------------
    def on_select(self, selection):
        """ Enable or disable based on a valid directory selection. """

        if self.filelist.curselection():
            self.rm_button.configure(state=NORMAL)
        else:
            self.rm_button.configure(state=DISABLED)    
    # -------------------------------------------------------------------------
    def add_data(self,parent):
        """ add a new data file into the listbox """
        new_filename, new_data_name = postProcess.add_data(parent)
        new_entry = "{} -- {}".format(new_data_name, new_filename)
        entry_list = self.filelist.get(0, END)
        if (len(new_filename) > 0) and (new_entry not in entry_list):
            self.filelist.insert(END, new_entry)
        
    # -------------------------------------------------------------------------
    def rm_data(self, parent):
        selected = sorted(self.filelist.curselection(), reverse=True)
        for item in selected:
            self.filelist.delete(item)
        
        if (len(self.filelist.get(0,END)) < 1): 
            parent.dataFlag = False 
    # -------------------------------------------------------------------------    
    def clear_data(self,parent):
        self.filelist.delete(0, END)
        parent.dataFlag = False

# =============================================================================
"""

Define frame for setting data name as well as other plot misc. plot options

 
"""  
# =============================================================================
class DataOptions(Frame):
    """ Frame containg buttons to name data sets and set misc. options"""
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)
        
        self.options_frame = Frame(parent.master)
        self.options_frame.grid(column=col, row=row, sticky=(N, S, E, W))
        
        self.eater_chkbtn = Checkbutton(self.options_frame, 
                             text="Exclude flies without meals", 
                             variable=parent.onlyEatersFlag, 
                             command= lambda: postProcess.toggle_eaterFlag(parent))
        self.eater_chkbtn.grid(column=col,row=row, padx=2, pady=2, 
                            sticky=(N, S, E, W))
                             
    # ------------------------------
    # callback functions
    def update_toggle_var(*args):
        """ function to update selection of plot variable """
        print(parent.onlyEatersFlag.get())
        
# =============================================================================
"""

Define frame for setting the variables to plot, as well as the type of 
statistical test to perform on the data 
 
"""  
# =============================================================================
class PlotOptions(Frame):
    """ Frame containg buttons to search for data"""
    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)
        
        # options for plot/stats types
        self.plot_var_choices = parent.plot_var_list
        self.stats_type_choices = parent.stats_type_list
        
        # ------------------------------------
        # dropdown menu to select plot type
        self.plot_type_frame = Frame(parent.master)
        self.plot_type_frame.grid(column=col, row=row+1, sticky=(N, S, E, W))
        
        # label
        self.plot_type_label = Label(self.plot_type_frame, 
                                     text='Variable to Plot:',
                                     font=guiParams['labelfontstr'])
        self.plot_type_label.grid(column=col, row=row, padx=2, pady=2, 
                            sticky=(N, S, E, W))
        
        # initialize menu and populate with choices
        self.plot_option_menu = OptionMenu(self.plot_type_frame, 
                                           parent.plotVar,
                                           self.plot_var_choices[0],
                                           *self.plot_var_choices) 
        self.plot_option_menu.grid(column=col, row=row+1,  padx=2, pady=2,
                                   sticky=(N, S, E, W))
        
        
        # -----------------------------------------
        # dropdown menu to select statistics test
        self.stats_type_frame = Frame(parent.master)
        self.stats_type_frame.grid(column=col, row=row+2, sticky=(N, S, E, W))
        # label
        self.stats_type_label = Label(self.stats_type_frame, 
                                     text='Statistical Test:',
                                     font=guiParams['labelfontstr'])
        self.stats_type_label.grid(column=col, row=row, padx=2, pady=2, 
                            sticky=(N, S, E, W))
        
        # initialize menu and populate with choices
        self.stats_option_menu = OptionMenu(self.stats_type_frame, 
                                           parent.statsType, 
                                           self.stats_type_choices[0],
                                           *self.stats_type_choices) 
        self.stats_option_menu.grid(column=col, row=row+1, padx=2, pady=2,
                                    sticky=(N, S, E, W))
        
        # -----------------------------------------
        # button to update plot window
        self.update_plot_button = Button(parent.master, text='Update Display',
                                   command= lambda: postProcess.update_display(parent))
        self.update_plot_button.grid(column=col+1, row=row+1,columnspan=1,
                                     padx=2, pady=2, sticky=(N, S, E, W))        
        
         # -----------------------------------------
        # button to update stats calculation
        self.update_stats_button = Button(parent.master, text='Update Stats',
                                   command= lambda: postProcess.update_stats(parent))
        self.update_stats_button.grid(column=col+1, row=row+2,columnspan=1, 
                                      padx=2, pady=2, sticky=(N, S, E, W))        
        # ---------------------------------------------------------------------
        # callback functions
        def update_plot_var(*args):
            """ function to update selection of plot variable """
            print(parent.plotVar.get())
            
        def update_stats_type(*args):
            """ function to update selection of stats type """
            print(parent.statsType.get())   
        
#        def update_plot(self, parent):
#            """ function to update plot window/stats vals """
#            postProcess.update_display()
        
        # ---------------------------------------------------------
        # bind plot and stats type variables to dropdown menu
        parent.plotVar.trace("w",update_plot_var)
        parent.statsType.trace("w",update_stats_type)

# =============================================================================
"""
Frame for displaying stats, saving, etc
 
"""  
# =============================================================================


      
#==============================================================================
# Main class for GUI
#==============================================================================
""" 

Main GUI class (called postProcess)

Here we define the layout, size of the full GUI. We also define global 
functions that can be used across different frames. 

"""
class postProcess:
    
    def __init__(self, master):
        #--------------------------------------------
        # allow references to root (called in main)
        self.master = master
        
        #--------------------------------------------
        # current data set
        self.data_file_list = []
        self.dataFlag = True
        
        # --------------------------------------------
        # options for plotting and stats tests
        self.plotVar = StringVar(master)
        self.plot_var_list = ['Number of Meals', 
                       'Total Volume (nL)', 'Total Duration Eating (s)',
                        'Latency to Eat (s)', 'Cumulative Dist. (cm)',
                        'Average Speed (cm/s)', 'Fraction Time Moving', 
                        'Pre Meal Dist. (cm)', 'Food Zone Frac. (pre meal)',
                        'Food Zone Frac. (post meal)', 'Mealwise Duration (s)', 
                            'Mealwise Volume (nL)', 'Mealwise Dwell Time (s)']
        self.statsType = StringVar(master)
        self.stats_type_list = ['Anderson-Darling', 'Conover', 'Mann-Whitney', 
                                    'Tukey HSD', 'Wilcoxon']
        
        # just initialize with first entry
        #self.plotVar.set(self.plot_var_list[0])
        #self.statsType.set(self.stats_type_list[0])
        
        # -------------------------------------------
        # data options
        self.onlyEatersFlag = BooleanVar()
        self.onlyEatersFlag.set(False)
        #--------------------------------------------
        # where to look for data (if defined in '_params' file)
        if os.path.exists(initDirectories[-1]):
            self.init_dir = initDirectories[-1]
        else:
            self.init_dir = sys.path[0]
        
        #--------------------------------------------
        # gui basics
        self.init_gui()
        
        """ Define blocks within GUI (defined as other classes) """
        #--------------------------------------------
        # initialize instances of frames created above
        self.data_loader = DataLoader(self, col=0, row=0)
        self.data_options = DataOptions(self, col=0, row=2)
        self.plot_options = PlotOptions(self, col=0, row=3)

        #--------------------------------------------  
        # initialize plot window
        #self.canvasFig= plt.figure(1) ;
        self.fig = matplotlib.figure.Figure(figsize=(7,4),facecolor='none') ;
        self.ax = self.fig.add_subplot(111) ;
        
        
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.fig, 
                                                            master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=2, row=0, padx=10, pady=5, sticky=N)
        self.canvas._tkcanvas.grid(column=2, row=0, rowspan=4,padx=10, pady=5, sticky=N)
        self.fig.tight_layout()
        self.ax.set_axis_off()
        #--------------------------------------------
        # extra bit for quit command
        self.master.protocol("WM_DELETE_WINDOW", self.on_quit)
    
    
    """ functions for main GUI class """
    #===================================================================    
    def on_quit(self):
        """Exits program."""
        if tkMessageBox.askokcancel("Quit","Do you want to quit?"):
            self.master.destroy()
            self.master.quit()
    #===================================================================    
    def init_gui(self):
        
        # label for GUI        
        self.master.title('Visual Expresso Post Processing')
        
        # menu bar (to be expanded upon)
        self.master.option_add('*tearOff', 'FALSE')
        self.master.menubar = Menu(self.master)
        
        # file menu
        self.master.menu_file = Menu(self.master.menubar)
        self.master.menu_file.add_command(label='Exit', command=self.on_quit)
 
        # add these bits to menu bar
        self.master.menubar.add_cascade(menu=self.master.menu_file, label='File')
        
    #===================================================================
    @staticmethod
    def add_data(self):
        """ add summary data file to listbox by requesting user input """
        data_fn = tkFileDialog.askopenfilename(initialdir=self.init_dir,
                              title='Select summary analysis file (file type)')
        self.dataFlag = True
        
        # also prompt user to assign the data set a name 
        user_assigned_name = tkSimpleDialog.askstring("Input", 
                                    "Assign name for file: {}".format(data_fn),
                                    parent=self.master)
        return (data_fn, user_assigned_name)
        
    #=================================================================== 
    @staticmethod 
    def load_data(self):
        """ read in data values from summary file """
        data_entry_list = self.data_loader.filelist.get(0, END)
        plot_var = self.plotVar.get() 
        onlyEatersFlag = self.onlyEatersFlag.get()
        
        if "Mealwise" in plot_var:
            mealwiseFlag = True
        else:
            mealwiseFlag = False 
        # read data filenames/user-assigned data names from entries
        data_fn_list = [s.split(' -- ')[1] for s in data_entry_list]
        data_names = [s.split(' -- ')[0] for s in data_entry_list]
        
        fn_list = [] 
        data_list = [] 
        for fn in data_fn_list:
            
            # get filetype extension and basename
            bn = os.path.basename(fn)
            bn_noExt, ext = os.path.splitext(bn)
            
            #print(bn)
            fn_list.append(bn_noExt)
            
            # load data differently depending on filetype
            if ext == ".xlsx":
                # load in full xlsx workbook
                 wb = load_workbook(fn)
                 
                 # restirct attention to either summary or event sheet 
                 if mealwiseFlag:
                     sheet_index = wb.sheetnames.index('Events')
                     prefixStr = 'Mealwise '
                 else:
                     sheet_index = wb.sheetnames.index('Summary')
                     prefixStr = ''
                 wb.active = sheet_index
                 sheet = wb.active
                 
                 # get column headers to see which data to grab
                 sheet_headers = [] 
                 max_col = sheet.max_column
                 for c in range(1,max_col):
                     sheet_headers.append(prefixStr + 
                             str(sheet.cell(row=1,column=c).value))
                 
                 try:
                     col_idx = sheet_headers.index(plot_var)
                 except ValueError:
                     print('Could not find variable {}'.format(plot_var))
                     continue
                 
                 # how many rows the sheet contains
                 max_row = sheet.max_row
                 
                 # grab data from that column
                 data_curr = np.full((max_row-1,1),np.nan)
                 for i in range(1,max_row):
                     data_curr[i-1] = sheet.cell(row=i+1,column=col_idx+1).value
                     
                 # remove non-eating flies, if selected, as well as nan values
                 if onlyEatersFlag and not mealwiseFlag:
                     num_meals_col_idx = sheet_headers.index('Number of Meals')
                     num_meals = np.full((max_row-1,1),np.nan)
                     for ii in range(1,max_row):
                         num_meals[ii-1] = sheet.cell(row=ii+1, 
                                            column=num_meals_col_idx+1).value
                                             
                     ignore_idx = (num_meals < 1) | (np.isnan(data_curr))
                 else:
                     ignore_idx = np.isnan(data_curr)
                 data_curr = data_curr[~ignore_idx]
                 
                 #data_curr = np.reshape(data_curr, (data_curr.size, 1))
                 
            elif ext == ".csv":
                print("under construction")
                continue
            else:
                print("Invalid data file")
                continue
            
            # add data from current file to list of data
            data_list.append(data_curr)
            
        return (data_list, data_names)
        
        
    # ===================================================================
    @staticmethod
    def update_display(self):
         """ update inline plot display """
         
         # get currently loaded data
         (data_list, data_names) = postProcess.load_data(self)
         N_dsets = len(data_list)
         if (N_dsets < 1):
             print('No data to plot')
             return
             
         # set of colors used for bar plots 
         cmap = cm.get_cmap('Pastel1')
         norm_range = np.linspace(0,1,num=N_dsets)
         colors = cmap(norm_range)
         
         # get current plotting data
         plot_var = self.plotVar.get() 
         
         # clear previous plot
         self.ax.cla()
         
         # box plot properties
         boxprops = dict(linestyle='-', linewidth=1, color='k')
         flierprops = dict(marker='o', markeredgecolor='k', markersize=6,
                        markerfacecolor='none')
         medianprops = dict(linestyle='-', linewidth=1.5, color='k')
         whiskerprops = dict(linestyle='-', linewidth=1, color='k')
         # plot new data
         self.bplot = self.ax.boxplot(data_list,
                                      vert=True, # vertical box alignment
                                      patch_artist=True, # fill with color
                                      labels=data_names, # to label boxes
                                      boxprops=boxprops, # additional box properties
                                      flierprops=flierprops, 
                                      medianprops=medianprops,
                                      whiskerprops=whiskerprops) 
                                      
         for patch, color in zip(self.bplot['boxes'], colors):
             patch.set_facecolor(color)
         # handle axis labels, title, etc
         self.ax.set_xticklabels(data_names, fontsize='small',rotation=25)
         self.ax.set_ylabel(plot_var)
         
         # correct y limits
         self.ax.autoscale(enable=True, axis='y', tight=True)
         
         # make sure axis labels fit         
         self.fig.tight_layout()
         
         
         # draw figure
         self.canvas.draw()
         
    # ===================================================================
    @staticmethod
    def update_stats(self):
         """ update statstics test """
         # get currently loaded data
         (data_list, data_names) = postProcess.load_data(self)
         
         # make sure we're comparing at least two groups
         if (len(data_list) < 2):
             print('Need to compare at least two data sets')
             return
         
         # convert data to array 
#         (data, labels) = convert_data_for_stats(data_list)
         
         # get name of current data
         plot_var = self.plotVar.get() 
         
         # get type of stats test
         stats_type = self.statsType.get()
         
         # run stats test (plan is to fill this in with more test as time goes on)
#         if (stats_type == 'Kruskal Wallis'):
#             args = [dset for dset in data_list]
#             H, pval = mstats.kruskalwallis(*args)
         if (stats_type == 'Anderson-Darling'):
             pval_mat = sp.posthoc_anderson
         elif (stats_type == 'Conover'):
             pval_mat = sp.posthoc_conover
         elif (stats_type == 'Mann-Whitney'):
             pval_mat = sp.posthoc_mannwhitney
         elif (stats_type == 'Tukey HSD'):
             pval_mat = sp.posthoc_tukey_hsd
         elif (stats_type ==  'Wilcoxon'):
             pval_mat = sp.posthoc_wilcoxon
         else:
             print('Invalid hypothesis test selection')
         
         # assign labels to data frame output
         pval_mat.columns = data_names
         pval_mat.index = data_names
         
         # print stats results (nned to move this to gui)
         print(pval_mat)
         
         # update title to display stats
#         self.ax.set_title('{} test comparing {}: p = {}'.format(stats_type,
#                           plot_var, pval), fontsize='small')
         # make sure axis labels fit         
         self.fig.tight_layout()
         # draw 
         self.canvas.draw()
    # ===================================================================
    @staticmethod
    def toggle_eaterFlag(self):
        """ toggle the checkbutton that excludes/includes non-eating flies"""
        curr_val = self.onlyEatersFlag.get()
        self.onlyEatersFlag.set(~curr_val)
    
    # =====================================================================
    """ TkDND functions """
    if TKDND_FLAG:
        @staticmethod
        def drop_enter(event):
            event.widget.focus_force()
            # print('Entering widget: %s' % event.widget)
            # print_event_info(event)
            return event.action

        @staticmethod
        def drop_position(event):
            # print('Position: x %d, y %d' %(event.x_root, event.y_root))
            # print_event_info(event)
            return event.action

        @staticmethod
        def drop_leave(event):
            # print('Leaving %s' % event.widget)
            # print_event_info(event)
            return event.action

        # define drag callbacks

        @staticmethod
        def drag_init_listbox(event):
            # print_event_info(event)
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
            # print_event_info(event)
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
            # print_event_info(event)
            # this callback is not really necessary if it doesn't do anything useful
            print('Drag ended for widget:', event.widget)

        # specific functions for different listboxes
        @staticmethod
        def file_drop(event):
            if event.data:
                valid_ext = [".xlsx", ".csv"]

                files = event.widget.tk.splitlist(event.data)
                for f in files:
                    if os.path.exists(f) and f.endswith(tuple(valid_ext)):
        
                        # also prompt user to assign the data set a name 
                        data_name = tkSimpleDialog.askstring("Input", 
                                    "Assign name for file: {}".format(f))
                        new_entry = "{} -- {}".format(data_name, f)            
                        event.widget.insert('end', new_entry)
                    else:
                        print('Not dropping file "%s": file does not exist or is invalid.' % f)

            return event.action


#==============================================================================           
""" Run main loop """
if __name__ == '__main__':
    screen_geometry = get_curr_screen_geometry()
    #print(screen_geometry)
    if TKDND_FLAG:
        root = TkinterDnD.Tk()
    else:
        root = Tk()
    postProcess(root)
    root.mainloop()




