# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:52:15 2019

@author: Fruit Flies
"""
# ------------------------------------------------------------------------------
import matplotlib

# matplotlib.use('TkAgg')

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

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
from PIL import Image, ImageTk

from v_expresso_gui_params import guiParams, trackingParams, initDirectories
from bout_and_vid_analysis import hdf5_to_flyCombinedData, interp_channel_time
from v_expresso_image_lib import invert_coord_transform


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class DataLoader(Frame):
    """ Frame containg buttons to search for data"""

    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)

        # buttons to allow data loading
        self.btnframe = Frame(parent.master)
        self.btnframe.grid(column=col, row=row, sticky=(N, S, E, W))
        # self.btnframe.config(bg='white')

        # section label
        self.frame_label = Label(self.btnframe, text='Select Data:',
                                 font=guiParams['labelfontstr'])
        # background=guiParams['bgcolor'])
        # foreground=guiParams['textcolor'],

        self.frame_label.grid(column=col, row=row, columnspan=2, padx=10, pady=2,
                              sticky=(N, S, E, W))
        # self.btnframe.config(background=guiParams['bgcolor'])

        # load data button
        self.lib_addbutton = Button(self.btnframe, text='Load Data',
                                    command=lambda: self.load_data(parent))
        self.lib_addbutton.grid(column=col, row=row + 1, padx=10, pady=2,
                                sticky=NSEW)

        # clear current data button
        self.lib_delbutton = Button(self.btnframe, text='Clear Data',
                                    command=lambda: self.rm_data(parent))

        self.lib_delbutton.grid(column=col + 1, row=row + 1, padx=10, pady=2,
                                sticky=NSEW)

        # self.data_name = StringVar()
        # self.data_name.set('')

        self.data_label = Label(self.btnframe, text='[Data file to analyze]',
                                wraplength=200, justify=LEFT)
        self.data_label.grid(column=col, row=row + 2, columnspan=2, padx=10,
                             pady=2, sticky=(N, S, E, W))

    # --------------------------------------------------------------------------
    def load_data(self, parent):
        new_filename = reviewTool.load_combined_data(parent)
        # update label
        self.data_label.config(text=new_filename)

    # --------------------------------------------------------------------------
    def rm_data(self, parent):
        reviewTool.clear_combined_data(parent)
        # update data label
        self.data_label.config(text='[Data file to analyze]')


# ------------------------------------------------------------------------------
class FrameSlider(Frame):
    """ Frame containing tools for moving through frames  """

    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)

        self.parent = parent
        # parent frame for slider stuff
        self.slide_frame = Frame(parent.master)
        self.slide_frame.grid(column=col, row=row, columnspan=6,
                              sticky=SW)

        # initialize slider
        self.slider = Scale(self.slide_frame, from_=0, to=100,
                            orient=HORIZONTAL, length=800,
                            command=self.slider_callback)
        self.slider.grid(column=col, row=row, padx=10, pady=2, columnspan=6,
                         sticky=SW)

        # initalize frame number entry box
        self.frame_entry_label = Label(self.slide_frame, text='Frame enter')
        self.frame_entry_label.grid(column=col, row=row + 1, padx=10, pady=10,
                                    sticky=E)
        self.frame_entry = Entry(self.slide_frame, width=8)
        self.frame_entry.insert(END, '1')
        self.frame_entry.grid(column=col + 1, row=row + 1, padx=10, pady=10,
                              sticky=W)
        self.frame_entry.bind('<Return>', self.entry_callback)

        # forward and backward buttons
        self.ffwd_button = Button(self.slide_frame, text='>>', command=self.ffwd)
        self.ffwd_button.grid(column=col + 5, row=row + 1, pady=10, sticky=W)

        self.fwd_button = Button(self.slide_frame, text='>', command=self.fwd)
        self.fwd_button.grid(column=col + 4, row=row + 1, pady=10, sticky=E)

        self.back_button = Button(self.slide_frame, text='<', command=self.back)
        self.back_button.grid(column=col + 3, row=row + 1, pady=10, sticky=W)

        self.bback_button = Button(self.slide_frame, text='<<', command=self.bback)
        self.bback_button.grid(column=col + 2, row=row + 1, pady=10, sticky=E)

    # --------------------------------------------------------------------------
    def slider_callback(self, event):
        slider_val = int(self.slider.get())
        self.try_update(slider_val)

    # --------------------------------------------------------------------------
    def try_update(self, val):
        if self.parent.dataFlag:
            reviewTool.update_display(self.parent, val)
        else:
            print('no data file loaded')

    # --------------------------------------------------------------------------
    def ffwd(self):
        curr_val = int(self.slider.get())
        self.slider.set(curr_val + 100)

    # --------------------------------------------------------------------------
    def fwd(self):
        curr_val = int(self.slider.get())
        self.slider.set(curr_val + 10)

    # --------------------------------------------------------------------------
    def back(self):
        curr_val = int(self.slider.get())
        self.slider.set(curr_val - 10)

    # --------------------------------------------------------------------------
    def bback(self):
        curr_val = int(self.slider.get())
        self.slider.set(curr_val - 100)

    # --------------------------------------------------------------------------
    def entry_callback(self, event):
        entry_val = int(self.frame_entry.get())
        self.slider.set(entry_val)


# ------------------------------------------------------------------------------
class ImageDisplay(Frame):
    """ Frame showing the current video frame """

    def __init__(self, parent, col=0, row=0):
        Frame.__init__(self, parent.master)

        self.im_label = Label(parent.master)
        self.im_label.grid(column=col, row=row)


# ==============================================================================
# Main class for GUI
# ==============================================================================
class reviewTool:
    """ main GUI class """

    def __init__(self, master):
        # --------------------------------------------
        # allow references to root (called in main)
        self.master = master

        # --------------------------------------------
        # current data set
        self.data_file = None
        self.vid_file = None
        self.N_frames = 100
        self.flyData = None
        self.cap = None
        self.dataFlag = False

        # --------------------------------------------
        # where to look for data
        if (len(initDirectories) > 0) and os.path.exists(initDirectories[-1]):
            self.init_dir = initDirectories[-1]
        else:
            self.init_dir = sys.path[0]
        # --------------------------------------------
        # image display settings
        self.im_height = 300
        self.im_width = 110

        # --------------------------------------------
        # gui basics
        self.init_gui()

        # --------------------------------------------
        # need to account for offset between video and data
        self.t_offset = trackingParams['t_offset']  # should really get this from params

        # --------------------------------------------
        # initialize instances of frames created above
        self.data_loader = DataLoader(self, col=0, row=0)
        self.frame_slider = FrameSlider(self, col=0, row=1)
        # self.im_display = ImageDisplay(self, col=1, row=0)

        # --------------------------------------------
        # initialize window for image display 
        init_img = np.zeros((self.im_width, self.im_height), dtype=np.uint8)
        img = Image.fromarray(init_img).resize((self.im_width, self.im_height))
        self.img = ImageTk.PhotoImage(image=img, master=self.master)
        self.im_label = Label(self.master, image=self.img)
        self.im_label.img = self.img
        self.im_label.grid(column=1, row=0, padx=10, pady=2, sticky=NSEW)

        # --------------------------------------------
        # initialize plot window
        # self.canvasFig= plt.figure(1) ;
        Fig = matplotlib.figure.Figure(figsize=(7, 4), facecolor='none');
        self.ax1 = Fig.add_subplot(211);
        self.ax2 = Fig.add_subplot(212);
        x = []
        y = []
        self.line11, = self.ax1.plot(x, y, 'b-')
        self.line12, = self.ax1.plot(x, y, 'r-')
        self.line21, = self.ax2.plot(x, y, 'b-')
        self.line22, = self.ax2.plot(x, y, 'r-')
        self.vline1 = self.ax1.axvline(x=0, color='k', linewidth=2)
        self.vline2 = self.ax2.axvline(x=0, color='k', linewidth=2)

        self.ax2.set_xlabel('Time (frames)')
        self.ax1.set_ylabel('Volume (nL)')
        self.ax2.set_ylabel('Volume (nL)')
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(Fig,
                                                                          master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=2, row=0, padx=10, pady=0, sticky=N)
        self.canvas._tkcanvas.grid(column=2, row=0, padx=10, pady=0, sticky=N)

        # --------------------------------------------
        # extra bit for quit command
        self.master.protocol("WM_DELETE_WINDOW", self.on_quit)

    """ functions for main GUI class """

    # ===================================================================
    def on_quit(self):
        """Exits program."""
        if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
            if self.cap:
                self.cap.release()
            self.master.destroy()
            self.master.quit()

    # ===================================================================
    def init_gui(self):
        """Label for GUI"""

        self.master.title('Visual Expresso Analysis Review Tool')

        """ Menu bar """
        self.master.option_add('*tearOff', 'FALSE')
        self.master.menubar = Menu(self.master)

        # file menu
        self.master.menu_file = Menu(self.master.menubar)
        self.master.menu_file.add_command(label='Exit', command=self.on_quit)

        # add these bits to menu bar
        self.master.menubar.add_cascade(menu=self.master.menu_file, label='File')

    # ===================================================================
    @staticmethod
    def load_combined_data(self):
        self.data_file = tkFileDialog.askopenfilename(initialdir=self.init_dir,
                                                      title='Select *_COMBINED_DATA.hdf5 to analyze')
        fn_split = self.data_file.split('_')
        vid_filename_no_ext = '_'.join(fn_split[:-2])
        vid_filename = vid_filename_no_ext + '.avi'
        if os.path.exists(vid_filename):
            # initiate video capture
            self.vid_file = vid_filename
            self.cap = cv2.VideoCapture(self.vid_file)

            self.N_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - self.t_offset
            self.dataFlag = True

            # set slide bar values
            self.frame_slider.slider.config(to=self.N_frames - 1)

            # load data 
            data_path, data_file = os.path.split(self.data_file)
            self.flyData = hdf5_to_flyCombinedData(data_path, data_file)

            # get data to add to plots
            self.dset, self.dset_smooth, self.bouts = interp_channel_time(self.flyData)
            self.frames = np.arange(self.dset.size)
            self.dset_bouts = np.empty(self.dset.shape)
            self.dset_bouts.fill(np.nan)
            self.dset_smooth_bouts = np.empty(self.dset.shape)
            self.dset_smooth_bouts.fill(np.nan)

            for i in np.arange(self.bouts.shape[1]):
                idx1 = self.bouts[0, i]
                idx2 = self.bouts[1, i]
                self.dset_smooth_bouts[idx1:idx2] = self.dset_smooth[idx1:idx2]
                self.dset_bouts[idx1:idx2] = self.dset[idx1:idx2]

            # set axis limits
            self.ax1.set_xlim(0, self.dset.size - 1)
            self.ax1.set_ylim(np.min(self.dset), np.max(self.dset))
            self.ax2.set_xlim(0, self.dset.size - 1)
            self.ax2.set_ylim(np.min(self.dset), np.max(self.dset))

            # add data to plot
            self.line11.set_data(self.frames, self.dset)
            self.line12.set_data(self.frames, self.dset_bouts)
            self.line21.set_data(self.frames, self.dset_smooth)
            self.line22.set_data(self.frames, self.dset_smooth_bouts)

            # draw points
            self.canvas.draw()

            # also load x,y track coordinates
            pix2cm = self.flyData['PIX2CM']
            cap_tip = self.flyData['cap_tip']
            if (sys.version_info[0] < 3):
                cap_tip_orient = self.flyData['cap_tip_orientation']
            else:
                encoding = 'utf-8'
                cap_tip_orient = self.flyData['cap_tip_orientation']
                if not isinstance(cap_tip_orient, str):
                    cap_tip_orient = cap_tip_orient.decode(encoding)

            xcm_trans = self.flyData['xcm_smooth']
            ycm_trans = self.flyData['ycm_smooth']
            self.xcm, self.ycm = invert_coord_transform(xcm_trans, ycm_trans,
                                                        pix2cm, cap_tip,
                                                        cap_tip_orient)
            # update display to first frame
            self.frame_slider.slider.set(0)
        else:
            tkMessageBox.showinfo(title='Error',
                                  message='No corresponding video file')
            vid_filename_no_ext = 'Error'

        return vid_filename_no_ext

    # ===================================================================
    @staticmethod
    def clear_combined_data(self):
        # reset data
        self.data_file = None
        self.vid_file = None
        self.N_frames = 100
        self.flyData = None
        self.cap = None
        self.dataFlag = False

        # clear liquid level plots display
        self.line11.set_data([], [])
        self.line12.set_data([], [])
        self.line21.set_data([], [])
        self.line22.set_data([], [])
        self.vline1.set_xdata(0)
        self.vline2.set_xdata(0)
        self.ax1.set_title('')

        # remove video frame and tracking point
        h_curr = self.img.height()  # get current image dimensions
        w_curr = self.img.width()
        img = np.zeros((w_curr, h_curr), dtype=np.uint8)  # generate a new image of all zeros
        img = Image.fromarray(img).resize((w_curr, h_curr))
        imgtk = ImageTk.PhotoImage(image=img, master=self.master)
        self.img = imgtk
        self.im_label.img = imgtk
        self.im_label.configure(image=imgtk)

        # implement changes
        self.canvas.draw()

        # reset slider to first frame
        self.frame_slider.slider.set(0)
    # ===================================================================
    @staticmethod
    def update_display(self, val):
        # update display image
        self.cap.set(1, val + self.t_offset)
        ret, frame = self.cap.read()

        # draw dot on fly
        if np.isnan(self.dset_bouts[val]):
            circ_color = (255, 0, 0)
            title_color = (0.0, 0.0, 0.0, 1.0)
        else:
            circ_color = (0, 0, 255)
            title_color = (1.0, 0.0, 0.0, 1.0)
        xcm = self.xcm[val]
        ycm = self.ycm[val]
        cv2.circle(frame, (int(xcm), int(ycm)), 2, circ_color, -1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # convert image so tkinter is happy
        img = Image.fromarray(cv2image).resize((self.im_width, self.im_height))
        imgtk = ImageTk.PhotoImage(image=img, master=self.master)
        self.img = imgtk
        self.im_label.img = imgtk
        self.im_label.configure(image=imgtk)

        # update plots
        self.vline1.set_xdata(val)
        self.vline2.set_xdata(val)

        self.ax1.set_title('Frame {}/{}'.format(val, self.N_frames),color=title_color)
        self.canvas.draw()


# ==============================================================================
""" Run main loop """
if __name__ == '__main__':
    root = Tk()
    reviewTool(root)
    root.mainloop()
