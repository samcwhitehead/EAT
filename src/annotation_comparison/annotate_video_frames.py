# -*- coding: utf-8 -*-
"""
#------------------------------------------------------------------------------
Script for manually annotating video files to check the performance of tracking
in Visual Expresso. 

Frames are chosen at random from the video, and the user is prompted to click
on the center of mass position for the fly in the given frame. This user input
can then be compared to results of the automated tracking
#------------------------------------------------------------------------------

INSTRUCTIONS:
    -To begin the annotation process, set the N_FRAMES variable to indicate the 
    number of frames that the user would like to correct. Once run, this script
    will try to obtain user input for the full N_FRAMES, so choosing too many 
    may result in having to terminate the script early (this should still save
    the data, but is best avoided if possible)
    
    -Run the script. A file dialog should pop up, prompting the user to select 
    a video file for annotation. The results will be saved in a .hdf5 file in 
    the same directory as the source video file, with suffix '_VID_ANNOTATION"

    -Once the script is run, and the video source selected, another window 
    should open with the first frame to be annotated. Inidcate the position of 
    the fly by a double (left) click of the mouse over the fly's estimated 
    position. NB: if the fly is not visible, press enter without clicking 
    (this will also be compared against the automated tracking results)
    
    -After all frames have been annotated, the program will close automatically

#------------------------------------------------------------------------------
Created on Wed Aug 01 15:25:04 2018

@author: Fruit Flies
"""
#------------------------------------------------------------------------------
import os
import sys
import cv2
import numpy as np
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

#-----------------------------------------------------------------------------
# callback function to return x,y coordinates of a mouse DOUBLE LEFT CLICK 
def get_xy(event,x,y,flags,param):
    global mouseX,mouseY,drawing
    if event == cv2.EVENT_LBUTTONDBLCLK:
        drawing = True
        mouseX,mouseY = x,y
        
#-----------------------------------------------------------------------------
# get fly position manually        
def get_fly_cm(img):
    clone = img.copy()
    win_name = 'click fly body position (double click, enter when finished)'
    
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name,get_xy) 
    cv2.imshow(win_name,img)
    
    while True:
        cv2.imshow(win_name,img)
        if drawing == True:
#            try:
            img = clone.copy()
            cv2.circle(img,(mouseX,mouseY),2,(255,0,0),-1)
#            except NameError:
#                pass
        
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break
        
    img = clone.copy()
    cv2.destroyAllWindows()
    
    if drawing == True:
        XCM = mouseX
        YCM = mouseY
    else:
        XCM = np.nan
        YCM = np.nan
    
    return (XCM,YCM)
    
#------------------------------------------------------------------------------
# run main function to load video, select random frames, and get user input
if __name__=="__main__":
    
    # define parameters 
    N_FRAMES = 100 # how many frames to annotate
    DELTA = 10      # how many of the initial frames to skip
    
    # initialize list for clicked points
    fly_xcm_list = [] 
    fly_ycm_list = [] 
    
    # load video file
    vid_filename = tkFileDialog.askopenfilename(initialdir=sys.path[0],
                                    title='Select video file to annotate') 
    cap = cv2.VideoCapture(vid_filename)
    
    
    # get frame indices for annotation
    N_TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))                             
    frames_to_annotate = np.random.randint(DELTA, N_TOTAL, N_FRAMES )
    
    # loop through randomly selected frames and prompt user to click on fly pos
    frame_counter = 1
    for frame_num in frames_to_annotate:
        cap.set(1,frame_num)
        ret, frame = cap.read()
    
        if not ret:
            print('Could not read frame--skipping')
            fly_xcm_list.append(np.nan)
            fly_ycm_list.append(np.nan)
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # get coordinates for frame
        print('Indicate fly position for frame number {:d}'.format(frame_num))
        mouseX = -1
        mouseY = -1 
        drawing = False
        fly_xcm, fly_ycm = get_fly_cm(img)
        
        fly_xcm_list.append(fly_xcm)
        fly_ycm_list.append(fly_ycm)
        
        print('Completed annotating {:d}/{:d}'.format(frame_counter, N_FRAMES))
        frame_counter += 1 

    # convert list to numpy array    
    xcm = np.array(fly_xcm_list,dtype=np.float32)
    ycm = np.array(fly_ycm_list,dtype=np.float32)
    
    # store/save results in hdf5 file
    save_filename = os.path.splitext(vid_filename)[0] + '_VID_ANNOTATION.hdf5'
    
    # if previous annotations have been performed, add to file rather than just
    #   overwrite. Note that there's no handling for non-unique frame numbers
    if os.path.exists(save_filename):
        with h5py.File(save_filename,'r') as f:
            frame_num_old = f['Time']['frame_num'].value 
            xcm_old = f['BodyCM']['xcm'].value 
            ycm_old = f['BodyCM']['ycm'].value
            
            frames_to_annotate = np.concatenate((frames_to_annotate,frame_num_old))
            xcm = np.concatenate((xcm, xcm_old))
            ycm = np.concatenate((ycm, ycm_old))
            
    with h5py.File(save_filename,'w') as f:
        f.create_dataset('Time/frame_num', data=frames_to_annotate)
        f.create_dataset('BodyCM/xcm', data=xcm)
        f.create_dataset('BodyCM/ycm', data=ycm)
    
    # close video file    
    cv2.destroyAllWindows()    
    cap.release()