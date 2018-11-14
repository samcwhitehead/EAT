# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 22:42:57 2017

@author: samcw
"""
import os
import sys
import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
from scipy import signal, interpolate

import h5py
import csv 

import matplotlib.pyplot as plt
#import time
import ast 

from statsmodels import robust
import progressbar

from v_expresso_utils import interp_nans, hampel
from v_expresso_gui_params import trackingParams

#-----------------------------------------------------------------------------
# TO DO: (11/14/2018)
#   -undistort
#   -make sure bg subtract works when fly doesn't move much (improve guess)
#   -improve threshold estimation for bg subtraction
#   -kalman filter (too smoothing)
#   -enhance contrast for background subtracted images?
#   -fix the fact that dropped frames don't make sense
#   -deal with corners, shadows, occlusion, etc
#   -interpolation--different approaches for long vs short duration periods of 
#     not visible fly (in longer cases, if ends match up, connect? otherwise ignore?)
#-----------------------------------------------------------------------------
# use camera calibration parameters to undistort image 
def undistort_im(img, mtx,dist,alpha=1):
    # get image dimensions    
    h, w = img.shape[:2]
    
    # new camera matrix    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h),
                                                      centerPrincipalPoint=True)
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    #x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]
    return dst
    
#-----------------------------------------------------------------------------
# return a cropped, grayscale image specified by an ROI=r
def get_cropped_im(im,r):
    
    #cap.set(1,framenum)
    #_, frame = cap.read()
    #im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im_copy = im.copy() 
    imCrop = im_copy[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    return imCrop

#-----------------------------------------------------------------------------
# plot cropped image with cm marked on it
def plot_im_and_cm(framenum,x_cm,y_cm,cap,r):
    cap.set(1,framenum)
    _, frame = cap.read()
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    imCrop = get_cropped_im(im,r)
    imsize_row, imsize_col = imCrop.shape
    fig, ax = plt.subplots()
    ax.imshow(imCrop)
    ax.plot(y_cm[framenum],x_cm[framenum],'rx')

#-----------------------------------------------------------------------------
# multilevel otsu thresholding - UNDER CONSTRUCTION
def my_otsu(img):
    
    hist = np.histogram(img,256)[0]
    total = img.size
    
    no_of_bins = len( hist ) # should be 256
    
    sum_total = 0
    for x in range( 0, no_of_bins ):
    		sum_total += x * hist[x]
    	
    weight_background 	  = 0.0
    sum_background   = 0.0
    inter_class_variances = []
    
    for threshold in range( 0, no_of_bins ):
    	# background weight will be incremented, while foreground's will be reduced
    	weight_background += hist[threshold]
    	if weight_background == 0 :
    		continue
    
    	weight_foreground = total - weight_background
    	if weight_foreground == 0 :
    		break
    
    	sum_background += threshold * hist[threshold]
    	mean_background = sum_background / weight_background
    	mean_foreground = (sum_total - sum_background) / weight_foreground
    
    	inter_class_variances.append( weight_background * weight_foreground * \
                                     (mean_background - mean_foreground)**2 )
    
    # find the threshold with maximum variances between classes
    return np.argmax(inter_class_variances)

#-----------------------------------------------------------------------------
# user defined ROI
def get_roi(img,frame_title='Select ROI (press enter when finished)',
            fullScreenFlag=False):
    
    if fullScreenFlag:
        cv2.namedWindow(frame_title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(frame_title,cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(frame_title, cv2.WINDOW_NORMAL)
    
    fromCenter = False
    showCrosshair = False
    r = cv2.selectROI(frame_title,img,fromCenter,showCrosshair)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    return r

#-----------------------------------------------------------------------------
# callback function to return x,y coordinates of a mouse DOUBLE LEFT CLICK 
def get_xy(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print((x,y))
        mouseX,mouseY = x,y
        #param = (x,y)
        #cv2.circle(img,(mouseX,mouseY),4,(255,0,0),1)
        #cv2.imshow('get capillary tip',img)
        
#-----------------------------------------------------------------------------
# get capillary tip manually        
def get_cap_tip(img):
    clone = img.copy()
    win_name = 'get capillary tip (double click, enter when finished)'
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name,get_xy) 
    cv2.imshow(win_name,img)
    
    while True:
        cv2.imshow(win_name,img)
        try:
            img = clone.copy()
            cv2.circle(img,(mouseX,mouseY),4,(255,0,0),-1)
        except NameError:
            pass
        
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break
        
    img = clone.copy()
    cv2.destroyAllWindows()
    
    return (mouseX,mouseY)

#-----------------------------------------------------------------------------
# draw a line with mouse
def draw_line(img,window_name):
    
    clone = img.copy()
    
    class LinePoints:
        line_points = []
        
    def select_line_pts(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            LinePoints.line_points = [] 
            LinePoints.line_points.append((x,y))
            cv2.circle(img,(x,y),2,(255,0,0),-1)
            
        elif event == cv2.EVENT_LBUTTONUP:
            LinePoints.line_points.append((x,y))
            cv2.line(img,LinePoints.line_points[-2],(x,y),(255,0,0))
    
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name,select_line_pts)
    while(1):
        cv2.imshow(window_name,img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            img = clone.copy()
        elif k == 13:
            break
    
    img = clone.copy()
    cv2.destroyAllWindows()
    #out_pts = np.asarray(LinePoints.line_points)
    return LinePoints.line_points
    
#-----------------------------------------------------------------------------
# define pixel to centimeter conversion
def get_pixel2cm(img, vial_length_cm=4.42, vial_width_cm=1.22):
    print('Draw line indicating vial length; press enter after completion')
    vial_length_pts = draw_line(img,'Vial length')
    vial_length_px = cv2.norm(vial_length_pts[0],vial_length_pts[1])
    print('Draw line indicating vial width; press enter after completion')
    vial_width_pts = draw_line(img,'Vial width')     
    vial_width_px = cv2.norm(vial_width_pts[0],vial_width_pts[1])
    
    print("vial length conversion: ", vial_length_cm/vial_length_px)
    print("vial width conversion: ", vial_width_cm/vial_width_px)
    
    pix2cm = np.mean([vial_length_cm/vial_length_px,vial_width_cm/vial_width_px])
    return pix2cm
#-----------------------------------------------------------------------------
# beter version of find background
def get_bg(filename,r,fly_size_range=[20,100],min_dist=40,morphSize=3,
               debugFlag=True, verbose=True):
                   
    if verbose:
        print("Finding background...")
               
    cap = cv2.VideoCapture(filename)
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    varThresh= 125 #75
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=varThresh, 
                                              detectShadows=False)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morphSize,morphSize))
    
    x_cm = [] 
    y_cm = [] 
    framenum_list = []
    mean_intensity = [] 
    #min_dist = 70
    delta_cm = 0 
    cc = 10
    cap.set(1,cc)
    
    # set up named window for debugging if requested
    if debugFlag:
        cv2.namedWindow('frame and foreground mask',cv2.WINDOW_NORMAL)
        
    while (delta_cm < min_dist) and (cc < N_frames):
        
        ret, frame = cap.read()
    
        #frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2]),:]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity.append(np.mean(frame_gray))
        fgmask = fgbg.apply(frame)
        
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        if debugFlag:
            img_combined = np.hstack((frame_gray,fgmask))
            cv2.imshow('frame and foreground mask',img_combined)
            #cv2.imshow('foreground mask',fgmask)
            
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        
        
        cnts = cv2.findContours(fgmask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        if len(cnts) > 0:
            cnt_areas = np.asarray([cv2.contourArea(cnt) for cnt in cnts])
            cnt_area_check = (cnt_areas > fly_size_range[0]) & \
                            (cnt_areas < fly_size_range[1])
            if np.sum(cnt_area_check) == 1:
                c = cnts[np.where(cnt_area_check)[0][0]]
                M = cv2.moments(c)
                x_cm.append(M['m10']/M['m00'])
                y_cm.append(M['m01']/M['m00'])
                
                framenum_list.append(cc)
                
                if len(x_cm) > 1:
                    fly_cm = np.transpose(np.vstack((np.asarray(x_cm),np.asarray(y_cm))))
                    D = pdist(fly_cm)
                    #D = squareform(D);
                    delta_cm = np.nanmax(D)
            
            if verbose and (np.mod(cc,50) == 0):
                print("Find BG: " + str(cc) + "/" +  str(N_frames) + " completed")
            
            
        cc+=1
    
    # if the fly is successfully detected in multiple frames
    try: 
        D = squareform(D)
        ind = np.where(D==delta_cm)
        t1 = ind[0][0]
        t2 = ind[1][0]
        
        #get background from combination of frames with distant cm
        dx = np.abs(x_cm[t2] - x_cm[t1])
        dy = np.abs(y_cm[t2] - y_cm[t1])
        
        xmid = int((x_cm[t2] + x_cm[t1])/2)
        ymid = int((y_cm[t2] + y_cm[t1])/2)
        
        
        # get frames corresponding to the fly being at most distant point
        cap.set(1,framenum_list[t1])
        _, frame_t1 = cap.read()
        imt1 = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)
       
        cap.set(1,framenum_list[t2])
        _, frame_t2 = cap.read()
        imt2 = cv2.cvtColor(frame_t2, cv2.COLOR_BGR2GRAY)
        
        # normalize to our ghess for mean intensity
        mean_intensity_guess = np.mean(mean_intensity) 
        
        imt1 = cv2.subtract(imt1,(np.mean(imt1)-mean_intensity_guess))
        imt2 = cv2.subtract(imt2,(np.mean(imt2)-mean_intensity_guess))
        
        # creat background by stiching images together, exlcuding fly regions
        bg = imt1.copy()
        if dx >= dy:
            if x_cm[t1] < xmid:
                bg[:,:xmid] = imt2[:,:xmid]
            else:
                bg[:,xmid:] = imt2[:,xmid:]
        else:
            if y_cm[t1] < ymid:
                bg[:ymid,:] = imt2[:ymid,:]
                #print('case 1')
            else:
                bg[ymid:,:] = imt2[ymid:,:]
                #print('case 2')
        
        # try to guess min_thresh for get_cm
        cap.set(1,framenum_list[t2-1])
        _, frame_t3 = cap.read()
        imt3 = cv2.cvtColor(frame_t3, cv2.COLOR_BGR2GRAY)
        #imt3 = get_cropped_im(imt3,r) #imt3[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        imt3 = cv2.subtract(imt3,(np.mean(imt3)-mean_intensity_guess))        
        
        test_sub = cv2.absdiff(bg,imt3)
        
        test_sub = test_sub.ravel() 
        test_sub_sym = np.append(test_sub,-1*test_sub)
        min_thresh_guess = np.min([11*robust.mad(test_sub_sym),25])
        #print('mean = {:f}'.format(np.mean(test_sub_sym)))
        #print('std = {:f}'.format(np.std(test_sub_sym)))
        min_thresh_guess = int(min_thresh_guess)
        
        cap.release()
        #cv2.namedWindow('Background', cv2.WINDOW_NORMAL)
        #cv2.imshow('Background',bg)
        #cv2.waitKey(20)
    
    #============================================================
    # if the fly is not detected (e.g. if it very rarely moves)  
    #============================================================
    except UnboundLocalError: 
        print("BACKGROUND COULD NOT BE FOUND USING MOG; PERFORMING GUESS INSTEAD")
        mean_intensity_guess = np.mean(mean_intensity) 
        
        cap.set(1,10)
        _, frame_start = cap.read()
        im_start = cv2.cvtColor(frame_start, cv2.COLOR_BGR2GRAY)
        im_start = cv2.subtract(im_start,(np.mean(im_start)-mean_intensity_guess))
        #im_start = get_cropped_im(im_start,r) #[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        
        cap.set(1,N_frames-5)
        _, frame_end = cap.read()
        im_end = cv2.cvtColor(frame_end, cv2.COLOR_BGR2GRAY)
        #im_end = get_cropped_im(im_end,r) #[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        im_end = cv2.subtract(im_end,(np.mean(im_end)-mean_intensity_guess))  
        
        im_diff = cv2.absdiff(im_end, im_start)
        im_diff_thresh = my_otsu(im_diff)
        _, im_diff_bw = cv2.threshold(im_diff.astype('uint8'), \
                                im_diff_thresh, 255, cv2.THRESH_BINARY) 
        
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        im_diff_bw_2 = cv2.morphologyEx(im_diff_bw, cv2.MORPH_OPEN, se)
        im_diff_bw_3 = cv2.dilate(im_diff_bw_2, se, iterations = 1)

        fly_ind = np.where(im_diff_bw_3 > 0)                        
        
        bg = im_start.copy()
        bg[fly_ind[0],fly_ind[1]] = im_end[fly_ind[0],fly_ind[1]]
        
        #mean_intensity_guess = np.mean(bg.ravel())
        
        test_sub = cv2.absdiff(bg,im_end)
        test_sub = test_sub.ravel() 
        test_sub_sym = np.append(test_sub,-1*test_sub)
        min_thresh_guess = min_thresh_guess = np.min([11*robust.mad(test_sub_sym),25])
        #print('mean = {:f}'.format(np.mean(test_sub_sym)))
        #print('std = {:f}'.format(np.std(test_sub_sym)))
        min_thresh_guess = int(min_thresh_guess)
        
        cap.release()
    
    return (bg, x_cm, y_cm, mean_intensity_guess, min_thresh_guess)

#-----------------------------------------------------------------------------
# get center of mass of fly from image ROI        
def get_cm(filename,bg,r,fly_size_range=[20,100],morphSize=5,min_thresh=25, 
           mean_intensity=130.0, ellipseFlag = False, debugFlag=True, 
           verbose=True):
    
    if verbose:
        print('Finding center of mass coordinates for ' + filename + '...')           
    cap = cv2.VideoCapture(filename)
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    kernelRad = 3
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelRad,kernelRad))
    se_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    
    kalman = cv2.KalmanFilter(4, 2, 0)
    processNoiseLevel = 0.003
    measurementNoiseLevel = 0.1 #0.1
    
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov =  processNoiseLevel * np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.measurementNoiseCov = measurementNoiseLevel * np.array([[1,0],[0,1]],np.float32)
    
    N_MISSING_MAX = 3 # number of predicted tracks to allow
    N_MISSING_COUNTER = 0 
    N_INIT_IGNORE = 25 # number of initial points to skip kalman filtering
    
        #kalman.errorCovPost = 1. * np.ones((2, 2))
        #kalman.statePost = 0.1 * np.random.randn(2, 1)
    #se_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelRad,kernelRad))    
    #se_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelRad+1,kernelRad+1))
   
    #searchRadius = 10 
    
    x_cm = []
    y_cm = [] 
    #thresh_list = []
    #x_ind = []
    #y_ind = []
    angle_list = []
    dropped_frames = []
    
    ellipse_width = []
    ellipse_height =[]
    cnt_area = []
    xcm_curr = np.nan
    ycm_curr = np.nan
    
    if verbose:
        widgets = [progressbar.FormatLabel('Finding CM:'), ' ', 
                   progressbar.Percentage(), ' ',
                   progressbar.Bar('/'), ' ', progressbar.RotatingMarker()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=(N_frames-1))
        pbar.start()
    
    if debugFlag:
        cv2.namedWindow('Finding CM...',cv2.WINDOW_NORMAL)
    ith = 0
    while ith < N_frames:
        
        if verbose and (np.mod(ith,50) == 0):
           #print("Find CM: " + str(ith) + "/" +  str(N_frames) + " completed")
            pbar.update(ith)   
       
        
        _, frame = cap.read()
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im1 = cv2.subtract(im,(np.mean(im)-mean_intensity))
    
        im_minus_bg = cv2.absdiff(bg,im1)
        im_minus_bg = cv2.GaussianBlur(im_minus_bg,(morphSize,morphSize),0)
        
        
        otsu_thresh, _ = cv2.threshold(im_minus_bg.astype('uint8'), \
                                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        _, th_otsu = cv2.threshold(im_minus_bg.astype('uint8'), \
                                np.max([otsu_thresh, min_thresh]), \
                                255, cv2.THRESH_BINARY)
        
        #morphologically open image?
        th_otsu = cv2.morphologyEx(th_otsu, cv2.MORPH_OPEN, se)
        th_otsu = cv2.morphologyEx(th_otsu, cv2.MORPH_CLOSE, se)
        #th_otsu = cv2.erode(th_otsu, se_erode, iterations=1)  
        th_otsu = cv2.dilate(th_otsu, se_dilate, iterations=1)
        
        #---------------------------------------------------
        # fit contour to image, check results
        #---------------------------------------------------
        cnts = cv2.findContours(th_otsu.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        # if you detect contours:
        if len(cnts) > 0:
            # check area of detected contour to make sure it's the right size
            cnt_areas = np.asarray([cv2.contourArea(cnt) for cnt in cnts])
            cnt_area_check = (cnt_areas > fly_size_range[0]) & \
                            (cnt_areas < fly_size_range[1])
            
            # this is the case when you find one contour of appropriate size
            if np.sum(cnt_area_check) == 1:
                c = cnts[np.where(cnt_area_check)[0][0]]
                M = cv2.moments(c)
                xcm_curr = M['m10']/M['m00']
                ycm_curr = M['m01']/M['m00']
                #x_cm.append(xcm_curr)
                #y_cm.append(ycm_curr)
                
                if ellipseFlag:
                    try:
                        ellipse_fit = cv2.fitEllipse(cnt)
                        angle_list.append(ellipse_fit[-1])
                        ellipse_width.append(ellipse_fit[2])
                        ellipse_height.append(ellipse_fit[3])
                    except Exception:
                        ellipse_fit = None
                        angle_list.append(np.nan)
                        ellipse_width.append(np.nan)
                        ellipse_height.append(np.nan)
            
            # if there's more than one fly-sized contour:
            else:
                xcm_curr = np.nan
                ycm_curr = np.nan
                angle_list.append(np.nan)
                ellipse_width.append(np.nan)
                ellipse_height.append(np.nan)
                cnt_area.append(np.nan)
        
        #if you don't detect any contours:         
        else: 
            xcm_curr = np.nan
            ycm_curr = np.nan
            angle_list.append(np.nan)
            ellipse_width.append(np.nan)
            ellipse_height.append(np.nan)
            
            ellipse_fit = None
            dropped_frames.append(ith)
            
            cnt_area.append(np.nan)
        
        
        #--------------------------------------------
        # apply Kalman filter
        #--------------------------------------------
        # if you've detected a center of mass, save that data point
        if ~np.isnan(xcm_curr):
            mp = np.array([np.float32(xcm_curr), np.float32(ycm_curr)])
            kalman.correct(mp)
            tp = kalman.predict()
            
            if (ith > N_INIT_IGNORE):
                x_cm.append(tp[0])
                y_cm.append(tp[1])
            else:
                x_cm.append(np.float32(xcm_curr))
                y_cm.append(np.float32(ycm_curr))
                
            N_MISSING_COUNTER = 0 
        
        # if you haven't detected a center of mass, but you did recently:
        elif np.isnan(xcm_curr) and (N_MISSING_COUNTER < N_MISSING_MAX):
            tp = kalman.predict()
            x_cm.append(tp[0])
            y_cm.append(tp[1])
            N_MISSING_COUNTER += 1
        
        # if you haven't detected center of mass in a while
        else:
            tp = kalman.predict()
            x_cm.append(np.nan)
            y_cm.append(np.nan)
            dropped_frames.append(ith)
            
            
        if debugFlag:    
            try:
                #cv2.circle(im1,(int(xcm_curr),int(ycm_curr)),2,(255,0,0),-1)
                #cv2.circle(th_otsu,(int(x_cm[ith]),int(y_cm[ith])),2,(255,0,0),-1)
                cv2.circle(im1,(int(x_cm[ith]),int(y_cm[ith])),2,(255,0,0),-1)
                #cv2.circle(th_otsu,(int(tp[0]),int(tp[1])),2,(255,0,0),-1)
            except ValueError:
                dropped_frames.append(ith)
            if ellipseFlag and ellipse_fit:
                cv2.ellipse(im1,ellipse_fit,(255,0,0),1)
            
            img_combined = np.hstack((im1,th_otsu,cv2.bitwise_not(im_minus_bg)))
            cv2.imshow('Finding CM...',img_combined)
            #cv2.imshow('image',im1)
            #cv2.imshow('thresh',th_otsu)
            #cv2.imshow('im_minus_bg',im_minus_bg)
            
            k = cv2.waitKey(5) & 0xff
            if k == 27:
                break
            
        ith += 1
    
    # convert to numpy array
    x_cm = np.array(x_cm,dtype=np.float32)
    y_cm = np.array(y_cm,dtype=np.float32)
    
    # get rid of repeated dropped frames
    dropped_frames_set = set(dropped_frames)
    dropped_frames = list(dropped_frames_set)
    
    # print and return
    if verbose:
        pbar.finish()    
        print('Finished center of mass for ' + filename)
        print('')
        drop_frame_str = "Dropped frames: " + str(len(dropped_frames)) 
        print(drop_frame_str)
    cap.release()
    return (x_cm,y_cm,angle_list,dropped_frames,ellipse_width,ellipse_height,
            cnt_area)

#------------------------------------------------------------------------------
def visual_expresso_main(DATA_PATH, DATA_FILENAME, DEBUG_BG_FLAG=False,
                         DEBUG_CM_FLAG=False, SAVE_DATA_FLAG=False,
                         ELLIPSE_FIT_FLAG = False, PARAMS=trackingParams):
    """
    Main script for taking a (cropped) movie file and tracking the fly's motion
    
    Inputs:
        -DATA_PATH = path to video file
        -DATA_FILENAME = file name of video
        -DEBUG_BG_FLAG = true or false; determines whether or not background 
                        estimation is visualized
        -DEBUG_CM_FLAG = true or false; determines whether or not center of mass
                        tracking is visualized
        -SAVE_DATA_FLAG = true or false; whether or not to save results
        -ELLIPSE_FIT_FLAG = true or false; whether or not to fit ellipse to 
                            fly contour in order to estimate body angle
        -PARAMS = parameters used for tracking. these can be set in 
                'expresso_gui_params.py'
                        
    Outputs:
        -TBD
                    
    """      
                  
    #CALIB_COEFF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
    #                                    "/CalibImages/calib_coeff.hdf5")  
    SAVE_PATH = DATA_PATH 
    
    #---------------------------------------------------------------
    # load analysis parameters
    #---------------------------------------------------------------
    FLY_SIZE_RANGE = PARAMS['fly_size_range'] #still in pixels
    T_OFFSET = PARAMS['t_offset'] # to be filled later--there's a delay between video and Expresso
    BG_MIN_DIST = PARAMS['bg_min_dist'] 
    LABEL_FONTSIZE = PARAMS['label_fontsize'] # for any plots that come up in the script
    
    #----------------------------------------------
    # flags for various debugging/analysis options
    #----------------------------------------------
    UNDISTORT_FLAG = False      #undistort images using calibration coefficients
    SHOW_RESULTS_FLAG = False   #after analysis, play movie with track pts overlaid
    PLOT_BG_FLAG = False        #plot all ROI backgrounds
    PLOT_CM_FLAG = False        #plot x and y center of mass
    
    #--------------------------------------------------------------------------
    #=======================================
    # load video and get proper units/ROIs
    #=======================================
    filename = os.path.join(DATA_PATH, DATA_FILENAME)
    cap = cv2.VideoCapture(filename)
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    VID_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    VID_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    #get base image from which to select ROIs
    ret, frame = cap.read(1)
    im0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # try to undistort video. UNDER CONSTRUCTION
    if UNDISTORT_FLAG:
        print('under construction')
        #with h5py.File(CALIB_COEFF_PATH,'r') as f:
        #    mtx = f['.']['mtx'].value 
        #    dist = f['.']['dist'].value 
        #im0 = undistort_im(im0,mtx,dist)
    
    
    #=======================================
    # check if pre-processing info exists
    #=======================================             
    data_prefix = os.path.splitext(DATA_FILENAME)[0]
    vid_prefix = '_'.join(data_prefix.split('_')[:-3])
    vid_info_filename = os.path.join(DATA_PATH,
                                     vid_prefix + "_VID_INFO.hdf5")
    xp_name = data_prefix.split('_')[-3]
    channel_name = 'channel_' + data_prefix.split('_')[-1]     
    
    vid_info_filename = os.path.abspath(vid_info_filename)
    
    if os.path.exists(vid_info_filename):
        with h5py.File(vid_info_filename,'r') as f:
            PIX2CM = f['Params']['PIX2CM'].value 
            ROI = f['ROI'][xp_name + '_' + channel_name].value
            cap_tip = f['CAP_TIP'][xp_name + '_' + channel_name].value
            cap_tip_orientation = f['CAP_TIP_ORIENTATION'][xp_name + \
                                    '_' + channel_name].value
    else:
        im_for_roi = im0.copy()    
        im_for_meas = im0.copy()
        
        # determine pixel to centimeter conversion 
        if not PARAMS['pix2cm']:
            PIX2CM = get_pixel2cm(im_for_meas)
        else:
            PIX2CM = PARAMS['pix2cm']
            
        # get ROI and capillary tip location for the video
        ROI = get_roi(im0)
        im_roi = get_cropped_im(im_for_roi,ROI) 
        cap_tip = get_cap_tip(im_roi)
        cap_tip = np.float32(cap_tip)
        cap_tip_orientation = [] 
    
    #=======================================
    # Estimate static background
    #=======================================
    BG,_, _,mean_intensity,min_thresh_guess = get_bg(filename,ROI,
                 fly_size_range=FLY_SIZE_RANGE, min_dist = BG_MIN_DIST, 
                 debugFlag = DEBUG_BG_FLAG)
            
    # if you want to plot background
    if PLOT_BG_FLAG:
        fig, ax = plt.subplots()
        ax.imshow(BG,cmap='Greys')
        ax.set_title(DATA_FILENAME)
        plt.tight_layout()    
        
    #=======================================
    # Get center of mass coordinates
    #=======================================
    xcm, ycm, body_angle, _, ell_width,ell_height, cnt_area= get_cm(filename, 
                                             BG, ROI, 
                                             mean_intensity= mean_intensity,
                                             min_thresh = min_thresh_guess, 
                                             ellipseFlag = ELLIPSE_FIT_FLAG,
                                             debugFlag=DEBUG_CM_FLAG)
    
    # adjust coordinates in case of different vial orientations
    roi_mid_x = ROI[2]/2.0
    roi_mid_y = ROI[3]/2.0
    
    #vial is oriented verically relative to camera coordinates
    if ROI[3] > ROI[2]:
        # capillary tip is at the top of the image (lower y)
        if cap_tip[1] < roi_mid_y:
            cap_tip_orientation = 'T'
            xcm_transformed = (PIX2CM*(xcm - cap_tip[0]))[T_OFFSET:]
            ycm_transformed = (PIX2CM*(ycm - cap_tip[1]))[T_OFFSET:]
        
        # capillary tip is at the bottom of the image (higher y)    
        else:
            cap_tip_orientation = 'B'
            xcm_transformed = (PIX2CM*(-1*xcm + cap_tip[0]))[T_OFFSET:]
            ycm_transformed = (PIX2CM*(-1*ycm + cap_tip[1]))[T_OFFSET:]
    
    #vial is oriented horizontally relative to camera coordinates
    else:
        # capillary tip is on the left 
        if cap_tip[0] < roi_mid_x:
            cap_tip_orientation = 'L'
            xcm_transformed = (PIX2CM*(ycm - cap_tip[1]))[T_OFFSET:]
            ycm_transformed = (PIX2CM*(xcm - cap_tip[0]))[T_OFFSET:]                         
        
        # capillary tip is on the right     
        else:
            cap_tip_orientation = 'L'
            xcm_transformed = (PIX2CM*(-1*ycm + cap_tip[1]))[T_OFFSET:]
            ycm_transformed = (PIX2CM*(-1*xcm + cap_tip[0]))[T_OFFSET:]   
            
#    xcm_list.append(xcm_transformed)
#    ycm_list.append(ycm_transformed)
#    body_angle_list.append(body_angle)
        
    # make arrays for frame number and real time
    frame_nums = np.arange(N_FRAMES)
    frame_nums= frame_nums[T_OFFSET:] #once T_OFFSET is defined
    t = frame_nums/FPS
    
    # temporary!
    frame_nums = frame_nums[:len(xcm_transformed)]
    t = t[:len(xcm_transformed)]
    
    
    if PLOT_CM_FLAG:
        
        # X vs t and Y vs t
        fig_vs_t, (ax0,ax1) = plt.subplots(2,1,sharex=True,figsize=(12,7))
        ax0.plot(t,xcm_transformed,color='r')
        ax1.plot(t,ycm_transformed,color='r')
        
        #ax0.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax1.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax0.set_ylabel('X [cm]',fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
        ax0.set_title(DATA_FILENAME)
        
        ax0.set_xlim([np.amin(t),np.amax(t)])
        plt.tight_layout()
        
        # X vs Y
        fig_spatial, ax = plt.subplots(figsize=(8.0,3.6))
        ax.plot(xcm_transformed,ycm_transformed,color='r')
        ax.set_xlabel('X [cm]',fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
        ax.set_title(DATA_FILENAME)
        plt.tight_layout()
        plt.axis('equal')
    #=======================================
    # Save results 
    #=======================================
    if SAVE_DATA_FLAG:
        savename_prefix = os.path.splitext(DATA_FILENAME)[0]
        save_filename = os.path.join(SAVE_PATH,savename_prefix + "_TRACKING.hdf5")
        
        with h5py.File(save_filename,'w') as f:
            f.create_dataset('Time/t', data=t)
            f.create_dataset('Time/frame_num', data=frame_nums)
            f.create_dataset('Params/pix2cm', data=PIX2CM)
            
            
            f.create_dataset('BodyCM/xcm', data=xcm_transformed)
            f.create_dataset('BodyCM/ycm', data=ycm_transformed)
            
            f.create_dataset('ROI/roi', data=ROI)                 
            f.create_dataset('BG/bg', data=BG)
            f.create_dataset('CAP_TIP/cap_tip', data=cap_tip)
            f.create_dataset('CAP_TIP/cap_tip_orientation',
                             data=cap_tip_orientation)
            if ELLIPSE_FIT_FLAG:
                f.create_dataset('BodyAngle/body_angle',
                             data=body_angle)
    #=======================================
    # Showing tracking results on frame
    #=======================================
    if SHOW_RESULTS_FLAG:
        
        # number of frames to display/save
        N_MOV_FRAMES = 3600 # ~2 minutes of data
        
        # video writer object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        savename_prefix = os.path.splitext(DATA_FILENAME)[0]
        save_filename_vid = savename_prefix + "_TRACKING_RESULTS.avi"
        writer = cv2.VideoWriter(os.path.join(SAVE_PATH,save_filename_vid),fourcc,
                                 FPS,(int(VID_WIDTH),int(VID_HEIGHT)))
        cv2.namedWindow('Tracking results')
           
        cc = T_OFFSET
        cap.set(1,cc)
        
        while cc < N_MOV_FRAMES:
            
            _, frame = cap.read()
            bgr_vec = (0,0,255)
            
            for xcm_curr, ycm_curr in zip(xcm_transformed[:cc],ycm_transformed[:cc]):
                try:
                    if cap_tip_orientation == 'T':
                        xcm_curr_transformed = int(xcm_curr/PIX2CM) + \
                                        cap_tip[0] + int(ROI[0])
                        ycm_curr_transformed = int(ycm_curr/PIX2CM) + \
                                        cap_tip[1] + int(ROI[1])
                    elif cap_tip_orientation == 'B': 
                        xcm_curr_transformed = -1*int(xcm_curr/PIX2CM) + \
                                        cap_tip[0] + int(ROI[0])
                        ycm_curr_transformed = -1*int(ycm_curr/PIX2CM) + \
                                        cap_tip[1] + int(ROI[1])
                    elif cap_tip_orientation == 'L':   
                        xcm_curr_transformed = int(ycm_curr/PIX2CM) + \
                                        cap_tip[0] + int(ROI[0])
                        ycm_curr_transformed = int(xcm_curr/PIX2CM) + \
                                        cap_tip[1] + int(ROI[1])
                    else:
                        xcm_curr_transformed = -1*int(ycm_curr/PIX2CM) + \
                                        cap_tip[0] + int(ROI[0])
                        ycm_curr_transformed = -1*int(xcm_curr/PIX2CM) + \
                                        cap_tip[1] + int(ROI[1])
                                        
                    cv2.circle(frame,(xcm_curr_transformed,
                                      ycm_curr_transformed),1,bgr_vec,-1)
                except ValueError:
                    continue
            
            cv2.imshow('Tracking results',frame)
            #cv2.imwrite(os.path.join(SAVE_PATH,'im_{:04d}.png'.format(cc)),frame)
            writer.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 :
                break
            cc+=1
        #close writer object
        writer.release()
        
    #=======================================
    # Close windows and release cv2 object
    #=======================================
    cap.release()    
    cv2.destroyAllWindows()
    
    #=======================================
    # save fly data in a dict object
    #=======================================
    flyTrackData = {'filepath' : DATA_PATH , 
                    'filename' : DATA_FILENAME , 
                    'xp_name' : xp_name , 
                    'channel_name' : channel_name , 
                    'ROI' : ROI , 
                    'cap_tip' : cap_tip , 
                    'cap_tip_orientation' : cap_tip_orientation ,
                    'PIX2CM' : PIX2CM , 
                    'BG' : BG , 
                    'frames' : frame_nums ,
                    't' : t , 
                    'xcm' : xcm_transformed , 
                    'ycm' : ycm_transformed , 
                    'trackingParams' : PARAMS}
    if ELLIPSE_FIT_FLAG:
        flyTrackData['body_angle'] = body_angle
                
    return flyTrackData
    
#------------------------------------------------------------------------------
def process_visual_expresso(DATA_PATH, DATA_FILENAME, PARAMS=trackingParams, 
                            SAVE_DATA_FLAG = False, DEBUG_FLAG = False):
    
    SAVE_PATH = DATA_PATH
    SMOOTHING_FACTOR = PARAMS['smoothing_factor']    # degree of smoothing for interpolant spline [0, Inf)
    MEDFILT_WINDOW = PARAMS['medfilt_window']     # window for median filter, in units of frame number
    HAMPEL_K = PARAMS['hampel_k']     # window for median filter, in units of frame number
    HAMPEL_SIGMA = PARAMS['hampel_sigma']     # window for median filter, in units of frame number
    VEL_THRESH = PARAMS['vel_thresh']           # (cm/s) min speed for the fly to be 'moving' 
    
    LABEL_FONTSIZE = PARAMS['label_fontsize']  # for any plots that come up in the script
    
    #------------------------------------------------------------------------------
    
    #=======================================
    # load tracking data 
    #=======================================
    
    filename = os.path.join(DATA_PATH, DATA_FILENAME)
    
    data_prefix = os.path.splitext(DATA_FILENAME)[0]
    xp_name = data_prefix.split('_')[-3]
    channel_name = 'channel_' + data_prefix.split('_')[-1]   
    
    with h5py.File(filename,'r') as f:
        # kinematics for processing        
        t = f['Time']['t'].value
        PIX2CM = f['Params']['pix2cm'].value      
        xcm_curr = f['BodyCM']['xcm'].value 
        ycm_curr = f['BodyCM']['ycm'].value 
                
        # various params from original tracking to be resaved
        ROI = f['ROI']['roi'].value 
        BG =  f['BG']['bg'].value 
        cap_tip =  f['CAP_TIP']['cap_tip'].value 
        cap_tip_orientation = f['CAP_TIP']['cap_tip_orientation'].value 
            
        try:
            body_angle = f['BodyAngle']['body_angle'].value 
        except KeyError:
            body_angle = None
            
    
    #=======================================
    # Interpolate, filter, and smooth
    #=======================================
    dt = np.mean(np.diff(t))
    
    
    # interpolate through nan values with a spline
    xcm_interp = interp_nans(xcm_curr)
    ycm_interp = interp_nans(ycm_curr)
    
    # kalman filter
    #filtered_states = kalman_filt(xcm_curr,ycm_curr)
    #xcm_interp = filtered_states[:,0]
    #ycm_interp = filtered_states[:,1]
    
    # apply median filter to data
    #xcm_filt = signal.medfilt(xcm_interp,MEDFILT_WINDOW)
    #ycm_filt = signal.medfilt(ycm_interp,MEDFILT_WINDOW)
    
    # savitzky golay filter
    #xcm_filt = signal.savgol_filter(xcm_interp,SAV_GOL_WINDOW, SAV_GOL_ORDER)
    #ycm_filt = signal.savgol_filter(ycm_interp,SAV_GOL_WINDOW, SAV_GOL_ORDER)
    
    # hampel filter for outlier detection
    xcm_filt = hampel(xcm_interp, HAMPEL_K, HAMPEL_SIGMA)   
    ycm_filt = hampel(ycm_interp, HAMPEL_K, HAMPEL_SIGMA) 
    
    # fit smoothing spline to calculate derivative
    sp_xcm = interpolate.UnivariateSpline(t,xcm_filt,s=SMOOTHING_FACTOR)
    sp_ycm = interpolate.UnivariateSpline(t,ycm_filt,s=SMOOTHING_FACTOR)
    
    sp_xcm_vel = sp_xcm.derivative(n=1)  
    sp_ycm_vel = sp_ycm.derivative(n=1)
    
    # append to lists
    xcm_smooth = sp_xcm(t)
    ycm_smooth = sp_ycm(t)
    xcm_vel = sp_xcm_vel(t)
    ycm_vel = sp_ycm_vel(t)
    
    vel_mag = np.sqrt(xcm_vel**2 + ycm_vel**2)
    moving_ind = (vel_mag > VEL_THRESH)
    cum_dist = dt*np.cumsum(vel_mag)
    
    #=======================================
    # Plot results for debugging
    #======================================= 
    if DEBUG_FLAG:
        fig_vs_t, (ax0,ax1) = plt.subplots(2,1,sharex=True,figsize=(12,7))
        
        #raw
        ax0.plot(t,xcm_curr,'k.',label='raw')
        ax1.plot(t,ycm_curr,'k.',label='raw')
        
        #smoothed
        ax0.plot(t,xcm_smooth,'r',label='smoothed')
        ax1.plot(t,ycm_smooth,'r',label='smoothed')
        
        #interpolated
        #ax0.plot(t,xcm_interp,'b:',label='interpolated')
        #ax1.plot(t,ycm_interp,'b:',label='interpolated')
        
        #filtered
        #ax0.plot(t,xcm_filt,'g:',label='filtered')
        #ax1.plot(t,ycm_filt,'g:',label='filtered')
        
        #ax0.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax1.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax0.set_ylabel('X [cm]',fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
        ax0.set_title(DATA_FILENAME)
        
        ax0.legend(loc='upper right')
        ax1.legend(loc='upper right')
        
        ax0.set_xlim([np.amin(t),np.amax(t)])
        plt.tight_layout()
    
        
    #=======================================
    # Save results (?)
    #=======================================          
        
    if SAVE_DATA_FLAG:
        savename_prefix = os.path.splitext(DATA_FILENAME)[0]
        save_filename = os.path.join(SAVE_PATH,savename_prefix + "_PROCESSED.hdf5")
        #save_filename = filename
        with h5py.File(save_filename,'w') as f:
            f.create_dataset('Time/t', data=t)
            f.create_dataset('Params/pix2cm', data=PIX2CM)
            f.create_dataset('Params/trackingParams', data=str(PARAMS))
            
            # body velocity data            
            f.create_dataset('BodyVel/vel_x', data=xcm_vel)
            f.create_dataset('BodyVel/vel_y', data=ycm_vel)
            f.create_dataset('BodyVel/vel_mag', data=vel_mag)       
            f.create_dataset('BodyVel/moving_ind', data=moving_ind)
                             
            # body position data                 
            f.create_dataset('BodyCM/xcm_smooth', data=xcm_smooth)
            f.create_dataset('BodyCM/ycm_smooth', data=ycm_smooth)
            f.create_dataset('BodyCM/cum_dist', data=cum_dist)
                             
            # unprocessed data/information from tracking output file
            f.create_dataset('BodyCM/xcm', data=xcm_curr)
            f.create_dataset('BodyCM/ycm', data=ycm_curr)
            f.create_dataset('ROI/roi', data=ROI)                 
            f.create_dataset('BG/bg', data=BG)
            f.create_dataset('CAP_TIP/cap_tip', data=cap_tip)
            f.create_dataset('CAP_TIP/cap_tip_orientation',
                             data=cap_tip_orientation)
            if body_angle:
                f.create_dataset('BodyAngle/body_angle', data=body_angle)                 
    
    #=======================================
    # save fly data in a dict object
    #=======================================
    flyTrackData = {'filepath' : DATA_PATH , 
                    'filename' : DATA_FILENAME , 
                    'xp_name' : xp_name , 
                    'channel_name' : channel_name , 
                    'ROI' : ROI , 
                    'cap_tip' : cap_tip , 
                    'cap_tip_orientation' : cap_tip_orientation ,
                    'PIX2CM' : PIX2CM , 
                    'BG' : BG , 
                    'frames' : np.arange(len(t)) ,
                    't' : t , 
                    'xcm' : xcm_curr , 
                    'ycm' : ycm_curr ,
                    'xcm_smooth' : xcm_smooth , 
                    'ycm_smooth' : ycm_smooth ,
                    'xcm_vel' : xcm_vel , 
                    'ycm_vel' : ycm_vel , 
                    'vel_mag' : vel_mag ,
                    'cum_dist' : cum_dist , 
                    'moving_ind' : moving_ind , 
                    'body_angle' : body_angle ,
                    'trackingParams' : PARAMS}
                    
    return flyTrackData
#------------------------------------------------------------------------------
    
def hdf5_to_flyTrackData(DATA_PATH, DATA_FILENAME):
    
    # file information    
    flyTrackData = {'filepath' : DATA_PATH , 
                    'filename' : DATA_FILENAME}
    
    data_prefix = os.path.splitext(DATA_FILENAME)[0]
    xp_name = data_prefix.split('_')[-3]
    channel_name = 'channel_' + data_prefix.split('_')[-1]   
    
    # bank and channel names
    flyTrackData['xp_name'] = xp_name
    flyTrackData['channel_name'] = channel_name
    
    # tracking results
    filename_full = os.path.join(DATA_PATH,DATA_FILENAME)
    with h5py.File(filename_full,'r') as f:
       
        flyTrackData['ROI'] = f['ROI']['roi'].value
        flyTrackData['cap_tip'] = f['CAP_TIP']['cap_tip'].value
        flyTrackData['cap_tip_orientation'] = \
                        f['CAP_TIP']['cap_tip_orientation'].value
        
        flyTrackData['PIX2CM'] = f['Params']['pix2cm'].value
        flyTrackData['trackingParams'] = ast.literal_eval(f['Params']['trackingParams'].value)
        
        t = f['Time']['t'].value 
        flyTrackData['frames'] = np.arange(len(t))
        flyTrackData['t'] = t
        
        flyTrackData['xcm'] = f['BodyCM']['xcm'].value 
        flyTrackData['ycm'] = f['BodyCM']['ycm'].value 
        flyTrackData['xcm_smooth'] = f['BodyCM']['xcm_smooth'].value 
        flyTrackData['ycm_smooth'] = f['BodyCM']['ycm_smooth'].value 
        flyTrackData['cum_dist'] = f['BodyCM']['cum_dist'].value 
        
        flyTrackData['xcm_vel'] = f['BodyVel']['vel_x'].value 
        flyTrackData['ycm_vel'] = f['BodyVel']['vel_y'].value 
        flyTrackData['vel_mag'] = f['BodyVel']['vel_mag'].value 
        flyTrackData['moving_ind'] = f['BodyVel']['moving_ind'].value 
        
        try:
            flyTrackData['body_angle'] = f['BodyAngle']['body_angle'].value
        except KeyError:
            flyTrackData['body_angle'] = np.nan
    
    return flyTrackData
    
#------------------------------------------------------------------------------
def save_vid_time_series(VID_FILENAMES):    
    h5_filenames = [] 
    for vid_fn in VID_FILENAMES:
        file_path, filename = os.path.split(vid_fn)
        
        try:
            savename_prefix = os.path.splitext(filename)[0]
            hdf5_filename = os.path.join(file_path, savename_prefix + \
                                                    "_TRACKING_PROCESSED.hdf5")
            hdf5_filename = os.path.abspath(hdf5_filename)
            
            if os.path.exists(hdf5_filename):
                h5_filenames.append(hdf5_filename)
            else:
                print(hdf5_filename + ' not yet analyzed--failed to save')
                
        except AttributeError:
            print(hdf5_filename + ' not yet analyzed--failed to save')

    for h5_fn in h5_filenames:
        filepath = os.path.splitext(h5_fn)[0]  
        with h5py.File(h5_fn,'r') as f:
            csv_filename = filepath + ".csv" 
            t = f['Time']['t'].value
            
            # get kinematics
            try:
                xcm = f['BodyCM']['xcm_smooth'].value 
                ycm = f['BodyCM']['ycm_smooth'].value 
                cum_dist = f['BodyCM']['cum_dist'].value 
                x_vel = f['BodyVel']['vel_x'].value 
                y_vel = f['BodyVel']['vel_y'].value 
                vel_mag = f['BodyVel']['vel_mag'].value 
                moving_ind = f['BodyVel']['moving_ind'].value 
                
                column_headers=['Time (s)','X Position (cm)','Y Position (cm)', 
                            'Cumulative Dist. (cm)','X Velocity (cm/s)',
                            'Y Velocity (cm/s)','Speed (cm/s)','Moving Idx']
                row_mat = np.vstack((t, xcm, ycm, cum_dist, x_vel, y_vel, 
                                     vel_mag, moving_ind))
                row_mat = np.transpose(row_mat) 
            except KeyError:
                xcm = f['BodyCM']['xcm'].value 
                xcm = f['BodyCM']['ycm'].value 
                column_headers = ['Time (s)', 'X CM (cm)', 'Y CM (cm)']
                row_mat = np.vstack((t, xcm, ycm))
                row_mat = np.transpose(row_mat) 

        if sys.version_info[0] < 3:
            out_path = open(csv_filename,mode='wb')
        else:
            out_path = open(csv_filename, 'w', newline='')
        save_writer = csv.writer(out_path)
        
        save_writer.writerow([filepath])
        save_writer.writerow(column_headers)
           
        for row in row_mat:
            save_writer.writerow(row)
            
        out_path.close()
 
 #------------------------------------------------------------------------------
def save_vid_summary(VID_FILENAMES, CSV_FILENAME):
    
    column_headers = ['Filename', 'Bank', 'Channel', 'Cumulative Dist. (cm)',
                      'Average Speed (cm/s)', 'Fraction Time Moving']
    h5_filenames = [] 
    for vid_fn in VID_FILENAMES:
        file_path, filename = os.path.split(vid_fn)
        
        try:
            savename_prefix = os.path.splitext(filename)[0]
            hdf5_filename = os.path.join(file_path, savename_prefix + \
                                                    "_TRACKING_PROCESSED.hdf5")
            hdf5_filename = os.path.abspath(hdf5_filename)
            
            if os.path.exists(hdf5_filename):
                h5_filenames.append(hdf5_filename)
            else:
                print(hdf5_filename + ' not yet analyzed--failed to save')
                print('')
        except AttributeError:
            print(hdf5_filename + ' not yet analyzed--failed to save')
            print('')
    if sys.version_info[0] < 3:
        out_path = open(CSV_FILENAME,mode='wb')
    else:
        out_path = open(CSV_FILENAME, 'w', newline='')
    save_writer = csv.writer(out_path)                      
    save_writer.writerow(column_headers)                      
    
        
    for h5_fn in h5_filenames:
        filepath = os.path.splitext(h5_fn)[0]  
        
        
        with h5py.File(h5_fn,'r') as f:
            try:
                cum_dist = f['BodyCM']['cum_dist'].value 
                vel_mag = f['BodyVel']['vel_mag'].value 
                moving_ind = f['BodyVel']['moving_ind'].value 
                
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
                print('')
                continue 

            for row in row_mat:
                save_writer.writerow(row)
                    
    out_path.close()
    
#------------------------------------------------------------------------------

def plot_body_cm(flyTrackData, plot_color=(1,0,0),
                 LABEL_FONTSIZE=trackingParams['label_fontsize']):
        
        # load data from fly tracking structure
        t = flyTrackData['t']
        xcm = flyTrackData['xcm']
        ycm = flyTrackData['ycm']
        xcm_smooth = flyTrackData['xcm_smooth']
        ycm_smooth = flyTrackData['ycm_smooth']
        DATA_FILENAME = flyTrackData['filename']
        
        # X vs t and Y vs t
        fig_vs_t, (ax0,ax1) = plt.subplots(2,1,sharex=True,figsize=(12,7))
        ax0.plot(t,xcm,'k.',markersize=2,label='raw')
        ax1.plot(t,ycm,'k.',markersize=2,label='raw')
        
        ax0.plot(t,xcm_smooth,color=plot_color,label='smoothed')
        ax1.plot(t,ycm_smooth,color=plot_color,label='smoothed')
    
        ax1.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax0.set_ylabel('X [cm]',fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
        ax0.set_title( DATA_FILENAME)
        
        ax0.set_xlim([np.amin(t),np.amax(t)])
        ax0.legend(loc='upper right')
        ax1.legend(loc='upper right')
        plt.tight_layout()
        
        # Y vs X
        fig_spatial, ax = plt.subplots(figsize=(3.6,8.0))
        ax.plot(xcm_smooth,ycm_smooth,color=plot_color)
        ax.set_xlabel('X [cm]',fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
        ax.set_title( DATA_FILENAME)
        plt.axis('equal')
        #ax.set_xlim([np.nanmin(xcm_smoothed_list[mth]), 
        #             np.nanmax(xcm_smoothed_list[mth])])
        #ax.set_ylim([np.nanmin(ycm_smoothed_list[mth]), 
        #             np.nanmax(ycm_smoothed_list[mth])])
                    
#------------------------------------------------------------------------------
                    
def plot_body_vel(flyTrackData, plot_color=(1,0,0),
                 LABEL_FONTSIZE=trackingParams['label_fontsize']):   
        
        # load data from fly tracking structure
        t = flyTrackData['t']
        xcm_vel = flyTrackData['xcm_vel']
        ycm_vel = flyTrackData['ycm_vel']
        DATA_FILENAME = flyTrackData['filename']
            
        # velX vs t and velY vs t
        fig_vel_vs_t, (ax0,ax1) = plt.subplots(2,1,sharex=True,figsize=(12,7))
        ax0.plot(t,xcm_vel,color=plot_color)
        ax1.plot(t,ycm_vel,color=plot_color)
        
        #ax0.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax1.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax0.set_ylabel('X Vel. [cm/s]',fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Y Vel. [cm/s]',fontsize=LABEL_FONTSIZE)
        ax0.set_title(DATA_FILENAME)
        
        ax0.set_xlim([np.amin(t),np.amax(t)])
        #ax0.set_xlim([np.amin(t),60])
        plt.tight_layout()

#------------------------------------------------------------------------------
    
def plot_body_angle(flyTrackData, plot_color=(1,0,0),
                 LABEL_FONTSIZE=trackingParams['label_fontsize']):  
                
        # load data from fly tracking structure
        t = flyTrackData['t']
        xcm_vel = flyTrackData['xcm_vel']
        ycm_vel = flyTrackData['ycm_vel']
        body_angle = flyTrackData['body_angle']
        DATA_FILENAME = flyTrackData['filename']
        
        # body angle vs t
        fig_angle_vs_t, (ax0,ax1) = plt.subplots(2,1,sharex=True,figsize=(12,7))
        ax0.plot(t,body_angle,color=plot_color)
        ax1.plot(t,(180.0/np.pi)*np.arctan2(ycm_vel,xcm_vel), color=plot_color)
        
        
        #ax0.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax1.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax0.set_ylabel('Angle from ellipse [deg]',fontsize=LABEL_FONTSIZE)
        ax1.set_ylabel('Angle from vel. [deg]',fontsize=LABEL_FONTSIZE)
        ax0.set_title(DATA_FILENAME)
        
        ax0.set_xlim([np.amin(t),np.amax(t)])
        plt.tight_layout()        

#------------------------------------------------------------------------------

def plot_moving_v_still(flyTrackData, plot_color=(1,0,0),
                 LABEL_FONTSIZE=trackingParams['label_fontsize']):
        
        # load data from fly tracking structure
        t = flyTrackData['t']
        vel_mag = flyTrackData['vel_mag']
        moving_ind = flyTrackData['moving_ind']
        DATA_FILENAME = flyTrackData['filename']
        trackingParams = flyTrackData['trackingParams']
        VEL_THRESH = trackingParams['vel_thresh']
        
        # velocity magnitude separated by moving vs still
        fig_moving, ax0 = plt.subplots(1,1,figsize=(12,3))
        ax0.plot(t[moving_ind], vel_mag[moving_ind],'.',color=plot_color)
        ax0.plot(t[np.logical_not(moving_ind)], 
                   vel_mag[np.logical_not(moving_ind)], '.',
                    color=tuple([0.5, 0.5, 0.5 ]))
        ax0.plot(t,VEL_THRESH*np.ones(t.shape),'k--')
       
        
        ax0.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        ax0.set_ylabel('Speed [cm/s]',fontsize=LABEL_FONTSIZE)
        ax0.set_title(DATA_FILENAME)
        
        ax0.set_xlim([np.amin(t),np.amax(t)])
        plt.tight_layout()

#------------------------------------------------------------------------------
  
def plot_cum_dist(flyTrackData, plot_color=(1,0,0),
                 LABEL_FONTSIZE=trackingParams['label_fontsize']):
            
        # load data from fly tracking structure
        t = flyTrackData['t']
        cum_dist = flyTrackData['cum_dist']
        DATA_FILENAME = flyTrackData['filename']
        
        # cumulative distance vs time
        fig_cum_dist, ax0 = plt.subplots(1,1,figsize=(7,6))
        ax0.plot(t,cum_dist,color=plot_color)
        
        ax0.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
        
        ax0.set_ylabel('Cumulative Dist. [cm]',fontsize=LABEL_FONTSIZE)
        ax0.set_title(DATA_FILENAME)
        ax0.set_xlim([np.amin(t),np.amax(t)])
        plt.tight_layout()  
        
#------------------------------------------------------------------------------
def batch_plot_cum_dist(VID_FILENAMES,
                        LABEL_FONTSIZE=trackingParams['label_fontsize']):    
    h5_filenames = [] 
    fig_cum_dist, ax_cum_dist = plt.subplots(1,1,figsize=(8,6))
    max_t = 0 
    for vid_fn in VID_FILENAMES:
        file_path, filename = os.path.split(vid_fn)
        
        try:
            savename_prefix = os.path.splitext(filename)[0]
            hdf5_filename = os.path.join(file_path, savename_prefix + \
                                                    "_TRACKING_PROCESSED.hdf5")
            hdf5_filename = os.path.abspath(hdf5_filename)
            
            if os.path.exists(hdf5_filename):
                h5_filenames.append(hdf5_filename)
            else:
                print(hdf5_filename + ' not yet analyzed--failed to save')
                
        except AttributeError:
            print(hdf5_filename + ' not yet analyzed--failed to save')

    for h5_fn in h5_filenames:
        _, filename_split = os.path.split(h5_fn)
        file_prefix = os.path.splitext(filename_split)[0]
        data_prefix = '_'.join(file_prefix.split('_')[:-2])
        with h5py.File(h5_fn,'r') as f:
            # get kinematics
            try:
                t = f['Time']['t'].value
                cum_dist = f['BodyCM']['cum_dist'].value
                if np.amax(t) > max_t:
                    max_t = np.amax(t)
            except KeyError:
                print('Cumulative distance not defined')
                continue

        ax_cum_dist.plot(t,cum_dist,label=data_prefix)
    
    ax_cum_dist.set_xlabel('Time [s]',fontsize=LABEL_FONTSIZE)
    ax_cum_dist.set_ylabel('Cumulative Dist. [cm]',fontsize=LABEL_FONTSIZE)
    ax_cum_dist.set_xlim([0,max_t])
          
    plt.legend(loc='upper left',fontsize='x-small')
    plt.tight_layout()
    plt.show() 

#------------------------------------------------------------------------------

def batch_plot_heatmap(VID_FILENAMES, bin_size = 0.1,
                        LABEL_FONTSIZE=trackingParams['label_fontsize']):    
    h5_filenames = [] 
    fig_heatmap, ax_heatmap = plt.subplots(1,1,figsize=(4,8))
    
    Xedges = np.arange(-0.8, 0.8, bin_size)
    Yedges = np.arange(-0.5, 5.0 ,bin_size)
    heatmap_sum = np.zeros((Yedges.size-1,Xedges.size-1))    
    heatmap_cc = 0 
    
    for vid_fn in VID_FILENAMES:
        file_path, filename = os.path.split(vid_fn)
        
        try:
            savename_prefix = os.path.splitext(filename)[0]
            hdf5_filename = os.path.join(file_path, savename_prefix + \
                                                    "_TRACKING_PROCESSED.hdf5")
            hdf5_filename = os.path.abspath(hdf5_filename)
            
            if os.path.exists(hdf5_filename):
                h5_filenames.append(hdf5_filename)
            else:
                print(hdf5_filename + ' not yet analyzed--failed to save')
                
        except AttributeError:
            print(hdf5_filename + ' not yet analyzed--failed to save')

    for h5_fn in h5_filenames:
        #_, filename_split = os.path.split(h5_fn)
        #file_prefix = os.path.splitext(filename_split)[0]
        #data_prefix = '_'.join(file_prefix.split('_')[:-2])
        with h5py.File(h5_fn,'r') as f:
            # get kinematics
            try:
                xcm = f['BodyCM']['xcm_smooth'].value
                ycm = f['BodyCM']['ycm_smooth'].value
                heatmap_curr, _, _ = np.histogram2d(xcm, ycm, [Xedges,Yedges],
                                                    normed=True)
            
                heatmap_sum += heatmap_curr.T
                heatmap_cc += 1
            except KeyError:
                print('Could not create CM position histogram')
                continue
            
        
    heatmap_mean = heatmap_sum/heatmap_cc
    
    # plot results
    
    cax = ax_heatmap.imshow(heatmap_mean,extent=[Xedges[0], Xedges[-1],
                                           Yedges[0], Yedges[-1]],
                                           cmap='inferno',
                                           interpolation='gaussian',
                                           origin='low')
    ax_heatmap.xaxis.set_ticks([-0.5, 0.0, 0.5]) 
    ax_heatmap.yaxis.set_ticks([0, 1, 2, 3, 4])                                       
    ax_heatmap.set_xlabel('X [cm]',fontsize=LABEL_FONTSIZE)
    ax_heatmap.set_ylabel('Y [cm]',fontsize=LABEL_FONTSIZE)
          
    cbar = fig_heatmap.colorbar(cax)
    cbar.set_label('PDF')
    
    #plt.tight_layout()
    #plt.axis('equal')
    plt.show() 
#------------------------------------------------------------------------------