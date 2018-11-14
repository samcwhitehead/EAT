# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:00:30 2017

@author: Saumya


"""
#------------------------------------------------------------------------------
import sys, os
import cv2
import numpy as np
import glob
import h5py
#------------------------------------------------------------------------------
# constants/terms related to calibration

N_COL_PTS = 9 #number of checkerboard COLUMN points to find
N_ROW_PTS = 7 #number of checkerboard ROW points to find
CORNER_SUBPIX_WIN = (11,11) #window size to use for corner detection refinement

# path for the directory containing checkerboard images
PATHNAME = os.path.dirname(sys.argv[0]) 
PATHNAME_ABS = os.path.abspath(PATHNAME)
PATHNAME_FULL = os.path.join(PATHNAME_ABS,'CalibImages')
PATHNAME_IM = os.path.join(PATHNAME_FULL,'\*.jpg')

#------------------------------------------------------------------------------
# function to get camera calibration coefficients based on checkerboard images
def get_calib_coeff(pathname=PATHNAME_IM, n_col_pts=N_COL_PTS, 
                    n_row_pts=N_ROW_PTS, corner_subpix_win=CORNER_SUBPIX_WIN):
                        
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((N_ROW_PTS*N_COL_PTS,3), np.float32)
    objp[:,:2] = np.mgrid[:N_COL_PTS,:N_ROW_PTS].T.reshape(-1,2)
    
    objpoints = [] 
    imgpoints = [] 
    
    pathname = os.path.dirname(sys.argv[0]) 
    pathname_full = os.path.abspath(pathname)
    pathname_im = os.path.join(pathname_full,'CalibImages\*.jpg')
    images = glob.glob(pathname_im)
    
    for imagefile in images:
        img = cv2.imread(imagefile)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (N_COL_PTS,N_ROW_PTS),None)
        
       
        if ret == True:
            objpoints.append(objp)
        
            cv2.cornerSubPix(gray,corners,CORNER_SUBPIX_WIN,(-1,-1),criteria)
            imgpoints.append(corners)
            
            cv2.drawChessboardCorners(gray, (N_COL_PTS,N_ROW_PTS), corners,ret)
            cv2.imshow('img',gray)
            cv2.waitKey(100)
            
    cv2.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    #reprojection error
    mean_error = 0
    
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    total_error =  mean_error/len(objpoints)
    print "total error: ", total_error
    
    return (ret, mtx, dist, rvecs, tvecs,total_error)
    
#------------------------------------------------------------------------------
# run a calibration    
if __name__ == "__main__":
    
    saveFlag = True # save calibration results?
    ret, mtx, dist, rvecs, tvecs, total_error = get_calib_coeff()         
    if saveFlag:
        savename = os.path.join(PATHNAME_FULL,'calib_coeff.hdf5')
        with h5py.File(savename,'w') as f:
            f.create_dataset('mtx',  data=mtx)
            f.create_dataset('dist', data=dist)
            f.create_dataset('rvecs', data=rvecs)
            f.create_dataset('tvecs', data=tvecs)
            f.create_dataset('error', data=total_error)
        
    
    #undistorting
    
    #undistortIm = cv2.imread('C:\\Users\\Saumya\\Google Drive\\Yapici Lab\\Camera Calibration\\CalibImages\\toTest.jpg')
    #h,  w = undistortIm.shape[:2]
    #newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    
    #dst = cv2.undistort(undistortIm, mtx, dist, None, newcameramtx)
    #cv2.imwrite('calibresult_toTest.png',dst)    