# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:18:21 2017

@author: Fruit Flies
"""
#------------------------------------------------------------------------------
#
#   wlevel = wavelet decomposition threshold level (e.g. 2, 3 ,4)
#   wtype = wavelet type for denoising (e.g. 'haar', 'db4', sym5')
#   medfilt_window = window size for median filter used on wavelet-denoised 
#       data (e.g. 11, 13, ...)
#   mad_thresh = threshold for the median absolute deviation of slopes in 
#       segmented dataset. slopes below this value are considered possible bouts
#   var_user = user set variation fed into the PELT change point detector. 
#
#------------------------------------------------------------------------------
analysisParams = {'wlevel' : 4,  #3 , #4
             'wtype' : 'sym4', #'db3' , #db4
             'medfilt_window' : 5, #7 , #11
             'mad_thresh' : 3.0, #3.0, #-8 -10
             'var_user' : 0.5 ,
             'min_bout_duration': 2, #3 ,
             'min_bout_volume': 4, #6 ,
             'min_pos_slope': 0.5}
             
guiParams = {'bgcolor' : 'white' ,
             'listbgcolor': '#222222' , 
             'textcolor' : '#ffffff' ,
             'buttontextcolor': '#fff7bc' ,
             'buttonbgcolor': '#222222' ,
             'plotbgcolor' : '#3e3d42' , 
             'plottextcolor' : '#c994c7' ,
             'labelfontstr' : 'Helvetica 14 bold'}
             
trackingParams = {'fly_size_range' : [20, 100] ,
                  'bg_min_dist' : 40 , 
                  't_offset' : 10 ,
                  'label_fontsize' : 14 , 
                  'pix2cm' : None , 
                  'smoothing_factor' : 0.5 ,  
                  'medfilt_window' : 7 ,
                  'vel_thresh' : 0.05 , 
                  'hampel_k' : 11 , 
                  'hampel_sigma' : 3 }

initDirectories = ["F:\\Expresso GUI\\for_testing\\fewbigmealsevents", 
                     "F:\\Expresso GUI\\for_testing\\fewsmallmealevents",
                     "F:\\Expresso GUI\\for_testing\\manysmallmeals",
                     "F:\\Expresso GUI\\for_testing\\nodrinkingevents",
                     "F:\\Expresso GUI\\for_testing\\fromNilay",
                     "F:\\Expresso GUI\\for_testing\\long_data_files",
                     "F:\\Expresso GUI\\Saumya Annotations\\annotations_expresso_data",
                     "C:\\Users\Fruit Flies\\Documents\\Python Scripts\\Visual Expresso GUI\\dat\\bout_annotations\\data",
                     "H:\\v_expresso data\\DROPPED_FRAMES_DEBUG\\8hr_1mM",
                     "H:\\v_expresso data\\staci example video"]
             
             