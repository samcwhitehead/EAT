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
analysisParams = {'wlevel' : 6,  #3 , #4
             'wtype' : 'db3', #'db2' 'db3' , #db4 #sym5
             'medfilt_window' : 3, #7 , #11
             'mad_thresh' : 3.0, #3.0, #-8 -10
             'var_user' : 0.5 ,
             'min_bout_duration': 1, #3 ,
             'min_bout_volume': 6, #6 ,
             'min_pos_slope': 0.5 ,
             'w_coeff_thresh': 0.05 , 
             'hampel_k' : 9 , 
             'hampel_sigma' : 2 ,
             'pos_der_thresh' : 10.0 ,
             'feeding_dist_max' : 0.5 , #cm #0.5
             'feeding_dist_min' : 0.35 , # cm #0.375
             'feeding_move_frac_thresh' : 0.5 ,
             'feeding_vel_max' : 0.1 } 
             
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
            'MOG_var_thresh' : 125 , 
            'morph_size_1' : 3 ,
            'morph_size_2' : 5 , 
            'fly_pix_val_max' : 100 ,
            't_offset' : 10 ,
            'min_pix_thresh_guess' : 20 ,
            'min_pix_thresh' : 10 , 
            'kalman_process_noise' : 0.003 , 
            'kalman_meas_noise' : 1.0 ,  #0.1
            'n_missing_max' : 2 , #3
            'n_init_ignore' : 25 , 
            'label_fontsize' : 14 , 
            'pix2cm' : None , 
            'smoothing_factor' : 0.1 ,  
            'medfilt_window' : 7 ,
            'vel_thresh' : 0.05 , #0.05
            'hampel_k' : 11 , 
            'hampel_sigma' : 3 }

initDirectories = ["H:\\v_expresso data\\Feeding_annotation_videos",
                   "F:\\Expresso GUI\\for_testing\\fewbigmealsevents", 
                     "F:\\Expresso GUI\\for_testing\\fewsmallmealevents",
                     "F:\\Expresso GUI\\for_testing\\manysmallmeals",
                     "F:\\Expresso GUI\\for_testing\\nodrinkingevents",
                     "F:\\Expresso GUI\\for_testing\\fromNilay",
                     "F:\\Expresso GUI\\for_testing\\long_data_files",
                     "F:\\Expresso GUI\\Saumya Annotations\\annotations_expresso_data",
                     "C:\\Users\Fruit Flies\\Documents\\Python Scripts\\Visual Expresso GUI\\dat\\bout_annotations\\data",
                     "H:\\v_expresso data\\MATRIX_DEBUG\\8hr_1mM",
                     "H:\\v_expresso data\\staci example video",
                     "D:\\v_expresso_data\\Feeding_annotation_videos"]
             
             