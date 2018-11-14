# -*- coding: utf-8 -*-
"""
Created on Thu Nov 08 12:33:38 2018

@author: Fruit Flies
"""
import numpy as np

A,B,C = np.mgrid[0:5,0:5,0:6]

for ai, bi, ci in np.nditer([A,B,C]):
    print("{}, {}, {}".format(ai,bi,ci))