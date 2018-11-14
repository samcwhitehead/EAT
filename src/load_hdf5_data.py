# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 10:47:57 2017

@author: Fruit Flies
"""
import h5py
import numpy as np
import sys

#------------------------------------------------------------------------------
def load_hdf5(filename,grpnum,dsetnum):
    dset = np.array([])
    t = np.array([])
    f = h5py.File(filename,'r')
    fileKeyNames = list(f.keys())
    
    if sys.version_info[0] < 3:
        strchecktype = unicode
    else:
        strchecktype = str
        
    if isinstance(grpnum,strchecktype):
        try:        
            grp = f[grpnum]
        except KeyError:
            print("Error: group name is invalid")
            return (dset, t)
            #return dset
    else:    
        try:
            grp = f.require_group(fileKeyNames[grpnum])
        except IndexError:
            print("Error: group index out of bounds")
            return (dset, t)
            #return dset
        
    groupKeyNames = list(grp.keys())
    
    if isinstance(dsetnum,strchecktype):
        try:
            dset = grp[dsetnum]
        except KeyError:
            print("Error: channel name is invalid")
            return (dset, t) 
            #return dset
    else:    
        try:
            dset = grp.get(groupKeyNames[dsetnum])
        except IndexError:
            print("Error: dataset index out of bounds")
            return (dset, t)
            #return dset
    
    try: 
        t = f['sample_t']
        if t.size != dset.size:
            N_banks = len(fileKeyNames) - 1
            t = t[::N_banks]
            
    except KeyError:
        print("Error: no sample time in hdf5 file")    
        
    t = t[()]    
    dset = dset[()]
    return (dset, t)