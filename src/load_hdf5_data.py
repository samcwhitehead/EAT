# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 10:47:57 2017

@author: Fruit Flies
"""
import h5py
import numpy as np
import sys
import os

# ------------------------------------------------------------------------------
# function to add dataset with units to h5py file object writer
def my_add_h5_dset(f, grp_name, dset_name, data, units='', long_name=''):
    ds = f.create_dataset('{}/{}'.format(grp_name, dset_name), data=data)
    # add units and long name
    ds.attrs['units'] = units
    ds.attrs['long_name'] = long_name

    return ds


# -------------------------------------------------------------------------------------
# function to add dataset with units to dictionary "d" where dset comes from hdf5 file
def my_add_dset_to_dict(d, key, f, grp_name, dset_name, scalar_flag=False):
    # read data set
    ds = f[grp_name][dset_name]

    # read data
    if scalar_flag:
        d[key] = ds[()]
    else:
        d[key] = ds[:]

    # get units and long name
    d['{}_units'.format(key)] = ds.attrs.get('units')
    d['{}_long_name'.format(key)] = ds.attrs.get('long_name')

    return d


# ------------------------------------------------------------------------------
# function to add data with units to dictionary "d"
def my_add_data_to_dict(dct, key, data, units='', long_name=''):
    # add data to dict
    dct[key] = data

    # also add units and longname
    dct['{}_units'.format(key)] = units
    dct['{}_long_name'.format(key)] = long_name

    return dct


#------------------------------------------------------------------------------
def load_hdf5(filename,grpnum,dsetnum):
    dset = np.array([])
    t = np.array([])
    # check that file exists
    if not os.path.exists(filename):
        print("Error: file does not exist")
        return dset, t

    f = h5py.File(filename, 'r')
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
            return dset, t
    else:    
        try:
            grp = f.require_group(fileKeyNames[grpnum])
        except IndexError:
            print("Error: group index out of bounds")
            return dset, t

    groupKeyNames = list(grp.keys())
    
    if isinstance(dsetnum, strchecktype):
        try:
            dset = grp[dsetnum]
        except KeyError:
            print("Error: channel name is invalid")
            return dset, t

    else:    
        try:
            dset = grp.get(groupKeyNames[dsetnum])
        except IndexError:
            print("Error: dataset index out of bounds")
            return dset, t

    try: 
        t = f['sample_t']
        if t.size != dset.size:
            N_banks = len(fileKeyNames) - 1
            t = t[::N_banks]
            
    except KeyError:
        print("Error: no sample time in hdf5 file")    
        
    t = t[()]    
    dset = dset[()]
    return dset, t