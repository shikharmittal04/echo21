'''
utils
=====
This module contains non-physics functions.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
import os
import numpy as np
import pickle
from scipy.interpolate import CubicSpline
import pandas as pd
import h5py
import sys
from time import localtime, strftime
try:
    import classy
except ImportError:
    print("")
from .const import *

#The following 2 functions will be useful if you want to save and load `pipeline` object.
def _save_pipeline(obj):
    '''    
    Save the class object :class:`pipeline` for later use. It will save the object in the path where you have all the other outputs from this package.
    
    Arguments
    ---------

    obj : class
        This should be the class object you want to save.    
    '''
    filename = 'pipeline.pkl'

    fullpath = os.path.join(obj.path, filename)
    with open(fullpath, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    return None
    
def load_pipeline(filename):
    '''To load the class object :class:`pipeline`.
    
    Arguments
    ---------

    filename : str
        Full path to the output directory (``output_<YYYYMMDD-HHMMSS>``) which contains the pickle file `pipeline.pkl`.
        
    Returns
    -------

    class object    
    '''
    fullpath = os.path.join(filename, 'pipeline.pkl')
    with open(fullpath, 'rb') as inp:
        echo21obj = pickle.load(inp)
    print('Loaded the echo21 pipeline class object.\n')
    return echo21obj
#--------------------------------------------------------------------------------------------

def _create_output_dir(pipe):
    '''
    Create an output directory if it does not exist.
    
    Arguments
    ---------
    pipe : class
        The pipeline object for which to create the output directory.
    '''
    if pipe.cpu_ind==0:
        if os.path.isdir(pipe.path)==False:
            print('\nThe requested directory does not exist. Creating ',pipe.path)
            os.mkdir(pipe.path)
        
        pipe.timestamp = strftime("%Y%m%d-%H%M%S", localtime())
        pipe.path = pipe.path + 'output_'+pipe.timestamp+'/'
        os.mkdir(pipe.path)
    return None

def _print_banner():
    banner = """\n\033[94m
    ███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  ██╗
    ██╔════╝██╔════╝██║  ██║██╔═══██╗╚════██╗███║
    █████╗  ██║     ███████║██║   ██║ █████╔╝╚██║
    ██╔══╝  ██║     ██╔══██║██║   ██║██╔═══╝  ██║
    ███████╗╚██████╗██║  ██║╚██████╔╝███████╗ ██║
    ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═╝
    Copyright 2026, Shikhar Mittal.                                     
    \033[00m\n"""
    print(banner)
    return None

def _ensure_array_dict(d, ignore_keys=None):
    '''
    Ensure that all the parameters are expressed as arrays except string-valued SFRD parameters.

    Arguments
    ---------
    d: dict
        Dictionary of parameters.
    
    ignore_keys: str
        Keys to ignore for conversion to arrays.
    
    Returns
    -------
    dict
        Same dictionary but with all values expressed as arrays.
    '''
    if ignore_keys is None:
        ignore_keys = []

    out = {}
    for k, v in d.items():
        if k in ignore_keys:
            out[k] = v
        else:
            out[k] = np.atleast_1d(v)
    return out

def _split_params(d):
    '''
    Split the given a dictionary into multi-valued and single-valued parameters dictionaries.

    Arguments
    ---------
    d: dict
        Dictionary of parameters.
    
    Returns
    -------
    dict, dict
        Dictionary of multi-valued and single-valued parameters, respectively.
    '''
    varying = {}
    fixed = {}

    for k, v in d.items():

        # strings or non-array types → fixed
        if isinstance(v, str):
            fixed[k] = v
            continue

        arr = np.atleast_1d(v)

        if arr.size > 1:
            varying[k] = arr
        else:
            fixed[k] = arr.item()

    return varying, fixed

def _grid_on_index(pipe, idx):
    inds = np.unravel_index(idx, pipe.shape)

    all_params = pipe.fixed_params.copy()
    varying_params_only = {}

    for name, i, arr in zip(pipe.param_names, inds, pipe.param_arrays):
        all_params[name] = arr[i]
        varying_params_only[name] = arr[i]

    return all_params, varying_params_only

def _grid_off_index(pipe, idx):

    all_params = pipe.fixed_params.copy()
    varying_params_only = {}

    for name, arr in zip(pipe.param_names, pipe.param_arrays):
        all_params[name] = arr[idx]
        varying_params_only[name] = arr[idx]

    return all_params, varying_params_only

def _write_summary(pipe, elapsed_time):
    '''
    Given the elapsed time of the code execution write the main summary of the run.
    '''
    sumfile = os.path.join(pipe.path, "summary_" + pipe.timestamp + ".txt")
    formatted_timestamp = pipe.timestamp[9:11]+':'+pipe.timestamp[11:13]+':'+pipe.timestamp[13:15]+' '+pipe.timestamp[6:8]+'/'+pipe.timestamp[4:6]+'/'+ pipe.timestamp[:4]


    myfile = open(sumfile, "w")
    myfile.write('''\n███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  ██╗
██╔════╝██╔════╝██║  ██║██╔═══██╗╚════██╗███║
█████╗  ██║     ███████║██║   ██║ █████╔╝╚██║
██╔══╝  ██║     ██╔══██║██║   ██║██╔═══╝  ██║
███████╗╚██████╗██║  ██║╚██████╔╝███████╗ ██║
╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═╝\n''')
    myfile.write('Shikhar Mittal, 2026\n')
    myfile.write('\nThis is output_'+pipe.timestamp)
    myfile.write('\n------------------------------\n')
    myfile.write('\nTime stamp: '+formatted_timestamp)
    myfile.write('\n\nExecution time: %.2f seconds' %elapsed_time) 
    myfile.write('\n\n')
    myfile.write('Dark matter type: {}\n'.format(pipe.dm_model))
    myfile.write('\nSimulation type: '+pipe.message)
    myfile.write('\nGrid on: {}'.format(pipe.grid_on))
    myfile.write('\n\nParameters given:\n')
    myfile.write('-----------------')
    [myfile.write('\n{} = {}'.format(k, v)) for k, v in pipe.cosmo.items()]
    myfile.write('\n')
    [myfile.write('\n{} = {}'.format(k, v)) for k, v in pipe.astro.items()]
    
    myfile.write('\n\nSFRD')
    [myfile.write('\n  {} = {}'.format(k, v)) for k, v in pipe.sfrd.items()]
    
    myfile.write('\n')
    return myfile

def print_input(pipe):
    '''Prints the input parameters you gave.'''

    print(f'Dark matter type: {pipe.dm_model}')

    print('\n\033[93mParameters given:\n')
    print('-----------------\n')
    [print('\n{} = {}'.format(k, v)) for k, v in pipe.cosmo.items()]
    print('\n')
    [print('\n{} = {}'.format(k, v)) for k, v in pipe.astro.items()]

    print('\n\nSFRD')
    [print('\n  {} = {}'.format(k, v)) for k, v in pipe.sfrd.items()]
    print('\033[00m\n')

    return None

def _get_As_for_sig8(class_set):
    '''
    For the given cosmological parameters (including sigma8), compute the primordial power spectrum amplitude.

    Arguments
    ---------
    class_set: dict
        A dictionary of cosmological parameters including user provided sigma8
    
    Returns
    -------
    
    float
        A_s

    '''
    class_obj = classy.Class()
    class_obj.set(class_set)
    class_obj.compute()

    A_s = class_obj.get_current_derived_parameters(['A_s'])['A_s']
    class_obj.struct_cleanup()
    class_obj.empty()

    return A_s


def _save_results(pipe, total_failed=0, gathered_failed_params=None, n_succeeded=None):
    '''
    Save the results of the simulation to an HDF5 file.

    Arguments
    ---------
    param_df: pandas.DataFrame
        DataFrame containing the parameters for each model.
    
    xe, Q_Hii, Tk, Ts, T21, xHI, tau: numpy.ndarray
        Arrays containing the evolution histories for each model.
    
    UVLF: numpy.ndarray
        Array containing the UV luminosity function for each model.
    
    total_failed: int, optional
        Total number of models that failed to converge. Default is 0.
    
    gathered_failed_params: pandas.DataFrame, optional
        DataFrame containing the parameters of the failed models. Default is None.
    
    n_succeeded: int, optional
        Number of models that succeeded. Default is None.
    '''
    save_path = os.path.join(pipe.path, 'echo21_output.h5')

    with h5py.File(save_path, "w") as f:

        # Parameter table (skip when there are no varying parameters to record)
        if not pipe.param_df.empty:
            f.create_dataset(
                "params_df",
                data=pipe.param_df.to_records(index=False)
            )

        # Redshift and magnitude grids
        f.create_dataset("one_plus_z", data=Z_default)
        f.create_dataset("one_plus_z_cd", data=Z_cd)
        f.create_dataset("MAB", data=MAB_default)

        # Evolution histories
        f.create_dataset("xe", data=pipe.xe)
        f.create_dataset("Q_Hii", data=pipe.Q_Hii)
        f.create_dataset("Tk", data=pipe.Tk)
        f.create_dataset("Ts", data=pipe.Ts)
        f.create_dataset("T21", data=pipe.T21)
        f.create_dataset("xHI", data=pipe.xHI)
        f.create_dataset("tau", data=pipe.tau)
        f.create_dataset("UVLF", data=pipe.UVLF)
        if pipe.dm_model == 'idm':
            f.create_dataset("Tx", data=pipe.Tx)
            f.create_dataset("v_bx", data=pipe.v_bx)
            
        # Failed parameter sets (if any)
        if total_failed > 0:
            print(
                f"\n\033[33mWarning: {total_failed}/{pipe.N_models} models failed "
                f"(solver did not converge). {n_succeeded} models saved.\033[00m"
            )
            f.create_dataset(
                "failed_params",
                data=gathered_failed_params.to_records(index=False)
            )
    return None

def load_results(filename, Z_eval = None):
    '''
    Read the output and return a dictionary of parameters, redshifts, global signal, etc. If you do not provide any redshift, the quantities are returned at their default redshift.

    Arguments
    ---------
    
    filename : str
        Full path to the output directory (``output_<YYYYMMDD-HHMMSS>``) which contains the HDF5 file ``echo21_output.h5``.

    Z_eval: array, optional
        array of :math:`1+z` values; can be in decreasing as well as increasing order. Default = ``None``
    
    Return
    ------
    
    dict
        params (pandas dataframe), one_plus_z (numpy array), MAB (numpy array), xe (numpy array), Q_Hii (numpy array), Tk (numpy array), Ts (numpy array), T21 (numpy array), xHI (numpy array), tau (numpy array), UVLF (numpy array)
    '''
    
    h5file = os.path.join(filename, 'echo21_output.h5')
    with h5py.File(h5file, "r") as f:
        try:
            params_df     = pd.DataFrame(f["params_df"][:])
        except KeyError:
            params_df = pd.DataFrame()
        one_plus_z = f["one_plus_z"][:]
        one_plus_z_cd = f["one_plus_z_cd"][:]
        MAB        = f["MAB"][:]
        xe         = f["xe"][:]
        Q_Hii      = f["Q_Hii"][:]
        Tk         = f["Tk"][:]
        Ts         = f["Ts"][:]
        T21        = f["T21"][:]
        xHI        = f["xHI"][:]
        tau        = f["tau"][:]
        UVLF       = f["UVLF"][:]   # shape (N_params, nMAB, nZ)
        try:
            Tx     = f["Tx"][:]
            v_bx   = f["v_bx"][:]
            dm_model = 'idm'
        except KeyError:
            pass

    if len(xe[0,:]) == 2300: #case of dark ages to today
        Z_lim, flipped_Z = Z_start, flipped_Z_default
    elif len(xe[0,:]) == 300: #case of cosmic dawn to today
        Z_lim, flipped_Z = Zstar, flipped_Z_cd

    if Z_eval is not None:
        if Z_eval[0] > Z_lim or Z_eval[-1] < Z_end:
            print(f'\033[31mYour requested redshift values should satisfy {Z_lim} > 1+z > {Z_end}')
            print('Terminating ...\033[00m')
            sys.exit()

        Z_eval = np.asarray(Z_eval)
        if Z_eval[1] > Z_eval[0]:
            Z_eval = Z_eval[::-1]

        # Z must be on axis 0 for CubicSpline; transpose so Z leads, flip, interpolate, restore.
        xe    = CubicSpline(flipped_Z, np.flip(xe.T,                    axis=0))(Z_eval).T
        Tk    = CubicSpline(flipped_Z, np.flip(Tk.T,                    axis=0))(Z_eval).T
        Ts    = CubicSpline(flipped_Z, np.flip(Ts.T,                    axis=0))(Z_eval).T
        xHI   = CubicSpline(flipped_Z, np.flip(xHI.T,                   axis=0))(Z_eval).T
        T21   = CubicSpline(flipped_Z, np.flip(T21.T,                   axis=0))(Z_eval).T
        
        #Q_Hii = CubicSpline(flipped_Z_cd, np.flip(Q_Hii.T,              axis=0))(Z_eval).T
        #UVLF  = CubicSpline(flipped_Z_cd, np.flip(UVLF.transpose(2, 0, 1), axis=0))(Z_eval).transpose(1, 2, 0)
        one_plus_z = Z_eval


    return_dic = {'params_df': params_df, 'one_plus_z': one_plus_z, 'MAB': MAB,
                'xe': xe, 'Q_Hii': Q_Hii, 'Tk': Tk, 'Ts': Ts,
                'T21': T21, 'xHI': xHI, 'tau': tau, 'UVLF': UVLF}
    
    if dm_model == 'idm':
        if Z_eval is not None:
            Tx    = CubicSpline(flipped_Z, np.flip(Tx.T,                    axis=0))(Z_eval).T
            v_bx  = CubicSpline(flipped_Z, np.flip(v_bx.T,                  axis=0))(Z_eval).T
        return_dic['Tx'] = Tx
        return_dic['v_bx'] = v_bx

    return return_dic



__all__ = ['_ensure_array_dict',
           '_split_params',
           '_grid_on_index',
           '_grid_off_index',
           '_create_output_dir',
           '_save_pipeline','load_pipeline','_save_results','load_results',
           '_print_banner','print_input','_write_summary',
           ]