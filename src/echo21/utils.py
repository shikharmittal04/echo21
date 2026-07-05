'''
``utils``
=========
This module contains non-physics functions.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
import os
import warnings
import numpy as np
import pickle
from scipy.interpolate import CubicSpline
import h5py
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

    class
        echo21.pipeline    
    '''
    fullpath = os.path.join(filename, 'pipeline.pkl')
    with open(fullpath, 'rb') as inp:
        echo21obj = pickle.load(inp)

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

def _ensure_scalar_dict(d):
    '''
    Coerce every value in the dictionary to a plain Python scalar, so that user-supplied
    parameters (which may come in as e.g. ``np.array([67.0])``) behave like the pure numbers
    the rest of the code expects. Strings are left untouched.

    Arguments
    ---------
    d: dict
        Dictionary of parameters.

    Returns
    -------
    dict
        Same dictionary with all non-string values converted to scalars.
    '''
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = v
            continue

        arr = np.atleast_1d(v)
        if arr.size != 1:
            raise ValueError(f"Parameter '{k}' must be a scalar or a size-1 array; got an array of size {arr.size}.")
        out[k] = arr.item()

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
        if not pipe.params_df.empty:
            f.create_dataset(
                "params_df",
                data=pipe.params_df.to_records(index=False)
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
        #UVLF is only meaningful (and computed) for the semi-empirical SFRD.
        if pipe.sfrd['type'] == 'semi-emp':
            f.create_dataset("UVLF", data=pipe.UVLF)
        if pipe.dm_model == 'idm':
            f.create_dataset("Tx", data=pipe.Tx)
            f.create_dataset("v_bx", data=pipe.v_bx)
            
        # Failed parameter sets (if any)
        if total_failed > 0:
            warnings.warn(
                f"{total_failed}/{pipe.N_models} models failed "
                f"(solver did not converge). {n_succeeded} models saved."
            )
            f.create_dataset(
                "failed_params",
                data=gathered_failed_params.to_records(index=False)
            )
    return None

def _interp_over_z(flipped_Z, Z_eval, arr, z_axis):
    '''
    Cubic-spline interpolate `arr` onto `Z_eval` along `z_axis` (`flipped_Z` must be increasing).
    '''
    moved = np.flip(np.moveaxis(arr, z_axis, 0), axis=0)
    return np.moveaxis(CubicSpline(flipped_Z, moved)(Z_eval), 0, z_axis)

def load_results(filename, Z_eval = None):
    '''
    Read the output and return a dictionary of parameters, redshifts, global signal, etc. If you do not provide any redshift, the quantities are returned at their default redshift(s).

    Arguments
    ---------

    filename : str
        Full path to the output directory (``output_<YYYYMMDD-HHMMSS>``) which contains the HDF5 file ``echo21_output.h5``.

    Z_eval: array, optional
        array of :math:`1+z` values; can be in decreasing as well as increasing order. Default = ``None``

    Return
    ------

    dict
        params (pandas dataframe), MAB (numpy array), xe (numpy array), Q_Hii (numpy array), Tk (numpy array), Ts (numpy array), T21 (numpy array), xHI (numpy array), tau (numpy array), UVLF (numpy array).
        Also one_plus_z_cd (numpy array), and one_plus_z (numpy array) too if the run is not astro-only. Neither is included if `Z_eval` is given, since you already have those redshifts.
    '''
    pipe = load_pipeline(filename)
    is_astro = pipe.run_type == 'astro' #case of cosmic dawn to today, as opposed to dark ages to today

    main_flipped_Z = flipped_Z_cd if is_astro else flipped_Z_default
    Z_lim = Zstar if is_astro else Z_start

    fields = {'xe': pipe.xe, 'Tk': pipe.Tk, 'Ts': pipe.Ts, 'xHI': pipe.xHI, 'T21': pipe.T21}
    if pipe.dm_model == 'idm':
        fields['Tx'] = pipe.Tx
        fields['v_bx'] = pipe.v_bx

    #Q_Hii and UVLF are only ever defined on the cosmic dawn redshift grid, regardless of run type.
    cd_fields = {'Q_Hii': (pipe.Q_Hii, 1)}
    #UVLF is only meaningful (and computed) for the semi-empirical SFRD.
    if pipe.sfrd['type'] == 'semi-emp':
        cd_fields['UVLF'] = (pipe.UVLF, 2)

    if Z_eval is not None:
        Z_eval = np.asarray(Z_eval)
        if Z_eval[1] > Z_eval[0]:
            Z_eval = Z_eval[::-1]

        if Z_eval[0] > Z_lim or Z_eval[-1] < Z_end:
            raise ValueError(f'Your requested redshift values should satisfy {Z_lim} > 1+z > {Z_end}')

        fields = {k: _interp_over_z(main_flipped_Z, Z_eval, v, 1) for k, v in fields.items()}
        cd_fields = {k: _interp_over_z(flipped_Z_cd, Z_eval, v, ax) for k, (v, ax) in cd_fields.items()}
        z_dict = {} #user already has Z_eval, no point handing it back
    else:
        cd_fields = {k: v for k, (v, ax) in cd_fields.items()}
        z_dict = {'one_plus_z_cd': Z_cd} if is_astro else {'one_plus_z': Z_default, 'one_plus_z_cd': Z_cd}

    return {'params_df': pipe.params_df, 'MAB': pipe.MAB, 'tau': pipe.tau,
            **z_dict, **fields, **cd_fields}



__all__ = ['_ensure_array_dict',
           '_ensure_scalar_dict',
           '_split_params',
           '_grid_on_index',
           '_grid_off_index',
           '_create_output_dir',
           '_save_pipeline','load_pipeline','_save_results','load_results','_interp_over_z',
           '_print_banner','print_input','_write_summary',
           ]