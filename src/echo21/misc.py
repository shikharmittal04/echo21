'''
misc
====
This module contains non-physics functions.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
import pickle
import numpy as np
import scipy.integrate as scint
from scipy.interpolate import CubicSpline
from .const import Zstar, Msolar_by_Mpc3_to_kg_by_m3

#The following 2 functions will be useful if you want to save and load `pipeline` object.
def save_pipeline(obj, filename):
    '''    
    Save the class object :class:`pipeline` for later use. It will save the object in the path where you have all the other outputs from this package.
    
    Arguments
    ---------

    obj : class
        This should be the class object you want to save.
        
    filename : str
        Give a file name only to your object, not the full path. obj will be saved in the ``obj.path`` directory.
    
    '''
    if filename[-4:]!='.pkl': filename=filename+'.pkl'
    fullpath = obj.path+filename
    with open(fullpath, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    return None
    
def load_pipeline(filename):
    '''To load the class object :class:`pipeline`.
    
    Arguments
    ---------

    filename : str
        This should be the name of the file you gave in :func:`save_pipeline()` for saving class object :class:`pipeline`. Important: provide the full path for ``filename`` with the extension ``.pkl``.
        
    Returns
    -------

    class object    
    '''
    with open(filename, 'rb') as inp:
        echo21obj = pickle.load(inp)
    print('Loaded the echo21 pipeline class object.\n')
    return echo21obj
#--------------------------------------------------------------------------------------------

def print_banner():
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

def ensure_array_dict(d, ignore_keys=None):
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

def split_params(d):
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

def params_from_index(pipe, idx):
    inds = np.unravel_index(idx, pipe.shape)

    all_params = pipe.fixed_params.copy()
    varying_params_only = {}

    for name, i, arr in zip(pipe.param_names, inds, pipe.param_arrays):
        all_params[name] = arr[i]
        varying_params_only[name] = arr[i]

    return all_params, varying_params_only

def write_summary(pipe, elapsed_time):
    '''
    Given the elapsed time of the code execution write the main summary of the run.
    '''
    sumfile = pipe.path+"summary_"+pipe.timestamp+".txt"
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
    myfile.write('\nTime stamp: '+pipe.formatted_timestamp)
    myfile.write('\n\nExecution time: %.2f seconds' %elapsed_time) 
    myfile.write('\n\n')
    myfile.write('Dark matter type: {}\n'.format(pipe.dm_model))
    myfile.write('\nSimulation type: '+pipe.message)
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

def frac_diff_temp_to_temp(funcs_obj, Z, frac_diff_temp):
    '''
    Given :math:`\\delta_T`, compute :math:`T_{\\mathrm{k}}`.

    Arguments
    ---------
    funcs_obj : funcs
        funcs class object
    
    Z : float
        redshift, :math:`1+z`

    frac_diff_temp: float
        :math:`\\delta_T`
    
    Returns
    -------
        :math:`T_{\\mathrm{k}} = (1+\\delta_T)T_{\\gamma}(z)`
    '''
    Tgamma = funcs_obj.basic_cosmo_Tcmb(Z)
    Tk = (1 + frac_diff_temp)*Tgamma
    return Tk

def build_fcoll_spline(funcs_obj, n_points=100):
    '''
    Precompute the collapse fraction on a redshift grid and return a CubicSpline.
    Called once per funcs() instantiation for non-press74 HMFs, replacing the
    per-ODE-step numerical integration with a cheap spline evaluation.

    Arguments
    ---------
    funcs_obj : funcs
        An initialised funcs instance whose cosmology and HMF are already set up.

    n_points : int
        Number of redshift grid points. Default 100.

    Returns
    -------
    CubicSpline
        Spline of f_coll(Z) over Z in [1, Zstar], with Z = 1+z.
    '''
    Z_grid = np.linspace(1, Zstar, n_points)
    f_grid = np.zeros(n_points)
    norm = Msolar_by_Mpc3_to_kg_by_m3 / (funcs_obj.Om_m * funcs_obj.basic_cosmo_rho_crit())
    for i, Zv in enumerate(Z_grid):
        M_space = np.logspace(np.log10(funcs_obj.m_min(Zv) / funcs_obj.h100), 18, 1500)
        f_grid[i] = scint.simpson(funcs_obj.dndlnM(M=M_space, Z=Zv), x=M_space)
    f_grid *= norm
    return CubicSpline(Z_grid, f_grid)