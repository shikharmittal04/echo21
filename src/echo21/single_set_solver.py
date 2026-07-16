"""
This module contains the functions required for running IGM solver for one set of parameters.
"""

import numpy as np
from .funcs import funcs
from .const import *
from .config import config

def cosmic_dawn_beyond(params_dict, *initial_conditions, dm_model='CDM'):
    '''
    Runs IGM solver starting from cosmic dawn until today, i.e., for :math:`z_{\\star} > z`.

    Arguments
    ---------
    params_dict: dict
        One single dictionary containing all the parameters.
    
    initial_conditions: tuple
        Initial conditions for the IGM solver.
        
    dm_model: str
        Model of dark matter. Default is 'CDM'.

    Returns
    -------
    21-cm signal, global-averaged neutral hydrogen fraction, and optical depth.
    '''
    configuration = config(params_dict, dm_model=dm_model)
    funcs_obj = funcs(configuration)
    
    sol_cd = funcs_obj.ivp.igm_solver(Z_CD, *initial_conditions, eqns_func = funcs_obj.ivp.igm_eqns_cd)
    
    xe = sol_cd[0]
    Tk = sol_cd[1]

    Q_Hii = funcs_obj.eor.QHii

    xHI = (1 - Q_Hii) * (1 - xe)   # Globally-averaged neutral hydrogen fraction

    Ts = funcs_obj.hyfi.spin_temp(Z=Z_CD,xe=xe,Tk=Tk) #Spin temperature

    T21 = funcs_obj.hyfi.twentyone_cm(Z=Z_CD,xHI=xHI,Ts=Ts) #21-cm signal
    
    tau = funcs_obj.eor.cmb_tau(Z_STAR) #CMB Optical depth

    #UV LF is only meaningful for the semi-empirical SFRD; skip the extra computation otherwise.
    UVLF = funcs_obj.uvlf.lum_func(MUV_default, Z_CD) if params_dict['type'] == 'semi-emp' else None

    results_tuple = (xe, Q_Hii, xHI, Tk, Ts, T21, tau, UVLF)

    if dm_model == 'IDM':
        Tx = sol_cd[2]
        v_bx = np.exp(sol_cd[3])
        results_tuple += (Tx, v_bx)
    
    return results_tuple
#================================================================================

def dark_ages_to_today(params_dict, *initial_conditions, dm_model='CDM'):
    '''
    Runs IGM solver starting from :math:`z=1500` to today. Note that arguments ``xe_init`` and ``Tk_init`` are provided only to match the signiture for :py:func:`cosmic_dawn_beyond`. The initial condition is calculated from Saha's equation at :math:`z=1500`. 

    Arguments
    ---------
    params_dict: dict
        One single dictionary containing all the parameters.
    
    initial_conditions: tuple
        Initial conditions for the IGM solver. This is not used in this function, but is provided only to match the signiture for :py:func:`cosmic_dawn_beyond`. The initial condition is calculated from Saha's equation at :math:`z=1500`.

    dm_model: str
        Model of dark matter. Default is 'CDM'.

    Returns
    -------
    21-cm signal, global-averaged neutral hydrogen fraction, and optical depth.
    '''
    #we solve this case in two parts. First DA then CD and later.
    configuration = config(params_dict, dm_model=dm_model)
    funcs_obj = funcs(configuration)
    
    ic = funcs_obj.ivp.initial_conditions()

    sol_da = funcs_obj.ivp.igm_solver(Z_DA, *ic, eqns_func = funcs_obj.ivp.igm_eqns_da)
    #in dark ages we solver for the transformed gas temperature. So we need to convert to physical temperature
    sol_da[1] = funcs_obj.ivp._logratio_to_temp(Z_DA, sol_da[1])

    #initial conditions for cosmic dawn solver
    ic_cd = tuple(s[-1] for s in sol_da)
    sol_cd = funcs_obj.ivp.igm_solver(Z_CD, *ic_cd, eqns_func = funcs_obj.ivp.igm_eqns_cd)

    #join the DA and CD part. Note that last entry of DA is same as first entry of CD. So we skip the last element of DA solution.
    xe = np.concatenate([sol_da[0][:-1], sol_cd[0]])
    Tk = np.concatenate([sol_da[1][:-1], sol_cd[1]])
    
    Q_Hii = funcs_obj.eor.QHii #HII region volume filling factor (CD only)

    xHI = (1 - xe) #Globally-averaged neutral hydrogen fraction
    xHI[N_DA-1:] *= (1 - Q_Hii)

    Ts = funcs_obj.hyfi.spin_temp(Z=Z_default,xe=xe,Tk=Tk) #Spin temperature
    
    T21 = funcs_obj.hyfi.twentyone_cm(Z=Z_default,xHI=xHI,Ts=Ts)  #21-cm signal

    tau = funcs_obj.eor.cmb_tau(Z_STAR) #CMB optical depth

    #UV LF is only meaningful for the semi-empirical SFRD; skip the extra computation otherwise.
    UVLF = funcs_obj.uvlf.lum_func(MUV_default, Z_CD) if params_dict['type'] == 'semi-emp' else None

    results_tuple = (xe, Q_Hii, xHI, Tk, Ts, T21, tau, UVLF)

    if dm_model == 'IDM':
        Tx = np.concatenate([sol_da[2][:-1], sol_cd[2]])
        v_bx = np.concatenate([np.exp(sol_da[3][:-1]), np.exp(sol_cd[3])])
        results_tuple += (Tx, v_bx)
    
    return results_tuple