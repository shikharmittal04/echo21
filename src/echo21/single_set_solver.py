"""
This module contains the functions required for running IGM solver for one set of parameters.
"""

import numpy as np
from .echofuncs import funcs
from .uvlf import uvlf
from .const import  *

def cosmic_dawn_beyond(params_dict, *initial_conditions, dm_model='CDM'):
    '''
    Runs IGM solver starting from cosmic dawn until today, i.e., for :math:`z_{\\star} > z`.

    Arguments
    ---------
    params_dict: dict
        Dictionary containing all the parameters.
    
    initial_conditions: tuple
        Initial conditions for the IGM solver.
        
    dm_model: str
        Model of dark matter. Default is 'CDM'.

    Returns
    -------
    21-cm signal, global-averaged neutral hydrogen fraction, and optical depth.
    '''
    myobj_cd = funcs(params_dict, dm_model=dm_model)
    sol_cd = myobj_cd.igm_solver(Z_cd, *initial_conditions, eqns_func=myobj_cd.igm_eqns_cd)
    
    xe_cd = sol_cd[0]
    Tk_cd = sol_cd[1]
    
    Q_cd = myobj_cd.QHii

    Ts = myobj_cd.hyfi_spin_temp(Z=Z_cd,xe=xe_cd,Tk=Tk_cd) #Spin temperature

    T21 = myobj_cd.hyfi_twentyone_cm(Z=Z_cd,xe=xe_cd,Q=Q_cd,Ts=Ts) #21-cm signal

    xHI_cd = (1 - Q_cd) * (1 - xe_cd)   # Globally-averaged neutral hydrogen fraction
    
    tau_cd = myobj_cd.reion_tau(Zstar) #CMB Optical depth

    UVLF = uvlf(myobj_cd).lum_func(MAB_default, Z_cd) #UV LF

    return T21, xHI_cd, tau_cd, UVLF
#================================================================================

def dark_ages_to_today(params_dict, *initial_conditions, dm_model='CDM'):
    '''
    Runs IGM solver starting from :math:`z=1500` to today. Note that arguments ``xe_init`` and ``Tk_init`` are provided only to match the signiture for :py:func:`cosmic_dawn_beyond`. The initial condition is calculated from Saha's equation at :math:`z=1500`. 

    Arguments
    ---------
    params_dict: dict
        Dictionary containing all the parameters.
    
    initial_conditions: tuple
        Initial conditions for the IGM solver. This is not used in this function, but is provided only to match the signiture for :py:func:`cosmic_dawn_beyond`. The initial condition is calculated from Saha's equation at :math:`z=1500`.

    dm_model: str
        Model of dark matter. Default is 'CDM'.

    Returns
    -------
    21-cm signal, global-averaged neutral hydrogen fraction, and optical depth.
    '''
    #we solve this case in two parts. First DA then CD and later.
    myobj = funcs(params_dict, dm_model=dm_model)
    ic = myobj.initial_conditions()

    sol_da = myobj.igm_solver(Z_da, *ic, eqns_func=myobj.igm_eqns_da)
    #in dark ages we solver for the transformed gas temperature. So we need to convert to physical temperature
    sol_da[1] = myobj._logratio_to_temp(Z_da, sol_da[1])

    #initial conditions for cosmic dawn solver
    ic_cd = tuple(s[-1] for s in sol_da)
    sol_cd = myobj.igm_solver(Z_cd, *ic_cd, eqns_func=myobj.igm_eqns_cd)

    #join the DA and CD part. Note that last entry of DA is same as first entry of CD. So we skip the last element of DA solution.
    xe = np.concatenate([sol_da[0][:-1], sol_cd[0]])
    Tk = np.concatenate([sol_da[1][:-1], sol_cd[1]])
    
    Q_Hii = myobj.QHii
    Q_Hii = np.concatenate((np.zeros(len(Z_da)-1), Q_Hii))

    Ts = myobj.hyfi_spin_temp(Z=Z_default,xe=xe,Tk=Tk) #Spin temperature
    
    T21 = myobj.hyfi_twentyone_cm(Z=Z_default,xe=xe,Q=Q_Hii,Ts=Ts)  #21-cm signal
    
    xHI = (1 - Q_Hii) * (1 - xe) #Globally-averaged neutral hydrogen fraction
    
    tau = myobj.reion_tau(Zstar) #CMB optical depth

    UVLF = uvlf(myobj).lum_func(MAB_default, Z_cd) #UV LF

    return T21, xHI, tau, UVLF