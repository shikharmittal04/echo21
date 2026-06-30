"""
This module contains the functions required for running IGM solver for one set of parameters.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from .echofuncs import funcs
from .const import  Zstar, Z_cd, flipped_Z_cd, Z_default, flipped_Z_default, Z_da

def cosmic_dawn_beyond(params_dict, *initial_conditions, Z_eval=None, dm_model='CDM'):
    '''
    Runs IGM solver starting from cosmic dawn until today, i.e., for :math:`z_{\\star} > z`.

    Arguments
    ---------
    params_dict: dict
        Dictionary containing all the parameters.
    
    initial_conditions: tuple
        Initial conditions for the IGM solver.
    
    Z_eval: float
        Array of :math:`1+z` where you want to compute the quantities. Default is ``Z_cd``.
    
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

    if Z_eval is not None:
        xe_cd = CubicSpline(flipped_Z_cd, np.flip(xe_cd))(Z_eval)
        Q_cd = np.interp(Z_eval, flipped_Z_cd, np.flip(Q_cd))
        Tk_cd = CubicSpline(flipped_Z_cd, np.flip(Tk_cd))(Z_eval)
        Ts = myobj_cd.hyfi_spin_temp(Z=Z_eval,xe=xe_cd,Tk=Tk_cd)
        T21 = myobj_cd.hyfi_twentyone_cm(Z=Z_eval,xe=xe_cd,Q=Q_cd,Ts=Ts) 
    else:
        Ts = myobj_cd.hyfi_spin_temp(Z=Z_cd,xe=xe_cd,Tk=Tk_cd)
        T21 = myobj_cd.hyfi_twentyone_cm(Z=Z_cd,xe=xe_cd,Q=Q_cd,Ts=Ts)

    xHI_cd = (1 - Q_cd) * (1 - xe_cd)   # Neutral hydrogen fraction
    
    tau_cd = myobj_cd.reion_tau(Zstar)
    return T21, xHI_cd, tau_cd
#================================================================================

def dark_ages_to_today(params_dict, *initial_conditions, Z_eval=None, dm_model='CDM'):
    '''
    Runs IGM solver starting from :math:`z=1500` to today. Note that arguments ``xe_init`` and ``Tk_init`` are provided only to match the signiture for :py:func:`cosmic_dawn_beyond`. The initial condition is calculated from Saha's equation at :math:`z=1500`. 

    Arguments
    ---------
    params_dict: dict
        Dictionary containing all the parameters.
    
    initial_conditions: tuple
        Initial conditions for the IGM solver. This is not used in this function, but is provided only to match the signiture for :py:func:`cosmic_dawn_beyond`. The initial condition is calculated from Saha's equation at :math:`z=1500`.

    Z_eval: float
        Array of :math:`1+z` where you want to compute the quantities. Default is ``Z_default``.
    
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

    if Z_eval is not None:
        xe = CubicSpline(flipped_Z_default, np.flip(xe))(Z_eval)
        Q_Hii = np.interp(Z_eval, flipped_Z_default, np.flip(Q_Hii))
        Tk = CubicSpline(flipped_Z_default, np.flip(Tk))(Z_eval)

        Ts = myobj.hyfi_spin_temp(Z=Z_eval,xe=xe,Tk=Tk)
        T21 = myobj.hyfi_twentyone_cm(Z=Z_eval,xe=xe,Q=Q_Hii,Ts=Ts)
    else:
        Ts = myobj.hyfi_spin_temp(Z=Z_default,xe=xe,Tk=Tk)
        T21 = myobj.hyfi_twentyone_cm(Z=Z_default,xe=xe,Q=Q_Hii,Ts=Ts)
    
    xHI = (1 - Q_Hii) * (1 - xe)
    tau = myobj.reion_tau(Zstar)

    return T21, xHI, tau