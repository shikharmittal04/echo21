"""
This module contains the functions required for running IGM solver for one set of parameters.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from .echofuncs import funcs
from .misc import smoother
from .const import  Z_start, Z_cd, flipped_Z_cd, Z_default, flipped_Z_default

def cosmic_dawn_beyond(params_dict, xe_init, Tk_init, Z_eval=None):
    '''
    Runs IGM solver starting from cosmic dawn until today, i.e., for :math:`z_{\\star} > z`.

    Arguments
    ---------
    params_dict: dict
        Dictionary containing all the parameters.
    
    xe_init: float
        Initial condition for electron fraction, i.e., :math:`x_{\\mathrm{e}}` at :math:`z = z_{\\star}`.
    
    Tk_init: float
        Initial condition for gas temperature, i.e., :math:`T_{\\mathrm{k}}` at :math:`z = z_{\\star}`.
    
    Z_eval: float
        Array of :math:`1+z` where you want to compute the quantities. Default is ``Z_cd``.
    
    Returns
    -------
    21-cm signal, global-averaged neutral hydrogen fraction, and optical depth.
    '''
    myobj_cd = funcs(params_dict)
    sol_cd = myobj_cd.igm_solver(Z_solver=Z_cd, xe_init=xe_init, Tk_init=Tk_init)
    
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
    
    tau_cd = myobj_cd.reion_tau(50)
    return T21, xHI_cd, tau_cd

#================================================================================

def dark_ages_to_today(params_dict, xe_init, Tk_init, Z_eval=None):
    '''
    Runs IGM solver starting from :math:`z=1500` to today. Note that arguments ``xe_init`` and ``Tk_init`` are provided only to match the signiture for :py:func:`cosmic_dawn_beyond`. The initial condition is calculated from Saha's equation at :math:`z=1500`. 

    Arguments
    ---------
    params_dict: dict
        Dictionary containing all the parameters.
    
    Z_eval: float
        Array of :math:`1+z` where you want to compute the quantities. Default is ``Z_default``.
    
    Returns
    -------
    21-cm signal, global-averaged neutral hydrogen fraction, and optical depth.
    '''
    myobj = funcs(params_dict)
    Tk_init = myobj.basic_cosmo_Tcmb(Z_start)
    xe_init = myobj.recomb_Saha_xe(Z_start,Tk_init)
    
    sol = myobj.igm_solver(Z_default, xe_init, Tk_init)

    xe = sol[0]
    Tk = sol[1]

    Q_Hii = myobj.QHii
    Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

    #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
    Tk[0:1806] = smoother(Z_default[0:1806],Tk[0:1806])

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
    tau = myobj.reion_tau(50)

    return T21, xHI, tau