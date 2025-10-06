from pybaselines import Baseline
import numpy as np
from .echofuncs import funcs
from scipy.interpolate import CubicSpline
from .const import  Z_cd, flipped_Z_cd, Z_default, flipped_Z_default

def print_banner():
    banner = """\n\033[94m
    ███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  ██╗
    ██╔════╝██╔════╝██║  ██║██╔═══██╗╚════██╗███║
    █████╗  ██║     ███████║██║   ██║ █████╔╝╚██║
    ██╔══╝  ██║     ██╔══██║██║   ██║██╔═══╝  ██║
    ███████╗╚██████╗██║  ██║╚██████╔╝███████╗ ██║
    ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═╝
    Copyright 2025, Shikhar Mittal.                                     
    \033[00m\n"""
    print(banner)
    return None

def to_array(params):
        try:
            for keys in params.keys():
                if type(params[keys])==list:
                    params[keys]=np.array(params[keys])
                elif type(params[keys])==float or type(params[keys])==int:
                    params[keys]=np.array([params[keys]])
        except:
            if type(params)==list:
                    params=np.array(params)
            elif type(params)==float or type(params)==int:
                params=np.array([params])        
        return params

def to_float(params):
    try:
        for keys in params.keys():
            if type(params[keys])==list:
                [params[keys]]=params[keys]
            elif type(params[keys])==np.ndarray:
                params[keys]=params[keys][0]
    except:
        if type(params)==list:
                [params]=params
        elif type(params)==np.ndarray:
            params=params[0]
    return params

def smoother(x,y):
    baseline_fitter = Baseline(x_data = x)
    y = baseline_fitter.imodpoly(y, poly_order=4)[0]
    return y

def idm_phy_cd(Ho,Om_m,Om_b,sig8,ns,Tcmbo,Yp,mx_gev,sigma45,fLy,sLy,fX,wX,fesc,Tmin_vir,hmf,mdef, xe_init, Tk_init, Tx_init, v_bx_init, self_Z_eval, Z_temp):
    '''
    IDM, Physically-motivated, cosmic dawn only
    '''
    #try:
    myobj_cd = funcs(Ho=Ho,Om_m=Om_m,Om_b=Om_b,sig8=sig8,ns=ns,Tcmbo=Tcmbo,Yp=Yp,mx_gev=mx_gev,sigma45=sigma45,fLy=fLy,sLy=sLy,fX=fX,wX=wX,fesc=fesc,type='phy',hmf=hmf, mdef=mdef, Tmin_vir=Tmin_vir)
                            
    sol_cd = myobj_cd.igm_solver(Z_eval=Z_cd,xe_init=xe_init,Tk_init=Tk_init,Tx_init=Tx_init,v_bx_init=v_bx_init)
    
    xe_cd = sol_cd[0]
    Tk_cd = sol_cd[1]
    Tx_cd = sol_cd[2]
    vbx_cd= sol_cd[3]

    Q_cd = myobj_cd.QHii

    if self_Z_eval is not None:
        xe_cd = CubicSpline(flipped_Z_cd, np.flip(xe_cd))(self_Z_eval)
        Q_cd = np.interp(self_Z_eval, flipped_Z_cd, np.flip(Q_cd))
        Tk_cd = CubicSpline(flipped_Z_cd, np.flip(Tk_cd))(self_Z_eval)         
        Tx_cd = CubicSpline(flipped_Z_cd, np.flip(Tx_cd))(self_Z_eval)
        vbx_cd = CubicSpline(flipped_Z_cd, np.flip(vbx_cd))(self_Z_eval)
        

    Ts_cd= myobj_cd.hyfi_spin_temp(Z=Z_temp,xe=xe_cd,Tk=Tk_cd)
    return myobj_cd.hyfi_twentyone_cm(Z=Z_temp,xe=xe_cd,Q=Q_cd,Ts=Ts_cd)
    '''
    except:
        print(mx_gev,sigma45,fLy, fX, wX, fesc, Tmin_vir)
        t21 = np.zeros_like(flipped_Z_cd)
        if self_Z_eval is not None:
            t21 = np.zeros_like(self_Z_eval)
        return t21
    '''
def idm_semi_cd(Ho,Om_m,Om_b,sig8,ns,Tcmbo,Yp,mx_gev,sigma45,fLy,sLy,fX,wX,fesc,Tmin_vir,tstar,hmf,mdef, xe_init, Tk_init, Tx_init, v_bx_init, self_Z_eval, Z_temp):
    '''
    IDM, semi-empirical, cosmic dawn only
    '''
    myobj_cd = funcs(Ho=Ho,Om_m=Om_m,Om_b=Om_b,sig8=sig8,ns=ns,Tcmbo=Tcmbo,Yp=Yp,mx_gev=mx_gev,sigma45=sigma45, fLy=fLy,sLy=sLy,fX=fX,wX=wX,fesc=fesc,type='phy',hmf=hmf, mdef=mdef, Tmin_vir=Tmin_vir, tstar= tstar)
                            
    sol_cd = myobj_cd.igm_solver(Z_eval=Z_cd,xe_init=xe_init,Tk_init=Tk_init,Tx_init=Tx_init,v_bx_init=v_bx_init)
    
    xe_cd = sol_cd[0]
    Tk_cd = sol_cd[1]
    Tx_cd = sol_cd[2]
    vbx_cd= sol_cd[3]

    Q_cd = myobj_cd.QHii

    if self_Z_eval is not None:
        xe_cd = CubicSpline(flipped_Z_cd, np.flip(xe_cd))(self_Z_eval)
        Q_cd = np.interp(self_Z_eval, flipped_Z_cd, np.flip(Q_cd))
        Tk_cd = CubicSpline(flipped_Z_cd, np.flip(Tk_cd))(self_Z_eval)         
        Tx_cd = CubicSpline(flipped_Z_cd, np.flip(Tx_cd))(self_Z_eval)
        vbx_cd = CubicSpline(flipped_Z_cd, np.flip(vbx_cd))(self_Z_eval)
        

    Ts_cd= myobj_cd.hyfi_spin_temp(Z=Z_temp,xe=xe_cd,Tk=Tk_cd)
    return myobj_cd.hyfi_twentyone_cm(Z=Z_temp,xe=xe_cd,Q=Q_cd,Ts=Ts_cd)

def cdm_phy_cd(Ho,Om_m,Om_b,sig8,ns,Tcmbo,Yp, fLy,sLy,fX,wX,fesc,Tmin_vir,hmf,mdef, xe_init, Tk_init, self_Z_eval, Z_temp):
    '''
    CDM, physically-motivated, cosmic dawn only
    '''
    myobj_cd = funcs(Ho=Ho,Om_m=Om_m,Om_b=Om_b,sig8=sig8,ns=ns,Tcmbo=Tcmbo,Yp=Yp, fLy=fLy,sLy=sLy,fX=fX,wX=wX,fesc=fesc,type='phy',hmf=hmf, mdef=mdef, Tmin_vir=Tmin_vir)
                            
    sol_cd = myobj_cd.igm_solver(Z_eval=Z_cd,xe_init=xe_init,Tk_init=Tk_init)
    
    xe_cd = sol_cd[0]
    Tk_cd = sol_cd[1]

    Q_cd = myobj_cd.QHii

    if self_Z_eval is not None:
        xe_cd = CubicSpline(flipped_Z_cd, np.flip(xe_cd))(self_Z_eval)
        Q_cd = np.interp(self_Z_eval, flipped_Z_cd, np.flip(Q_cd))
        Tk_cd = CubicSpline(flipped_Z_cd, np.flip(Tk_cd))(self_Z_eval)        

    Ts_cd= myobj_cd.hyfi_spin_temp(Z=Z_temp,xe=xe_cd,Tk=Tk_cd)
    return myobj_cd.hyfi_twentyone_cm(Z=Z_temp,xe=xe_cd,Q=Q_cd,Ts=Ts_cd)

def cdm_emp_cd(Ho,Om_m,Om_b,sig8,ns,Tcmbo,Yp, fLy,sLy,fX,wX,fesc,a_sfrd, xe_init,Tk_init, self_Z_eval,Z_temp):
    '''
    CDM, empirically-motivated, cosmic dawn only
    '''
    myobj_cd = funcs(Ho=Ho,Om_m=Om_m,Om_b=Om_b,sig8=sig8,ns=ns,Tcmbo=Tcmbo,Yp=Yp, fLy=fLy,sLy=sLy,fX=fX,wX=wX,fesc=fesc,type='emp',a_sfrd=a_sfrd)
                            
    sol_cd = myobj_cd.igm_solver(Z_eval=Z_cd,xe_init=xe_init,Tk_init=Tk_init)
    
    xe_cd = sol_cd[0]
    Tk_cd = sol_cd[1]

    Q_cd = myobj_cd.QHii

    if self_Z_eval is not None:
        xe_cd = CubicSpline(flipped_Z_cd, np.flip(xe_cd))(self_Z_eval)
        Q_cd = np.interp(self_Z_eval, flipped_Z_cd, np.flip(Q_cd))
        Tk_cd = CubicSpline(flipped_Z_cd, np.flip(Tk_cd))(self_Z_eval)        

    Ts_cd= myobj_cd.hyfi_spin_temp(Z=Z_temp,xe=xe_cd,Tk=Tk_cd)
    return myobj_cd.hyfi_twentyone_cm(Z=Z_temp,xe=xe_cd,Q=Q_cd,Ts=Ts_cd)

def cdm_semi_cd(Ho,Om_m,Om_b,sig8,ns,Tcmbo,Yp, fLy,sLy,fX,wX,fesc,Tmin_vir,tstar,hmf,mdef, xe_init, Tk_init, self_Z_eval, Z_temp):
    '''
    CDM, Semi-empirical, cosmic dawn only
    '''
    myobj_cd = funcs(Ho=Ho,Om_m=Om_m,Om_b=Om_b,sig8=sig8,ns=ns,Tcmbo=Tcmbo,Yp=Yp, fLy=fLy,sLy=sLy,fX=fX,wX=wX,fesc=fesc,type='phy',hmf=hmf, mdef=mdef, Tmin_vir=Tmin_vir, tstar=tstar)
                            
    sol_cd = myobj_cd.igm_solver(Z_eval=Z_cd,xe_init=xe_init,Tk_init=Tk_init)
    
    xe_cd = sol_cd[0]
    Tk_cd = sol_cd[1]

    Q_cd = myobj_cd.QHii

    if self_Z_eval is not None:
        xe_cd = CubicSpline(flipped_Z_cd, np.flip(xe_cd))(self_Z_eval)
        Q_cd = np.interp(self_Z_eval, flipped_Z_cd, np.flip(Q_cd))
        Tk_cd = CubicSpline(flipped_Z_cd, np.flip(Tk_cd))(self_Z_eval)        

    Ts_cd= myobj_cd.hyfi_spin_temp(Z=Z_temp,xe=xe_cd,Tk=Tk_cd)
    return myobj_cd.hyfi_twentyone_cm(Z=Z_temp,xe=xe_cd,Q=Q_cd,Ts=Ts_cd)

#================================================================================
def idm_phy_full(Ho,Om_m,Om_b,sig8,ns,Tcmbo,Yp,mx_gev,sigma45,fLy,sLy,fX,wX , fesc, Tmin_vir, hmf, mdef, self_Z_eval, Z_temp):
    '''
    IDM, physically-motivated SFRD but full range.
    '''
    #try:
    myobj = funcs(Ho=Ho,Om_m=Om_m,Om_b=Om_b,sig8=sig8,ns=ns,Tcmbo=Tcmbo,Yp=Yp,mx_gev=mx_gev,sigma45=sigma45,fLy=fLy,sLy=sLy,fX=fX,wX =wX, fesc=fesc, type='phy', Tmin_vir=Tmin_vir, mdef = mdef, hmf=hmf)
                                
    sol = myobj.igm_solver(Z_eval=Z_default)

    xe = sol[0]
    Tk = sol[1]
    Tx = sol[2]
    v_bx=sol[3]

    Q_Hii = myobj.QHii
    Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

    #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
    #Tk[0:1806] = smoother(Z_default[0:1806],Tk[0:1806])

    if self_Z_eval is not None:
        splxe = CubicSpline(flipped_Z_default, np.flip(xe))
        xe = splxe(self_Z_eval)
        Q_Hii = np.interp(self_Z_eval, flipped_Z_default, np.flip(Q_Hii))
        splTk = CubicSpline(flipped_Z_default, np.flip(Tk))
        Tk = splTk(self_Z_eval)
        splTx = CubicSpline(flipped_Z_default, np.flip(Tx))
        Tx = splTx(self_Z_eval)
        splvbx = CubicSpline(flipped_Z_default, np.flip(v_bx))
        v_bx = splvbx(self_Z_eval)

    Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)
    return myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)
    #except:
    #    print(mx_gev,sigma45,fLy, fX, wX, fesc, Tmin_vir)
    #    t21 = np.zeros_like(flipped_Z_default)
    #    if self_Z_eval is not None:
    #        t21 = np.zeros_like(self_Z_eval)
    #    return t21

def idm_semi_full(Ho,Om_m,Om_b,sig8,ns,Tcmbo,Yp,mx_gev,sigma45,fLy,sLy,fX,wX , fesc, Tmin_vir, tstar, hmf, mdef, self_Z_eval, Z_temp):
    '''
    IDM, semi-empirical SFRD but full range.
    '''
    myobj = funcs(Ho=Ho,Om_m=Om_m,Om_b=Om_b,sig8=sig8,ns=ns,Tcmbo=Tcmbo,Yp=Yp,mx_gev=mx_gev,sigma45=sigma45,fLy=fLy,sLy=sLy,fX=fX,wX =wX, fesc=fesc, type='semi-emp', Tmin_vir=Tmin_vir, tstar = tstar, mdef = mdef, hmf=hmf)
                                
    sol = myobj.igm_solver(Z_eval=Z_default)

    xe = sol[0]
    Tk = sol[1]
    Tx = sol[2]
    v_bx=sol[3]

    Q_Hii = myobj.QHii
    Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

    #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
    Tk[0:1806] = smoother(Z_default[0:1806],Tk[0:1806])

    if self_Z_eval is not None:
        splxe = CubicSpline(flipped_Z_default, np.flip(xe))
        xe = splxe(self_Z_eval)
        Q_Hii = np.interp(self_Z_eval, flipped_Z_default, np.flip(Q_Hii))
        splTk = CubicSpline(flipped_Z_default, np.flip(Tk))
        Tk = splTk(self_Z_eval)
        splTx = CubicSpline(flipped_Z_default, np.flip(Tx))
        Tx = splTx(self_Z_eval)
        splvbx = CubicSpline(flipped_Z_default, np.flip(v_bx))
        v_bx = splvbx(self_Z_eval)

    Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)
    return myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)

def cdm_phy_full(Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fLy, sLy, fX, wX , fesc, Tmin_vir, hmf, mdef,  self_Z_eval, Z_temp):
    '''
    CDM, physically-motivated but full range
    '''
    myobj = funcs(Ho=Ho,Om_m=Om_m,Om_b=Om_b,sig8=sig8,ns=ns,Tcmbo=Tcmbo,Yp=Yp, fLy=fLy,sLy=sLy,fX=fX,wX = wX, fesc=fesc, type='phy', Tmin_vir=Tmin_vir, mdef = mdef, hmf=hmf)
    sol = myobj.igm_solver(Z_eval=Z_default)

    xe = sol[0]
    Tk = sol[1]

    Q_Hii = myobj.QHii
    Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

    #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
    Tk[0:1806] = smoother(Z_default[0:1806],Tk[0:1806])

    if self_Z_eval is not None:
        xe = CubicSpline(flipped_Z_default, np.flip(xe))(self_Z_eval)
        Q_Hii = np.interp(self_Z_eval, flipped_Z_default, np.flip(Q_Hii))
        Tk = CubicSpline(flipped_Z_default, np.flip(Tk))(self_Z_eval)

    Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)
    return myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)

def cdm_semi_full(Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fLy, sLy, fX, wX , fesc, Tmin_vir, tstar, hmf, mdef, self_Z_eval, Z_temp):
    '''
    CDM, semi-empirical SFRD but full range.
    '''
    myobj = funcs(Ho=Ho,Om_m=Om_m,Om_b=Om_b,sig8=sig8,ns=ns,Tcmbo=Tcmbo,Yp=Yp,fLy=fLy,sLy=sLy,fX=fX,wX =wX, fesc=fesc, type='semi-emp', Tmin_vir=Tmin_vir, tstar = tstar, mdef = mdef, hmf=hmf)
                                
    sol = myobj.igm_solver(Z_eval=Z_default)

    xe = sol[0]
    Tk = sol[1]

    Q_Hii = myobj.QHii
    Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

    #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
    Tk[0:1806] = smoother(Z_default[0:1806],Tk[0:1806])

    if self_Z_eval is not None:
        splxe = CubicSpline(flipped_Z_default, np.flip(xe))
        xe = splxe(self_Z_eval)
        Q_Hii = np.interp(self_Z_eval, flipped_Z_default, np.flip(Q_Hii))
        splTk = CubicSpline(flipped_Z_default, np.flip(Tk))
        Tk = splTk(self_Z_eval)

    Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)
    return myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)

def cdm_emp_full(Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fLy, sLy, fX, wX , fesc, a_sfrd, self_Z_eval, Z_temp):
    '''
    CDM, empirical SFRD but full range.
    '''
    myobj = funcs(Ho=Ho,Om_m=Om_m,Om_b=Om_b,sig8=sig8,ns=ns,Tcmbo=Tcmbo,Yp=Yp,fLy=fLy,sLy=sLy,fX=fX,wX =wX, fesc=fesc, type='emp', a_sfrd=a_sfrd)
                                
    sol = myobj.igm_solver(Z_eval=Z_default)

    xe = sol[0]
    Tk = sol[1]

    Q_Hii = myobj.QHii
    Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

    #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
    Tk[0:1806] = smoother(Z_default[0:1806],Tk[0:1806])

    if self_Z_eval is not None:
        xe = CubicSpline(flipped_Z_default, np.flip(xe))(self_Z_eval)
        Q_Hii = np.interp(self_Z_eval, flipped_Z_default, np.flip(Q_Hii))
        Tk = CubicSpline(flipped_Z_default, np.flip(Tk))(self_Z_eval)

    Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)
    return myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)
