'''
This module evaluates the initial conditions and solves the differential equations given some parameters
Choose 'Radau' method as eq1 is stiff
Also, solve_ivp can only integrate forward in "time", therefore I made a variable change as a=1/Z
'''

from .extras import *
from .const import *
from .basic_cosmo import Tcmb, H, nH, xHe 
from .recomb import *
from .heating import *

import numpy as np
import scipy.integrate as scint 


class dep_var():
	def __init__(self,Zs,xe,Tk,Tx,v_bx):
		self.Zs=Zs
		self.xe=xe
		self.Tk=Tk
		self.Tx=Tx
		self.v_bx=v_bx

#The following function has the differential equations governing the ionisation and thermal history.
def _eqns(Z,V,Ho,Om_m,Om_b,Tcmbo,Yp,falp,fX,fstar,Tmin_vir,fdm,mx_gev,sigma45):
	'''
	When solving upto the end of dark ages, only cosmological parameters will be used.
	Beyond Zstar, i.e., beginning of cosmic dawn astrophysical will also be used.
	'''
	xe = V[0]
	Tk = V[1]
	Tx = V[2]
	v_bx= V[3]
	#eq1 is (1+z)d(xe)/dz; see Weinberg's Cosmology book or eq.(71) from Seager et al (2000), ApJSS
	eq1 = 1/H(Z,Ho,Om_m, Tcmbo)*Peebles_C(Z,xe,Tk, Ho,Om_m,Om_b,Tcmbo,Yp)*(xe**2*nH(Z,Ho,Om_b,Yp)*alpha(Tk)-beta(Tk)*(1-xe)*np.exp(-Ea/(kB*Tk)))
	
	#eq2 is (1+z)dTk/dz; see eq.(2.31) from Mittal et al (2022), JCAP
	
	if Z>Zstar:
		eq2 = 2*Tk-Tk*eq1/(1+xHe(Yp)+xe)-Ecomp(Z,xe,Tk,Ho,Om_m,Tcmbo,Yp)-Ex2b(Z,xe,Tk,Tx,v_bx,Ho,Om_m, Om_b, Tcmbo, Yp, fdm,mx_gev,sigma45)-Ecmb(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp,falp,fstar,Tmin_vir)
	else:
		eq2 = 2*Tk-Tk*eq1/(1+xHe(Yp)+xe)-Ecomp(Z,xe,Tk,Ho,Om_m,Tcmbo,Yp)-Ex(Z,xe,Ho,Om_m,Om_b,Tcmbo,fX,fstar,Tmin_vir)-Elya(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp,falp,fstar,Tmin_vir)-Ex2b(Z,xe,Tk,Tx,v_bx,Ho,Om_m,Om_b, Tcmbo, Yp, fdm,mx_gev,sigma45)-Ecmb(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp,falp,fstar,Tmin_vir)
	
	#eq3 is (1+z)dTx/dz;
	eq3 = 2*Tx-Eb2x(Z,xe,Tk,Tx,v_bx,Ho,Om_m, Om_b, Tcmbo, Yp, fdm,mx_gev,sigma45)
	
	#eq2 is (1+z)dv_bx/dz;
	eq4 = v_bx + D(Z,xe,Tk,Tx,v_bx, Ho,Om_m,Om_b,Yp,fdm,mx_gev,sigma45)/H(Z,Ho,Om_m, Tcmbo)
	
	return np.array([eq1,eq2,eq3,eq4])

def run_solver(Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fX=0.1,fstar=0.1,Tmin_vir=1e4,fdm=1,mx_gev=1,sigma45=1, Z_start=1501,Z_end=6,Z_eval=Z_default, xe_init=None,Tk_init=None,cosmo=None, astro=None):
	
	if cosmo!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
		Yp = cosmo['Yp']
	if astro!=None:
		falp = astro['falp']
		fX = astro['fX']
		fstar = astro['fstar']
		Tmin_vir = astro['Tmin_vir']
		fdm = astro['fdm']
		mx_gev = astro['mx_gev']
		sigma45= astro['sigma45']
	
	if Z_start==1501:
		Tk_init = Tcmb(Z_start,Tcmbo)
		xe_init = Saha_xe(Z_start,Tk_init, Ho,Om_b,Yp)
	elif xe_init==None and Tk_init==None:
		raise Exception('Initial conditions missing.')
		
	Sol = scint.solve_ivp(lambda a, Var: -_eqns(1/a,Var,Ho,Om_m,Om_b,Tcmbo,Yp,falp,fX,fstar,Tmin_vir,fdm,mx_gev,sigma45)/a, t_span=[1/Z_start, 1/Z_end],y0=[xe_init,Tk_init,0,29000],method='Radau',t_eval=1/Z_eval) 
	
	#Obtaining the solutions ...
	
	Zs=Z_eval
	xe=Sol.y[0]
	Tk=Sol.y[1]
	Tx=Sol.y[2]
	v_bx=Sol.y[3]
	
	return dep_var(Zs,xe,Tk,Tx,v_bx)


