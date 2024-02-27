'''
Shikhar Mittal
General comments:
This module has all the 21-cm-related functions.

'Z' is 1+z, where z is the redshift.
'Tk' is gas kinetic temperature in kelvin.
'xe' is the electron fraction defined as number density of electron relative to total hydrogen number density.
'''

from .const import *
from .hmf import SFRD
from .basic_cosmo import Tcmb, nH, H
from .extras import lya_spec_inten, scatter_corr
import scipy.integrate as scint
import numpy as np

def kHH(Tk):
	'''
	Volumetric spin flip rate for H-H collision; m^3/s
	This fitting function and the next one is available from Pritchard & Loeb (2012)
	'''
	return 3.1*10**(-17)*Tk**0.357*np.exp(-32/Tk)

def keH(Tk):
	'''
	Volumetric spin flip rate for e-H collision; m^3/s
	'''
	return np.where(Tk<10**4,10**(-15.607+0.5*np.log10(Tk)*np.exp(-(np.abs(np.log10(Tk)))**4.5/1800) ),10**(-14.102))

def col_coup(Z,xe,Tk, Ho=67.4,Om_b=0.049,Tcmbo=2.725,Yp=0.245,cosmo=None):
	'''
	Collisional coupling, x_c; dimensionless
	'''
	if cosmo!=None:
		Ho = cosmo['Ho']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
		Yp = cosmo['Yp']

	return Tstar*nH(Z,Ho,Om_b,Yp)*((1-xe)*kHH(Tk)+xe*keH(Tk))/(A10*Tcmb(Z,Tcmbo))

def lya_coup(Z,xe,Tk, Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
	'''
	Ly-alpha coupling or the Wouthuysen-Field coupling, x_alpha; dimensionless.
	'''
	if cosmo!=None and astro!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
		Yp = cosmo['Yp']
		
		falp = astro['falp']
		fstar = astro['fstar']
		Tmin_vir = astro['Tmin_vir']
	
	S = scatter_corr(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)
	J = lya_spec_inten(Z,xe,Ho,Om_m,Om_b,Tcmbo,falp,fstar,Tmin_vir)	#'undistorted' background Spec. Inte. of Lya photons.
	Jo = 5.54e-8*Z 		#eq.(24) in Mittal & Kulkarni (2021)
	return S*J/Jo

def spin_temp(Z,xe,Tk, Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
	'''
	Spin temperature in K.
	'''
	if cosmo!=None and astro!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
		Yp = cosmo['Yp']
		
		falp = astro['falp']
		fstar = astro['fstar']
		Tmin_vir = astro['Tmin_vir']

	xa = lya_coup(Z,xe,Tk, Ho,Om_m,Om_b,Tcmbo,Yp, falp,fstar,Tmin_vir)
	xk = col_coup(Z,xe,Tk, Ho,Om_b,Tcmbo,Yp)
	Ts = ( 1  + xa + xk)/(1/Tcmb(Z, Tcmbo) +  (xk+xa)/Tk )	#We assume the colour temperature is same as Tk.
	return Ts

def Terb(Z,Tcmbo,zeta_erb): #Net background temperature (includes CMB)
	return np.where(Z<Zstar,Tcmbo*Z*(1+0.169*zeta_erb*Z**2.6),Tcmb(Z,Tcmbo))

def twentyone_cm(Z,xe,Tk, Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
	'''
	The global (sky-averaged) 21-cm signal in mK.
	'''
	if cosmo!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
		Yp = cosmo['Yp']
	if astro!=None:
		falp = astro['falp']
		fstar = astro['fstar']
		Tmin_vir = astro['Tmin_vir']

	xHI=(1-xe)
	h100 = Ho/100
	Ts = spin_temp(Z,xe,Tk, Ho, Om_m, Om_b, Tcmbo, Yp,falp,fstar,Tmin_vir)
	return 27*xHI*((1-Yp)/0.76)*(Om_b*h100**2/0.023)*np.sqrt(0.15*Z/(10*Om_m*h100**2))*(1-Tcmb(Z)/Ts)

