'''
Shikhar Mittal
General comments:
This module has all the basic LCDM-cosmology-related functions, such as Hubble function, CMB temperature, etc.

'Z' is 1+z, where z is the redshift.
'Tk' is gas kinetic temperature in kelvin.
'xe' is the electron fraction defined as number density of electron relative to total hydrogen number density.
'''

import numpy as np
from .const import *

def mu(xe,Yp=0.245):
	'''
	Average baryon mass in amu
	'''
	return 4/(4-3*Yp+4*xe*(1-Yp))

def xHe(Yp=0.245):
	'''
	Ratio of He number to H number; n_He/n_H
	'''
	return 0.25*Yp/(1-Yp)

def Tcmb(Z,Tcmbo=2.725):
	'''
	CMB temperature in K
	'''
	return Tcmbo*Z

def rho_crit(Ho=67.4):
	'''
	Critical density of the Universe today; kg.m^-3 
	'''
	return 3*Ho**2/(8*np.pi*GN*Mpc2km**2)

def nH(Z,Ho=67.4,Om_b=0.049,Yp=0.245):
	'''
	Hydrogen number density (proper); m^-3
	'''
	return rho_crit(Ho)*Om_b*(1-Yp)*Z**3/mP

def H(Z,Ho=67.4,Om_m=0.315, Tcmbo=2.725):
	'''
	Hubble factor in SI units; sec^-1
	'''
	Om_lam = 1-Om_m
	Om_r = (1+fnu)*aS*Tcmbo**4/(cE**2*rho_crit(Ho))
	return Ho*(Om_r*Z**4+Om_m*Z**3+Om_lam)**0.5/Mpc2km
