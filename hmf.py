'''
Shikhar Mittal
This module contains halo mass function and derived quantities. Most importantly, this module is used to get the star formation rate density.

'Z' is 1+z, where z is the redshift.
'Tk' is gas kinetic temperature in kelvin.
'''

from .basic_cosmo import mu, rho_crit, H
import scipy.special as scsp
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.lss import mass_function
import numpy as np


def sfr_model(hmf_name='press74',sfe_name='const'):
	global hmf_model
	hmf_model=hmf_name
	global sfe_model
	sfe_model=sfe_name
	return
	
def dndlnM(M,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Tmin_vir=1e4,cosmo=None):
	'''
	M is the halo mass in units of solar mass
	Output is dn/d(ln M)=M.dn/dM in units of cMpc**-3
	This function is usually what people refer to, when they write 'HMF'
	'''
	if cosmo!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
		Tmin_vir = astro['Tmin_vir']
		
	my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
	cosmology.setCosmology('my_cosmo', my_cosmo)
	h100=Ho/100
	M_by_h = M*h100 #M in units of solar mass/h
	return h100**3*mass_function.massFunction(M_by_h, Z-1, q_in='M', q_out='dndlnM', model = hmf_model)

def dndM(M,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Tmin_vir=1e4,cosmo=None):
	'''
	M in the halo mass in units of solar mass
	Output is dn/dM in units of cMpc**-3.Msun**-1
	'''
	if cosmo!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
		Tmin_vir = astro['Tmin_vir']
	
	my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
	cosmology.setCosmology('my_cosmo', my_cosmo)
	return 1/M*dndlnM(M,Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir)

#For details see eq.(50),(52) and (53) from Mittal & Kulkarni (2021), MNRAS
def m_min(Z,Om_m=0.315,Tmin_vir=1e4):
	'''
	This function gives the minimum halo mass for star formation
	The mass returned is in units of solar mass/h
	Optional parameters are relative matter density & miminum virial temperature (Tmin_vir in K)
	'''
	return 1e8*Om_m**(-0.5)*(10/Z*0.6/1.22*Tmin_vir/1.98e4)**1.5

def f_coll(Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Tmin_vir=1e4,cosmo=None):
	'''
	Collapse fraction: fraction of baryons that collapsed into the DM haloes
	IMPORTANT: This function is only applicable for Press-Schechter halo mass function
	'''
	if cosmo!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
		Tmin_vir = astro['Tmin_vir']
	
	my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
	cosmology.setCosmology('my_cosmo', my_cosmo)
	
	return scsp.erfc(peaks.peakHeight(m_min(Z,Om_m,Tmin_vir),Z-1)/np.sqrt(2))

def dfcoll_dz(Z,Ho,Om_m,Om_b,Tcmbo, Tmin_vir):
	'''
	Derivative of the collapse fraction
	'''
	return (f_coll(Z+1e-3, Ho, Om_m,Om_b,Tcmbo, Tmin_vir)-f_coll(Z, Ho, Om_m,Om_b,Tcmbo, Tmin_vir))*1e3

 
def SFRD(Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
	'''
	This function returns the comoving star formation rate density in units of kg/s/m^3
	'''
	if cosmo!=None and astro!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
		fstar = astro['fstar']
		Tmin_vir = astro['Tmin_vir']
	
	return -Z*fstar*Om_b*rho_crit(Ho)*dfcoll_dz(Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir)*H(Z, Ho,Om_m, Tcmbo)
