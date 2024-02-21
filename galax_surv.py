#Luminosity function and the number of galaxies

from .const import *
from .basic_cosmo import H
from .hmf import dndlnM, dndM
import numpy as np
import scipy.integrate as scint 
from colossus.cosmology import cosmology

def luminosity(Mh,Z,fstar = 0.1, astro=None):
	'''
	For given halo mass (in solar mass) and 1+redshift, what is the luminosity in W/Hz
	'''
	if astro!=None:
		fstar = astro['fstar']
	
    return fstar*Mdot0*(Mh/1e10)**a*(Z/7)**b*l

def MAB(Mh,Z,fstar=0.1, astro=None):
	'''
	Absolute AB magnitude, Mh in solar mass units.
	'''
	if astro!=None:
		fstar = astro['fstar']
    return -2.5*np.log10(luminosity(Mh,Z,fstar)/(4*np.pi*(10*3.086e16)**2))-56.1

def muv2Mh(muv,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Tmin_vir=1e4, fstar=0.1, cosmo=None,astro=None):
	'''
	muv is apparent magnitude; output is halo mass in solar mass units 
	'''
	if cosmo!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
		
	if astro!=None:
		fstar = astro['fstar']
		Tmin_vir = astro['Tmin_vir']
		
	my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
	colo_cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
	
    DL = colo_cosmo.luminosityDistance(Z-1)/h100*3.086e22 #luminosity distance in meters.
    Lum = 4*np.pi*DL**2*10**(-0.4*(muv+56.1)) #Halo luminosity in W/Hz
    return 1e10*(Lum/(fstar*Mdot0*(Z/7)**b*l))**(1/a)


#For given halo mass (in solar mass) and redshift, get LF. It is in same units as HMF.
#Also, note that this model is valid when SFE is a constant.
def lum_func(Mh,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Tmin_vir=1e4,cosmo=None,astro=None):
	if cosmo!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
	
	if astro!=None:
		Tmin_vir = astro['Tmin_vir']
    return 2*np.log(10)/5/a*dndlnM(Mh,Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir)

#Given a limiting apparent magnitude and survey area (in deg), what is the number of galaxies seen at z
def num_gal(muv_lim,area,Z,fstar=0.1):
    def Ngal(muv_lim,area,Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir,fstar):
    	my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
		colo_cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
		
        Mh_lim = muv2Mh(muv_lim,Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir,fstar)
        halo_masses=np.logspace(np.log10(Mh_lim),18,2000)
        integral=scint.trapz(dndM(halo_masses,z),halo_masses)    #number per unit cMpc^3
        DL = colo_cosmo.luminosityDistance(Z-1)/h100 #luminosity distance in Mpc
        return 1/(Mpc2km*1e3)*cE/H(Z)*(DL/Z)**2*(np.pi/180)**2*integral*area
    
    if type(Z)==np.ndarray:
        leng = len(Z)
        N=np.zeros(leng)
        count=0
        for i in Z:
            N[count]=Ngal(muv_lim,area,i,fstar)
            count=count+1
    elif type(Z)==list:
        leng = len(Z)
        N=np.zeros(leng)
        count=0
        for i in Z:
            N[count]=Ngal(muv_lim,area,i,fstar)
            count=count+1
    else:
        N=Ngal(muv_lim,area,Z,fstar)
        print('For survey area =',area,'deg and limiting magnitude =',muv_lim,'there are',round(N),'galaxies at z =',Z-1)
    return N
