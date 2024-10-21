import scipy.special as scsp
import scipy.integrate as scint
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys
import time
import os
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.lss import mass_function
from time import localtime, strftime

#========================================================================================================
#Universal constants
GN=6.67e-11 #Gravitational constant
cE=2.998e8  #Speed of light
kB=1.38e-23 #Boltzmann constant
hP=6.634e-34 #Planck's contant
mP=1.67e-27 #Mass of proton
me=9.1e-31 #Mass of electron
eC=1.6e-19 #Charge of electron
epsilon=8.85e-12 #Permittivity of free space

aS=7.52e-16 #Stephan's radiation constant
sigT=6.65e-29 #Thomson scattering cross-section, m^2

fnu = 0.68 #neutrino contribution to energy density in relativistic species; 3 massless nu's
#-------------------------------------------------------------
#Conversions
 
Mpc2km = 3.0857e19
Msolar = 1.989e30 #Mass of sun in kg
Msolar_by_Mpc3_to_kg_by_m3 = Msolar*(1000*Mpc2km)**-3


#-------------------------------------------------------------
#Hardcoded but later we want to change some of these

fstar = 0.1
Nion = 1000
alpha_A = 4.2e-19 #Case-B recombination coefficient (m^3/s) at T=2 X 10^4 K (Osterbrock & Ferland 2006)

Zstar = 60 #redshift of the beginning of star formation

Z_start = 1501
Z_end = 6
Ngrid = 1600
Z_default = np.linspace(Z_start,Z_end,Ngrid)

#-------------------------------------------------------------
#Recombination related
Lam_H = 8.22458 #The H 2s–1s two photon rate in s^−1
A,b,c,d = 4.309, -0.6166, 0.6703, 0.53
Feff = 1.14 #This extra factor gives the effective 3-level recombination model
lam_alpha = 121.5682e-9 #Wavelength of Lya photon in m
nu_alpha = cE/lam_alpha #Frequency in Hz
B2 = 3.4*eC #Bind energy of level 2 in J
B1 = 13.6*eC #Bind energy of level 1 in J
Ea = B1-B2  #Energy of Lya photon in J
A_alpha = 6.25e8 #Spontaneous emission coeffecient in Hz
#-------------------------------------------------------------
#Others
Tstar = 0.068 #Hyperfine energy difference in temperature (K)
A10 = 2.85e-15 # Einstein's spontaneous emission rate, sec^-1
Pn=np.array([0.2609,0.3078,0.3259,0.3353,0.3410,0.3448,0.3476,0.3496,0.3512,0.3524,0.3535,0.3543,0.355,0.3556,
    0.3561,0.3565,0.3569,0.3572,0.3575,0.3578])
            
#========================================================================================================
            
def print_banner():
    banner = """\n\033[94m
    ███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  ██╗
    ██╔════╝██╔════╝██║  ██║██╔═══██╗╚════██╗███║
    █████╗  ██║     ███████║██║   ██║ █████╔╝╚██║
    ██╔══╝  ██║     ██╔══██║██║   ██║██╔═══╝  ██║
    ███████╗╚██████╗██║  ██║╚██████╔╝███████╗ ██║
    ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═╝                                         
    \033[00m\n"""
    print(banner)
    return None

class basic_cosmo():
    '''
    This module has all the basic :math:`\\Lambda`CDM-cosmology-related functions, such as Hubble function, CMB temperature, etc.

    Methods
    ~~~~~~~
    '''
    def __init__(self):
        return None

    def mu(self,xe,Yp=0.245):
        '''
        Average baryon mass in amu

        Arguments
        ---------

        xe : float
            Electron fraction (dimensionless)

        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``
                    
                    
        Returns
        -------

        float
            Average baryon mass in amu
        
        '''
        return 4/(4-3*Yp+4*xe*(1-Yp))

    def xHe(self,Yp=0.245):
        '''
        Ratio of helium number density to hydrogen number density

        Arguments
        ---------

        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``

        Returns
        -------

        float
            :math:`n_{\\mathrm{He}}/n_{\\mathrm{H}}`
        
        '''
        return 0.25*Yp/(1-Yp)

    def Tcmb(self,Z,Tcmbo=2.725):
        '''
        CMB temperature at a given redshift

        Arguments
        ---------

        Z : float
            1+z

        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``

        Returns
        -------

        float
            CMB temperature at the given redshift in kelvin
        '''
        return Tcmbo*Z

    def rho_crit(self,Ho=67.4):
        '''
        Critical density of the Universe today

        Arguments
        ---------

        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``
      
        Returns
        -------
        
        float
            Critical density today, :math:`\\rho_{\mathrm{crit}}=\\frac{3H_0^2}{8\\pi G_{\\mathrm{N}}}` in units of :math:`\\mathrm{kg}\\,\\mathrm{m}^{-3}`
        '''
        return 3*Ho**2/(8*np.pi*GN*Mpc2km**2)

    def nH(self,Z,Ho=67.4,Om_b=0.049,Yp=0.245):
        '''
        Hydrogen number density (proper)
        
        Arguments
        ---------
        
        Z : float
            1+z
            
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``
            
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``
        
        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``
        
        Returns
        -------
        
        float
            Proper hydrogen number density at given redshift in units of :math:`\\mathrm{m}^{-3}`
        '''
        return self.rho_crit(Ho)*Om_b*(1-Yp)*Z**3/mP

    def H(self,Z,Ho=67.4,Om_m=0.315, Tcmbo=2.725):
        '''
        Hubble factor in SI units
        
        Arguments
        ---------
        
        Z : float
            1+z
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``
            
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``
        
        Returns
        -------
        
        float
            Hubble parameter at a given redshift in units of :math:`\\mathrm{s}^{-1}`
        '''
        Om_lam = 1-Om_m
        Om_r = (1+fnu)*aS*Tcmbo**4/(cE**2*self.rho_crit(Ho))
        
        return Ho*(Om_r*Z**4+Om_m*Z**3+Om_lam)**0.5/Mpc2km

#End of class basic_cosmo
#========================================================================================================
     
class recomb():
    '''
    This has eveything related to recombination physics.
    
    Methods
    ~~~~~~~
    '''
    
    def __init__(self):
        return None
        
    def alpha(self, Tk):
        '''
        :math:`\\alpha_{\\mathrm{B}}=\\alpha_{\\mathrm{B}}(T)`
        
        The effective case-B recombination coefficient for hydrogen. See Eq. (70) from `Seager et al (2000) <https://iopscience.iop.org/article/10.1086/313388>`__.
        
        Arguments
        ---------
        
        Tk : float
            Gas temperature in units of kelvin.
        
        Returns
        -------
        
        float
            The effective case-B recombination coefficient for hydrogen :math:`(\\mathrm{m}^3\\mathrm{s}^{-1})`.
            
        '''
        t=Tk/10000
        return (1e-19)*Feff*A*t**b/(1+c*t**d)

    def beta(self, Tk):
        '''
        :math:`\\beta=\\beta(T)`
        
        The total photoionisation rate. See description below Eq. (71) from `Seager et al (2000) <https://iopscience.iop.org/article/10.1086/313388>`__
        Relation between :math:`\\alpha_{\\mathrm{B}}` and :math:`\\beta`:
        
        :math:`\\beta=\\alpha_{\\mathrm{B}}\\left(\\frac{2\\pi m_{\\mathrm{e}}k_{\\mathrm{B}}T}{h_{\\mathrm{P}}^2}\\right)^{3/2}\\exp\\left(-\\frac{B_2}{k_{\\mathrm{B}}T}\\right)`
        
        Arguments
        ---------
        
        Tk : float
            Gas temperature in units of kelvin.
        
        Returns
        -------
        
        float
            The total photoionisation rate in :math:`(\\mathrm{s}^{-1})`.
            
        '''
        return self.alpha(Tk)*(2*np.pi*me*kB*Tk/hP**2)**1.5*np.exp(-B2/(kB*Tk))

    def Krr(self, Z,Ho=67.4,Om_m=0.315, Tcmbo=2.725):
        '''
        Redshifting rate appearing in the Peebles' 'C' factor
        
        Arguments
        ---------
        
        Z : float
            1+z
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``
            
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``
        
        Returns
        -------
        
        float
            Redshifting rate in units of :math:`\\mathrm{m^3s}`
        
        '''
        return lam_alpha**3/(8*np.pi*basic_cosmo().H(Z,Ho,Om_m, Tcmbo))

    def Peebles_C(self,Z,xe,Tk, Ho=67.4,Om_m=0.315, Om_b=0.049, Tcmbo=2.725, Yp=0.245):
        '''
        :math:`C_{\mathrm{P}}`
        
        Arguments
        ---------
        
        Z : float
            1 + redshift, dimensionless
        
        xe : float
            Electron fraction, dimensionless
            
        Tk : float
            Gas temperature in units of kelvin
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``
            
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``
        
        Yp : float, optional
            Primordial helium fraction by mass. Default value 0.245
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value 2.725
        
        Returns
        -------
        
        float
            Peebles 'C' factor appearing in Eq. (71) from `Seager et al (2000) <https://iopscience.iop.org/article/10.1086/313388>`__, dimensionless.
        '''
        
        return (1+self.Krr(Z,Ho,Om_m, Tcmbo)*Lam_H*basic_cosmo().nH(Z, Ho,Om_b,Yp)*(1-xe))/(1+self.Krr(Z, Ho,Om_m, Tcmbo)*(Lam_H+self.beta(Tk))*basic_cosmo().nH(Z, Ho,Om_b,Yp)*(1-xe))

    def Saha_xe(self,Z,Tk, Ho=67.4,Om_b=0.049,Yp=0.245):
        '''
        Electron fraction predicted by the Saha's equation. This is important to initialise the differential equation for :math:`x_{\mathrm{e}}`. At high redshift such as :math:`z\\approx1500`, Saha's equation give accurate estimate of :math:`x_{\mathrm{e}}`.
        
        Arguments
        ---------
        
        Z : float
            1 + redshift, dimensionless
          
        Tk : float
            Gas temperature in units of kelvin
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``
            
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``
        
        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``
        
        Returns
        -------
        
        float
            Electron fraction predicted by Saha's equation.
        '''
        S=1/basic_cosmo().nH(Z, Ho,Om_b,Yp)*(2*np.pi*me*kB*Tk/hP**2)**1.5*np.exp(-B1/(kB*Tk))
        return (np.sqrt(S**2+4*S)-S)/2

#End of class recomb
#========================================================================================================

class hmf():
    '''
    This class contains :math:`\\mathrm{d}n/\\mathrm{d\\,ln}M`, :math:`\\mathrm{d}n/\\mathrm{d}M`, :math:`m_{\\mathrm{min}}`, :math:`f_{\\mathrm{coll}}`, :math:`\\mathrm{d}f_{\\mathrm{coll}}/\\mathrm{d}z`, and :math:`\\dot{\\rho}_{\\star}`.
    
    Use this function to set your choice of HMF model name and also the choice of star formation efficiency (SFE) model name. Eg. ``hmf.hmf_name = 'press74'`` and ``hmf.sfe_name = 'const'``.
    
    Available HMF model names:
        - press74 (default, for Press & Schechter 1974),
        - sheth99 (for Sheth & Tormen 1999),
        -
        
    Available SFE models:
        - const (default, for a constant SFE throughout the redshift)
        - log
        - dpl
        
    Methods
    ~~~~~~~
    '''
    def __init__(self, hmf_name='press74',sfe_name='const'):
        global hmf_model
        self.hmf_name=hmf_name
        hmf_model=self.hmf_name
        global sfe_model
        self.sfe_name=sfe_name
        sfe_model=self.sfe_name
        return
        
    def dndlnM(self, M,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,cosmo=None):
        '''
        The halo mass function (HMF) in the form of :math:`\\mathrm{d}n/\\mathrm{d\\,ln}M`. Note the natural logarithm.
        
        Arguments
        ---------
        
        M : float
            The desired halo mass at which you want to evaluate HMF. Input M in units of solar mass.
        
        Z : float
            1 + redshift, dimensionless.
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
            
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``.
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Note: cosmological and astrophysical parameters can also be supplied through dictionaries ``cosmo`` and ``astro``.
        
        Returns
        -------
        
        float
            HMF, :math:`\\mathrm{d}n/\\mathrm{d\\,ln}M=M\\mathrm{d}n/\\mathrm{d}M`, in units of :math:`\\mathrm{cMpc}^{-3}`, where 'cMpc' represents comoving Mega parsec.
        '''
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
            
        my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
        cosmology.setCosmology('my_cosmo', my_cosmo)
        h100=Ho/100
        M_by_h = M*h100 #M in units of solar mass/h
        return h100**3*mass_function.massFunction(M_by_h, Z-1, q_in='M', q_out='dndlnM', model = hmf_model)

    def dndM(self,M,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,cosmo=None):
        '''
        The halo mass function (HMF) in a different form, i.e., :math:`\\mathrm{d}n/\\mathrm{d}M`.
        
        Arguments
        ---------
        
        M : float
            The desired halo mass at which you want to evaluate HMF. Input M in units of solar mass (:math:`\mathrm{M}_{\\odot}`).
        
        Z : float
            1 + z, dimensionless.
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
            
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``.
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Note: cosmological and astrophysical parameters can also be supplied through dictionaries ``cosmo`` and ``astro``.
        
        Returns
        -------
        
        float
            HMF in a different form, :math:`\\mathrm{d}n/\\mathrm{d}M`, in units of :math:`\\mathrm{cMpc}^{-3}\mathrm{M}_{\\odot}^{-1}`, where 'cMpc' represents comoving Mega parsec and :math:`\mathrm{M}_{\\odot}` represents the solar mass.
        '''

        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
        
        my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
        cosmology.setCosmology('my_cosmo', my_cosmo)
        return 1/M*self.dndlnM(M,Z,Ho,Om_m,Om_b,Tcmbo)

    #For details see eq.(50),(52) and (53) from Mittal & Kulkarni (2021), MNRAS
    def m_min(self,Z,Om_m=0.315,Tmin_vir=1e4):
        '''
        The minimum halo mass for which star formation is possible.
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless. It can be a single number or an array.
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
        
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.
        
        Returns
        -------
        
        float
            The mass returned is in units of :math:`\mathrm{M}_{\\odot}/h`.
        '''
        return 1e8*Om_m**(-0.5)*(10/Z*0.6/1.22*Tmin_vir/1.98e4)**1.5

    def f_coll(self,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Tmin_vir=1e4,cosmo=None,astro=None):
        '''
        Collapse fraction -- fraction of total matter that collapsed into the haloes. See definition below.
        :math:`F_{\\mathrm{coll}}=\\frac{1}{\\bar{\\rho}^0_{\\mathrm{m}}}\\int_{M_{\\mathrm{min}}}^{\\infty} M\\frac{\\mathrm{d}n}{\\mathrm{d} M}\,\\mathrm{d} M\,,`
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless. Can be a single quantity or an array.
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
            
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``.
        
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Note: cosmological and astrophysical parameters can also be supplied through dictionaries ``cosmo`` and ``astro``.
        
        Returns
        -------
        
        float 
            Collapse fraction. Single number or an array accordingly as ``Z`` is single number or an array.
        '''
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
        if astro!=None:
            Tmin_vir = astro['Tmin_vir']

        if hmf_model=='press74':
            my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
            cosmology.setCosmology('my_cosmo', my_cosmo)
            return scsp.erfc(peaks.peakHeight(self.m_min(Z,Om_m,Tmin_vir),Z-1)/np.sqrt(2))
        else:
            h100=Ho/100
            numofZ = np.size(Z)
                
            if numofZ == 1:
                if type(Z)==np.ndarray: Z=Z[0]
                M_space = np.logspace(np.log10(self.m_min(Z,Om_m,Tmin_vir)/h100),18,1500)    #These masses are in solar mass. 
                hmf_space = self.dndlnM(M=M_space,Z=Z,Ho=Ho,Om_m=Om_m,Om_b=Om_b,Tcmbo=Tcmbo)    #Corresponding HMF values are in cMpc^-3 
                rho_halo = Msolar_by_Mpc3_to_kg_by_m3*np.trapz(hmf_space,M_space)    #matter density collapsed as haloes (in kg/m^3, comoving)
            else:    
                rho_halo = np.zeros(numofZ)
                counter=0
                for i in Z:
                    M_space = np.logspace(np.log10(self.m_min(i,Om_m,Tmin_vir)/h100),18,1500)    #These masses are in solar mass. 
                    hmf_space = self.dndlnM(M=M_space,Z=i,Ho=Ho,Om_m=Om_m,Om_b=Om_b,Tcmbo=Tcmbo)    #Corresponding HMF values are in cMpc^-3 
                    rho_halo[counter] = Msolar_by_Mpc3_to_kg_by_m3*np.trapz(hmf_space,M_space)    #matter density collapsed as haloes (in kg/m^3, comoving)
                    counter=counter+1
            return rho_halo/(Om_m*basic_cosmo().rho_crit(Ho))

    def dfcoll_dz(self,Z,Ho,Om_m,Om_b,Tcmbo, Tmin_vir):
        '''
        Redshift derivative of the collapse fraction, i.e., :math:`\\mathrm{d}F_{\\mathrm{coll}}/\\mathrm{d}z`
        '''
        return (self.f_coll(Z+1e-3, Ho, Om_m,Om_b,Tcmbo, Tmin_vir)-self.f_coll(Z, Ho, Om_m,Om_b,Tcmbo, Tmin_vir))*1e3

     
    def SFRD(self,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Tmin_vir=1e4,cosmo=None,astro=None):
        '''
        This function returns the comoving star formation rate density (SFRD, :math:`\\dot{\\rho}_{\\star}`).
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless. Can be a single quantity or an array.
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
            
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``.
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.
        
        Note: cosmological and astrophysical parameters can also be supplied through dictionaries ``cosmo`` and ``astro``.
        
        Returns
        -------
        
        float 
            Comoving SFRD in units of :math:`\\mathrm{kgs^{-1}m^{-3}}`. Single number or an array accordingly as ``Z`` is single number or an array.
        '''
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
        if astro!=None:
            Tmin_vir = astro['Tmin_vir']

        return -Z*fstar*Om_b*basic_cosmo().rho_crit(Ho)*self.dfcoll_dz(Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir)*basic_cosmo().H(Z, Ho,Om_m, Tcmbo)

#End of class hmf
#========================================================================================================

class heating():
    '''
    
    This class contains all the heating (or cooling) terms. All the terms are in the form of :math:`-(1+z)\\mathrm{d}T_{\\mathrm{k}}/\\mathrm{d}z` and hence in units of temperature.
    
    Methods
    ~~~~~~~
    '''
    def __init__(self):
        return None
    
    def Ecomp(self,Z,xe,Tk,Ho=67.4,Om_m=0.315,Tcmbo=2.725,Yp=0.245,cosmo=None):
        '''
        See Eq.(2.32) from Mittal et al (2022), JCAP.
        (However, there is a minor typo in that equation; numerator has an :math:`x_{\\mathrm{e}}` missing.)
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature.
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``
        
        Returns
        -------    
        
        float
            Compton heating. Units kelvin.
            
        Note: cosmological parameters can also be supplied through dictionary ``cosmo``
        '''
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Tcmbo = cosmo['Tcmbo']
            Yp = cosmo['Yp']
        
        return (8*sigT*aS)/(3*me*cE)*basic_cosmo().Tcmb(Z,Tcmbo)**4*xe*(basic_cosmo().Tcmb(Z,Tcmbo)-Tk)/(basic_cosmo().H(Z,Ho,Om_m,Tcmbo)*(1+basic_cosmo().xHe(Yp)+xe))


    def Elya(self,Z,xe,Tk,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,Tmin_vir=1e4,cosmo=None,astro=None):
        '''
        Ly-:math:`\\alpha` heating rate. For details see `Mittal & Kulkarni (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.4264M/abstract>`__
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature.
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
        
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``.            
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``.
        
        falp : float, optional
            :math:`f_{\\alpha}`, a dimensionless parameter which controls the emissivity of the Lyman series photons. Default value 1.
        
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.
        
        Returns
        -------    
        
        float
            Net heating by the Lyman series photons. Units kelvin.
            
        Note: cosmological ans astrophysical parameters can also be supplied through dictionaries ``cosmo`` and ``astro``, respectively.
        '''
        
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
        if astro!=None:
            falp = astro['falp']
            Tmin_vir = astro['Tmin_vir']
            
        eta = extras().recoil(Tk)
        S = extras().scatter_corr(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)
        atau = extras().a_tau(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)
        arr = scsp.airy(-extras().xi2(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp))
        
        Ic = eta*(2*np.pi**4*atau**2)**(1/3)*(arr[0]**2+arr[2]**2)
        Ii = eta*np.sqrt(atau/2)*scint.quad(lambda y:y**(-1/2)*np.exp(-2*eta*y-np.pi*y**3/(6*atau))*scsp.erfc(np.sqrt(np.pi*y**3/(2*atau))),0,np.inf)[0]-S*(1-S)/(2*eta)
        J = extras().lya_spec_inten(Z,Ho,Om_m,Om_b,Tcmbo,falp,Tmin_vir)
        nbary = (1+basic_cosmo().xHe(Yp))*basic_cosmo().nH(Z,Ho,Om_b,Yp)
        return 8*np.pi/3 * hP/(kB*lam_alpha) * J*extras().dopp(Tk)/nbary * (Ic+Ii)
       
    '''
    def tau(E,Z,Z1,x_HI):     #X-ray optical depth
            Z2=np.linspace(Z,Z1[1:,],20)
            taux=(Z/E)**3*7341856114*x_HI**(1/3)*scint.trapz(1/(Z2*H(Z2)) ,Z2,axis=0)
            return np.insert(taux,0,0)

    def Jx(E,Z,x_HI):        #Number of X-ray photons/(area-time-solid angle-energy(in eV))
            Steps=int(5*(31-Z))
            Z1=np.linspace(Z,31,Steps+1)
            return 944580047*Z**2*scint.trapz(epsilon_x(E*Z1/Z,Z1)*np.exp(-tau(E,Z,Z1,x_HI))/H(Z1),Z1)

    def epsilon_x(E,Z):
            return 10**log_fx*(w-1)/Eo*(E/Eo)**(-1-w)*SFRD(Z)/(1-(Eo/3e4)**(w-1))

    def sig(E):           #Phototionisation cross section of hydrogen, but took the constant factor into Γx and Ex
            X=E/0.4298
            return (X-1)**2*X**-4.0185/(1+np.sqrt(X/32.88))**2.963

    def Gam_and_Ex(Z,x_e):      #Photoheating
            mu=4/(4-3*Yp+4*x_e*(1-Yp))
            x_HI=1-x_e
            if x_e<1.0:
                    f_heat=1-(1-x_e**0.2663)**1.3163
            else:
                    f_heat=1

            f_ion=0.3908*(1-x_e**0.4092)**1.7592
            Γx=4*np.pi*scint.quad(lambda E: sig(E)*Jx(E,Z,x_HI),Eo,30000)[0]
            Hx=4*np.pi*scint.quad(lambda E:(E-Ei)*sig(E)*Jx(E,Z,x_HI),Eo,30000)[0]
            Gam_x=(Γx+f_ion*Hx/Ei)*168.94/mu
            Ex=f_heat*1306992.8*(1-Yp)/H(Z)*(1-x_e)*Hx
            return np.array([Gam_x,Ex])
    '''

    def Ex(self,Z,xe,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,fX=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
        '''
        See Eq.(11) from `Furlanetto (2006) <https://academic.oup.com/mnras/article/371/2/867/1033021>`__
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
        
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``.            
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        fX : float, optional
            :math:`f_{\\mathrm{X}}`, a dimensionless parameter which controls the emissivity of the X-ray photons. Default value 0.1.
        
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.
        
        Returns
        -------    
        
        float
            Net heating by the X-ray photons. Units kelvin.
            
        Note: cosmological and astrophysical parameters can also be supplied through dictionaries ``cosmo`` and ``astro``, respectively.
        '''
        def fXh(xe):
            return 1-(1-xe**0.2663)**1.3163
        
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
        if astro!=None:
            fX = astro['fX']
            Tmin_vir = astro['Tmin_vir']
        
        return 5e5*fX*fstar*fXh(xe)*Z*np.abs(hmf().dfcoll_dz(Z,Ho,Om_m,Om_b,Tcmbo, Tmin_vir))

#End of class heating
#========================================================================================================

class cosmic_history():  
    def __init__(self):
        return None

    #The following function has the differential equations governing the ionisation and thermal history.
    def _eqns(self, Z,V,Ho,Om_m,Om_b,Tcmbo,Yp,falp,fX,fesc,Tmin_vir):
        '''
        When solving upto the end of dark ages, only cosmological parameters will be used.
        Beyond Zstar, i.e., beginning of cosmic dawn astrophysical will also be used.
        '''
        xe = V[0]
        QHII = V[1]
        Tk = V[2]

        #eq1 is (1+z)d(xe)/dz; see Weinberg's Cosmology book or eq.(71) from Seager et al (2000), ApJSS
        eq1 = 1/basic_cosmo().H(Z,Ho,Om_m, Tcmbo)*recomb().Peebles_C(Z,xe,Tk, Ho,Om_m,Om_b,Tcmbo,Yp)*(xe**2*basic_cosmo().nH(Z,Ho,Om_b,Yp)*recomb().alpha(Tk)-recomb().beta(Tk)*(1-xe)*np.exp(-Ea/(kB*Tk)))

        #eq2 is (1+z)dQ/dz; Pritchard & Furlanetto (2007) eq.(11)
        #eq3 is (1+z)dT/dz; see eq.(2.31) from Mittal et al (2022), JCAP
        
        if Z>Zstar:
            eq2 = 0
            eq3 = 2*Tk-Tk*eq1/(1+basic_cosmo().xHe(Yp)+xe)-heating().Ecomp(Z,xe,Tk,Ho,Om_m,Tcmbo,Yp)
        else:
            if QHII<0.99:
                eq2 = 1/basic_cosmo().H(Z,Ho,Om_m, Tcmbo)*(alpha_A*2*basic_cosmo().nH(Z,Ho,Om_b,Yp)*xe*QHII) + (1-xe)*(1+basic_cosmo.xHe(Yp))*fstar*fesc*Nion*Z*hmf().dfcoll_dz(Z,Ho,Om_m,Om_b,Tcmbo, Tmin_vir)
            else:
                eq2 = 0
            eq3 = 2*Tk-Tk*eq1/(1+basic_cosmo().xHe(Yp)+xe)-heating().Ecomp(Z,xe,Tk,Ho,Om_m,Tcmbo,Yp)-heating().Ex(Z,xe,Ho,Om_m,Om_b,Tcmbo,fX,Tmin_vir)-heating().Elya(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp,falp,Tmin_vir)
        
        return np.array([eq1,eq2,eq3])

    def run_solver(self,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fX=0.1,fesc=0.1,Tmin_vir=1e4,Z_start=1501,Z_end=6,Z_eval=Z_default, xe_init=None,Tk_init=None,cosmo=None, astro=None):
        
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
            Yp = cosmo['Yp']
        if astro!=None:
            falp = astro['falp']
            fX = astro['fX']
            fesc = astro['fesc']
            Tmin_vir = astro['Tmin_vir']

        if Z_start==1501:
            Tk_init = basic_cosmo().Tcmb(Z_start,Tcmbo)
            xe_init = recomb().Saha_xe(Z_start,Tk_init, Ho,Om_b,Yp)
        elif xe_init==None and Tk_init==None:
            raise Exception('Initial conditions missing.')
            
        Sol = scint.solve_ivp(lambda a, Var: -self._eqns(1/a,Var,Ho,Om_m,Om_b,Tcmbo,Yp,falp,fX,fesc,Tmin_vir)/a, [1/Z_start, 1/Z_end],[xe_init,0,Tk_init],method='Radau',t_eval=1/Z_eval) 
        
        #Obtaining the solutions ...
        xe=Sol.y[0]
        QHII=Sol.y[1]
        Tk=Sol.y[2]

        return [xe,QHII,Tk]

#End of class cosmic_history
#========================================================================================================

class hyperfine():
    '''
    This class has all the functions related to computation of 21-cm signal, :math:`T_{21}`. These are
    
    :math:`\\kappa_{\\mathrm{HH}}, \\kappa_{\\mathrm{eH}}, x_{\\mathrm{k}}, x_{\\alpha}, T_{\\mathrm{s}}` and :math:`T_{21}`.
    
    Methods
    ~~~~~~~
    '''
    def __init__(self):
        return None

    def kHH(self,Tk):
        '''
        Volumetric spin flip rate for hydrogen-hydrogen collision. This fitting function and the next one is available from `Pritchard & Loeb (2012) <https://ui.adsabs.harvard.edu/abs/2012RPPh...75h6901P/abstract>`__.
        
        Arguments
        ---------
        
        Tk : float
            Gas kinetic temperature.
            
        Returns
        -------
        
        float
            :math:`k_{\\mathrm{HH}}` in units of :math:`\\mathrm{m^3s^{-1}}`.
        '''
        return 3.1*10**(-17)*Tk**0.357*np.exp(-32/Tk)

    def keH(self,Tk):
        '''
        Volumetric spin flip rate for electron-hydrogen collision.
        
        Arguments
        ---------
        
        Tk : float
            Gas kinetic temperature.
            
        Returns
        -------
        
        float
            :math:`k_{\\mathrm{eH}}` in units of :math:`\\mathrm{m^3s^{-1}}`.
        '''
        return np.where(Tk<10**4,10**(-15.607+0.5*np.log10(Tk)*np.exp(-(np.abs(np.log10(Tk)))**4.5/1800) ),10**(-14.102))

    def col_coup(self,Z,xe,Tk, Ho=67.4,Om_b=0.049,Tcmbo=2.725,Yp=0.245,cosmo=None):
        '''
        Collisional coupling
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature.
            
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``.            
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``.
        
        Returns
        -------
        
        float 
            :math:`x_{\\mathrm{k}}`, dimensionless.
        '''
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
            Yp = cosmo['Yp']

        return Tstar*basic_cosmo().nH(Z,Ho,Om_b,Yp)*((1-xe)*self.kHH(Tk)+xe*self.keH(Tk))/(A10*basic_cosmo().Tcmb(Z,Tcmbo))

    def lya_coup(self,Z,xe,Tk, Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,Tmin_vir=1e4,cosmo=None,astro=None):
        '''
        Ly-:math:`\\alpha` coupling or the Wouthuysen--Field coupling.
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature.
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
        
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``.            
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``.
        
        falp : float, optional
            :math:`f_{\\alpha}`, a dimensionless parameter which controls the emissivity of the Lyman series photons. Default value 1.
        
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.
        
        Returns
        -------
        
        float
            :math:`x_{\\alpha}`, dimensionless.
        '''
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
            Yp = cosmo['Yp']
        if astro!=None:
            falp = astro['falp']
            Tmin_vir = astro['Tmin_vir']
        
        S = extras().scatter_corr(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)
        J = extras().lya_spec_inten(Z,Ho,Om_m,Om_b,Tcmbo,falp,Tmin_vir)    #'undistorted' background Spec. Inte. of Lya photons.
        Jo = 5.54e-8*Z         #eq.(24) in Mittal & Kulkarni (2021)
        return S*J/Jo

    def spin_temp(self,Z,xe,Tk, Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,Tmin_vir=1e4,cosmo=None,astro=None):
        '''
        Spin temperature.
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature.
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
        
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``.            
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``.
        
        falp : float, optional
            :math:`f_{\\alpha}`, a dimensionless parameter which controls the emissivity of the Lyman series photons. Default value 1.
        
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.
        
        Returns
        -------
        
        float
            :math:`T_{\\mathrm{s}}`, K.
        '''
        if cosmo!=None and astro!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
            Yp = cosmo['Yp']
            
            falp = astro['falp']
            Tmin_vir = astro['Tmin_vir']

        xa = self.lya_coup(Z,xe,Tk, Ho,Om_m,Om_b,Tcmbo,Yp, falp,Tmin_vir)
        xk = self.col_coup(Z,xe,Tk, Ho,Om_b,Tcmbo,Yp)
        Ts = ( 1  + xa + xk)/(1/basic_cosmo().Tcmb(Z, Tcmbo) +  (xk+xa)/Tk )    #We assume the colour temperature is same as Tk.
        return Ts

    def Terb(self,Z,Tcmbo,zeta_erb): #Net background temperature (includes CMB)
        return np.where(Z<Zstar,Tcmbo*Z*(1+0.169*zeta_erb*Z**2.6),basic_cosmo().Tcmb(Z,Tcmbo))

    def twentyone_cm(self,Z,xe,Tk, Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,Tmin_vir=1e4,cosmo=None,astro=None):
        '''
        The global (sky-averaged) 21-cm signal.
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature.
        
        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_m : float, optical
            Relative matter density. Default value ``0.315``.
        
        Om_b : float, optical
            Relative baryon density. Default value ``0.049``.            
        
        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``.
        
        falp : float, optional
            :math:`f_{\\alpha}`, a dimensionless parameter which controls the emissivity of the Lyman series photons. Default value 1.
        
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.
        
        Returns
        -------
        
        float
            :math:`T_{21}`, mK.
        '''
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
            Yp = cosmo['Yp']
        if astro!=None:
            falp = astro['falp']
            Tmin_vir = astro['Tmin_vir']

        xHI=(1-xe)
        h100 = Ho/100
        Ts = self.spin_temp(Z,xe,Tk, Ho, Om_m, Om_b, Tcmbo, Yp,falp,Tmin_vir)
        return 27*xHI*((1-Yp)/0.76)*(Om_b*h100**2/0.023)*np.sqrt(0.15*Z/(10*Om_m*h100**2))*(1-basic_cosmo().Tcmb(Z)/Ts)

#========================================================================================================

class pipeline():
    '''
    This class runs the cosmic history solver and produces the global signal and the corresponding redshifts. Additionally, this class also has the function plotter which you can use to plot gas temperature, spin temperature or the global signal.
    
    Methods
    ~~~~~~~
    '''
    def __init__(self,cosmo={'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'Tcmbo':2.725,'Yp':0.245},astro= {'falp':1,'fX':0.1,'fesc':0.1,'Tmin_vir':1e4},Z_eval=None,path=''):

        self.cosmo=cosmo
        self.astro=astro

        self.path=path

        self.Z_eval = Z_eval
        self.Ho = cosmo['Ho']
        self.Om_m = cosmo['Om_m']
        self.Om_b = cosmo['Om_b']
        self.Tcmbo = cosmo['Tcmbo']
        self.Yp = cosmo['Yp']
        
        self.falp = astro['falp']
        self.fX = astro['fX']
        self.fesc = astro['fesc']
        self.Tmin_vir = astro['Tmin_vir']
        
        if os.path.isdir(self.path)==False:
            print('The requested directory does not exist. Creating one ...')
            os.mkdir(self.path)
        
        self.timestamp = strftime("%Y%m%d%H%M%S", localtime())
        self.path = self.path + 'output_'+self.timestamp+'/'
        os.mkdir(self.path)

        self.formatted_timestamp = self.timestamp[8:10]+':'+self.timestamp[10:12]+':'+self.timestamp[12:14]+' '+self.timestamp[6:8]+'/'+self.timestamp[4:6]+'/'+ self.timestamp[:4]
        return None

    def glob_sig(self):

        comm = MPI.COMM_WORLD
        cpu_ind = comm.Get_rank()
        n_cpu = comm.Get_size()

        model = 0
        for keys in self.astro.keys():
            if np.size(self.astro[keys])>1:
                model = model+1
                break
                
        for keys in self.cosmo.keys():
            if np.size(self.cosmo[keys])>1:
                model = model+2
                break
        
        if model==0:
            self.astro=extras().to_float(self.astro)
            self.cosmo=extras().to_float(self.cosmo)
        elif model==1:
            self.astro=extras().to_array(self.astro)
            self.cosmo=extras().to_float(self.cosmo)
        elif model==3:
            self.astro=extras().to_array(self.astro)
            self.cosmo=extras().to_array(self.cosmo)
        else:
            print('Currently not designed to work with varying cosmological parameters only!')
            sys.exit()
                
        
                    
        if model==0:
        #Cosmological and astrophysical parameters are fixed.
            if cpu_ind==0:
                print_banner()
                print('Both cosmological and astrophysical parameters are fixed.\n')
                
                if type(self.Z_eval)==np.ndarray or type(self.Z_eval)==list:
                    self.Z_eval=np.array(self.Z_eval)
                    if self.Z_eval[1]>self.Z_eval[0]:
                        self.Z_eval = self.Z_eval[::-1]
                elif self.Z_eval==None:
                    self.Z_eval = np.linspace(1501,6,1500)
                else:
                    print('\033[31mError! Z_eval not recognised!\033[00m')
                    sys.exit()
                
                st = time.process_time()
                
                print('Obtaining the thermal and ionisation history ...')
                sol = cosmic_history().run_solver(self.Ho,self.Om_m,self.Om_b,self.Tcmbo,self.Yp,self.falp,self.fX,self.fesc,self.Tmin_vir,1501,6,self.Z_eval)
                
                x_glob = sol[1] + (1-sol[1])*sol[0]

                print('Obtaining spin temperature ...')
                Ts = hyperfine().spin_temp(self.Z_eval,x_glob,sol[2], self.Ho,self.Om_m, self.Om_b,self.Tcmbo, self.Yp, self.falp,self.Tmin_vir)

                print('Computing the 21-cm signal ...')
                T21 = hyperfine().twentyone_cm(self.Z_eval,x_glob,sol[2], self.Ho,self.Om_m, self.Om_b,self.Tcmbo, self.Yp, self.falp,self.Tmin_vir)
                
                print('Done.')

                xe_save_name = self.path+'xe'
                Q_save_name = self.path+'Q'
                Tk_save_name = self.path+'Tk'
                Ts_save_name = self.path+'Ts'
                Tcmb_save_name = self.path+'Tcmb'
                T21_save_name = self.path+'T21'
                z_save_name = self.path+'one_plus_z'
                
                np.save(xe_save_name,sol[0])
                np.save(Q_save_name,sol[1])
                np.save(Tk_save_name,sol[2])
                np.save(Ts_save_name,Ts)
                np.save(Tcmb_save_name,basic_cosmo().Tcmb(self.Z_eval,self.Tcmbo))
                np.save(T21_save_name,T21)
                np.save(z_save_name,self.Z_eval)
                
                print('\033[32m1+z, xe, Q_HII, Tk, Ts, T_CMB, and T_21 have been saved into folder:',self.path,'\033[00m')
                
                et = time.process_time()
                # get the execution time
                elapsed_time = et - st
                print('\nExecution time: %.2f seconds' %elapsed_time)

                #========================================================
                #Writing to a summary file
                max_T21 = np.min(T21)
                max_ind = np.where(T21==max_T21)
                [max_z] = self.Z_eval[max_ind]

                sumfile = self.path+"summary_"+self.timestamp+".txt"
                myfile = open(sumfile, "w")
                myfile.write('''\n███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  ██╗
██╔════╝██╔════╝██║  ██║██╔═══██╗╚════██╗███║
█████╗  ██║     ███████║██║   ██║ █████╔╝╚██║
██╔══╝  ██║     ██╔══██║██║   ██║██╔═══╝  ██║
███████╗╚██████╗██║  ██║╚██████╔╝███████╗ ██║
╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═╝\n''')
                myfile.write('Shikhar Mittal, 2024\n')
                myfile.write('\nThis is output_'+self.timestamp)
                myfile.write('\n------------------------------\n')
                myfile.write('\nGMT time stamp: '+self.formatted_timestamp)
                myfile.write('\n\nExecution time: %.2f seconds' %elapsed_time) 
                myfile.write('\n')
                myfile.write('\nParameters given:\n')
                myfile.write('Ho = {}'.format(self.Ho))
                myfile.write('\nOm_m = {}'.format(self.Om_m))
                myfile.write('\nOm_b = {}'.format(self.Om_b))
                myfile.write('\nTcmbo = {}'.format(self.Tcmbo))
                myfile.write('\nYp = {}'.format(self.Yp))
                myfile.write('\n\nfalp = {}'.format(self.falp))
                myfile.write('\nfX = {}'.format(self.fX))
                myfile.write('\nfesc = {}'.format(self.fesc))
                myfile.write('\nmin(T_vir) = {}'.format(self.Tmin_vir))
                myfile.write('\n')
                myfile.write('\nStrongest signal is {:.2f} mK, observed at z = {:.2f}'.format(max_T21,max_z-1))
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
                return None
            
        elif model==1:
        #Cosmological parameters are fixed so dark ages is solved only once.
            if cpu_ind==0:
                print_banner()
                print('Cosmological parameter are fixed. Astrophysical parameters are varied.')
            
            if(n_cpu==1):
                print("\033[91mBetter to parallelise. Eg. 'mpirun -np 4 python3 %s', where 4 specifies the number of tasks.\033[00m" %(sys.argv[0]))

            if cpu_ind==0: print('Generating once the thermal and ionisation history up to dark ages ...')
            Z_da = np.linspace(1501,Zstar,1400)
            sol_da = cosmic_history().run_solver(self.Ho,self.Om_m,self.Om_b,self.Tcmbo,self.Yp,self.falp[0],self.fX[0],self.fesc[0],self.Tmin_vir[0],Z_start=1501,Z_end=Zstar, Z_eval=Z_da)
            xe_da = sol_da[0]
            Tk_da = sol_da[1]

            if type(self.Z_eval)==np.ndarray or type(self.Z_eval)==list:
                self.Z_eval=np.array(self.Z_eval)
                if self.Z_eval[1]>self.Z_eval[0]:
                    self.Z_eval = self.Z_eval[::-1]
                if self.Z_eval[0]>self.Zstar:
                    print('Error: first value should be below or equal to Zstar (= 60)')
                    sys.exit()
            elif self.Z_eval==None:
                self.Z_eval = np.linspace(Zstar,6,200)

            n_values=len(self.Z_eval)
            
            n_mod = extras().no_of_mdls(self.astro)
            arr = np.arange(n_mod)
            arr = np.reshape(arr,[np.size(self.falp),np.size(self.fX),np.size(self.fesc),np.size(self.Tmin_vir)])
            T21 = np.zeros((np.size(self.falp),np.size(self.fX),np.size(self.fesc),np.size(self.Tmin_vir),n_values))
            
            if cpu_ind==0: print('Done.\nGenerating',n_mod,'21-cm global signals ...\n')
            st = time.process_time()
            for i in range(n_mod):
                if (cpu_ind == int(i/int(n_mod/n_cpu))%n_cpu):
                    ind=np.where(arr==i)
                    sol_cd = cosmic_history().run_solver(self.Ho,self.Om_m,self.Om_b,self.Tcmbo,self.Yp,self.falp[ind[0][0]],self.fX[ind[1][0]],self.fesc[ind[2][0]],
                    self.Tmin_vir[ind[3][0]],Zstar,6,self.Z_eval,xe_da[-1],Tk_da[-1])
                    T21[ind[0][0],ind[1][0],ind[2][0],ind[3][0],:]= hyperfine().twentyone_cm(self.Z_eval,sol_cd[0],sol_cd[1],self.Ho,self.Om_m,self.Om_b,self.Tcmbo,self.Yp,self.falp[ind[0][0]],self.Tmin_vir[ind[3][0]])
            
            comm.Barrier()
            if cpu_ind!=0:
                comm.send(T21, dest=0)
            else:
                print('Done.')
                for j in range(1,n_cpu):
                    T21 = T21 + comm.recv(source=j)
                
                T21_save_name = self.path+'T21_'+str(np.size(self.falp))+str(np.size(self.fX))+str(np.size(self.fesc))+str(np.size(self.Tmin_vir))
                z_save_name = self.path+'one_plus_z'
                
                np.save(T21_save_name,T21)
                np.save(z_save_name,self.Z_eval)
                print('\033[32m\nYour T21s have been saved into file:',T21_save_name,'\033[00m')
                
                et = time.process_time()
                # get the execution time
                elapsed_time = et - st
                print('\nProcessing time: %.2f seconds' %elapsed_time)

                #========================================================
                #Writing to a summary file

                sumfile = self.path+"summary_"+self.timestamp+".txt"
                myfile = open(sumfile, "w")
                myfile.write('''\n███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  ██╗
██╔════╝██╔════╝██║  ██║██╔═══██╗╚════██╗███║
█████╗  ██║     ███████║██║   ██║ █████╔╝╚██║
██╔══╝  ██║     ██╔══██║██║   ██║██╔═══╝  ██║
███████╗╚██████╗██║  ██║╚██████╔╝███████╗ ██║
╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═╝\n''')
                myfile.write('Shikhar Mittal, 2024\n')
                myfile.write('\nThis is output_'+self.timestamp)
                myfile.write('\n------------------------------\n')
                myfile.write('\nTime stamp: '+self.formatted_timestamp)
                myfile.write('\n\nExecution time: %.2f seconds' %elapsed_time) 
                myfile.write('\n')
                myfile.write('\nParameters given:\n')
                myfile.write('Ho = {}'.format(self.Ho))
                myfile.write('\nOm_m = {}'.format(self.Om_m))
                myfile.write('\nOm_b = {}'.format(self.Om_b))
                myfile.write('\nTcmbo = {}'.format(self.Tcmbo))
                myfile.write('\nYp = {}'.format(self.Yp))
                myfile.write('\n\nfalp = {}'.format(self.falp))
                myfile.write('\nfX = {}'.format(self.fX))
                myfile.write('\nfesc = {}'.format(self.fesc))
                myfile.write('\nmin(T_vir) = {}'.format(self.Tmin_vir))
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')

        elif model==3:

            if cpu_ind==0:
                print_banner()
                print('Both cosmological and astrophysical parameters are varied.')
            
            if type(self.Z_eval)==np.ndarray or type(self.Z_eval)==list:
                self.Z_eval=np.array(self.Z_eval)
                if self.Z_eval[1]>self.Z_eval[0]:
                    self.Z_eval = self.Z_eval[::-1]
                if self.Z_eval[0]>1501 or self.Z_eval[-1]<6:
                    print('Error: redshift values not within the range')
                    sys.exit()
            elif self.Z_eval==None:
                self.Z_eval = np.linspace(1501,6,2000)
            
            n_values=len(self.Z_eval)
                
            if(n_cpu==1):
                print('Error: you want to generate global signals for multiple parameter values.')
                print("Run as, say, 'mpirun -n 4 python3 %s', where 4 specifies the number of CPUs." %(sys.argv[0]))
                sys.exit()		
            
            n_mod = extras().no_of_mdls(self.astro)*extras().no_of_mdls(self.cosmo)
            arr = np.arange(n_mod)
            arr = np.reshape(arr,[np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.falp),np.size(self.fX),np.size(self.fesc),np.size(self.Tmin_vir)])
            T21 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.Tcmbo),np.size(self.Yp),
            np.size(self.falp),np.size(self.fX),np.size(self.fesc),np.size(self.Tmin_vir),n_values))
            
            if cpu_ind==0: print('Generating',n_mod,'models ...')
            st = time.process_time()
            
            for i in range(n_mod):
                if (cpu_ind == int(i/int(n_mod/n_cpu))%n_cpu):
                    ind=np.where(arr==i)
                    sol = cosmic_history().run_solver(self.Ho[ind[0][0]],self.Om_m[ind[1][0]],self.Om_b[ind[2][0]],self.Tcmbo[ind[3][0]],self.Yp[ind[4][0]],
                    self.falp[ind[5][0]],self.fX[ind[6][0]],self.fesc[ind[7][0]],self.Tmin_vir[ind[8][0]],Z_start=1501,Z_end=6,Z_eval=self.Z_eval)
                    T21[ind[0][0],ind[1][0],ind[2][0],ind[3][0],[ind[4][0]],[ind[5][0]],
                    [ind[6][0]],[ind[7][0]],[ind[8][0]],:] = hyperfine().twentyone_cm(self.Z_eval,sol[0],sol[1],self.Ho[ind[0][0]],self.Om_m[ind[1][0]],self.Om_b[ind[2][0]],self.Tcmbo[ind[3][0]],self.Yp[ind[4][0]],self.falp[ind[5][0]],self.Tmin_vir[ind[8][0]])
            
            comm.Barrier()
            if cpu_ind!=0:
                comm.send(T21, dest=0)
            else:
                print('Done.')
                for j in range(1,n_cpu):
                    T21 = T21 + comm.recv(source=j)
                save_name = self.path+'T21_'+str(np.size(self.Ho))+str(np.size(self.Om_m))+str(np.size(self.Om_b))+str(np.size(self.Tcmbo))+str(np.size(self.Yp))+str(np.size(self.falp))+str(np.size(self.fX))+str(np.size(self.fesc))+str(np.size(self.Tmin_vir))+'.npy'
                np.save(save_name,T21)
                
                print('\033[32m\nYour T21s have been saved into file:',T21_save_name,'\033[00m')
                
                et = time.process_time()
                # get the execution time
                elapsed_time = et - st
                print('\nProcessing time: %.2f seconds' %elapsed_time)
                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
        return None
    #End of function glob_sig               

    def Z2nu(self,Z):
        return 1420/Z

    def nu2Z(self,nu):
        return 1420/nu

    def plotter(self,x=None,y=None,xlog=True,ylog=False,quant_name='',add_edges=False,xlow=6,xhigh=200):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        if (x==None and y==None):
            x=np.load(self.path+'one_plus_z.npy')
            y=np.load(self.path+'T21.npy')
            quant_name = 'T21'

        fig,ax=plt.subplots(figsize=(8.3,7.5),dpi=300)
        fig.subplots_adjust(left=0.12, bottom=0.07, right=0.88, top=0.97)
        clr=['b','r','limegreen']
        linsty=['-','--',':']
        if type(y)==dict:
            leng = len(y)
            keys = list(y.keys())
            for i in range(leng):
                if keys[i]=='Tk':
                    lbl = '$T_{\mathrm{k}}$'
                elif keys[i]=='Ts':
                    lbl = '$T_{\mathrm{s}}$'
                elif keys[i]=='Tcmb':
                    lbl = '$T_{\mathrm{cmb}}$'
                else:
                    print('Warning: unknown quantity given in the dictionary!')
                ax.plot(x,y[keys[i]],color=clr[i],ls=linsty[i],label=lbl)
            if leng>1:
                ax.set_ylabel(r'$T\,$(K)',fontsize=20)
                ax.legend(fontsize=18,frameon=False)
            else:
                if quant_name=='Tk':
                    ax.set_ylabel(r'$T_{\mathrm{k}}\,$(K)',fontsize=20)
                elif quant_name=='Tcmb':
                    ax.set_ylabel(r'$T_{\mathrm{cmb}}\,$(K)',fontsize=20)
                elif quant_name=='Ts':
                    ax.set_ylabel(r'$T_{\mathrm{s}}\,$(K)',fontsize=20)
                
                
        elif type(y)==np.ndarray:
            ax.plot(x,y,'b')
            if quant_name=='xe':
                ax.set_ylabel(r'$x_{\mathrm{e}}$',fontsize=20)
            elif quant_name=='Tk':
                ax.set_ylabel(r'$T_{\mathrm{k}}\,$(K)',fontsize=20)
            elif quant_name=='Tcmb':
                ax.set_ylabel(r'$T_{\mathrm{cmb}}\,$(K)',fontsize=20)
            elif quant_name=='Ts':
                ax.set_ylabel(r'$T_{\mathrm{s}}\,$(K)',fontsize=20)
            elif quant_name=='sfrd' or quant_name=='SFRD':
                ax.set_ylabel(r'$\dot{\rho}_{\star}\,(\mathrm{kg\,m^{-3}s^{-1}})$',fontsize=20)
            elif quant_name=='T21':
                if add_edges==True:
                    nu_edges=np.load('nu_edges.npy')
                    Z_edges=1420/nu_edges
                    T21_edges=np.load('T21_edges.npy')
                    res=np.load('residue.npy')
                    ax.plot(Z_edges,1000*(T21_edges+res),'r--',lw=1.5)
                    ax.legend(['Theory','EDGES'],fontsize=18,frameon=False)
                    secax = ax.secondary_xaxis('top', functions=(self.Z2nu,self.nu2Z))
                    secax.set_xlabel(r'$\nu\,(\mathrm{MHz})$',fontsize=20, labelpad=12)
                    secax.minorticks_on()
                    secax.tick_params(axis='x',which='major', length=5, width=1, labelsize=20,direction='in')
                    secax.tick_params(axis='x',which='minor', length=3, width=1, direction='in')
                ax.axhline(y=0,ls=':',color='k')
                ax.set_xlim([xlow,xhigh])
                ax.set_ylabel(r'$T_{21}\,$(mK)',fontsize=20)
            else:
                print("Warning: enter the quantity name. Use argument 'quant_name'")
            
        else:
            print("Error: incorrect syntax! Give y as an array or dictionary of arrays. eg. {'Tk':Tk,'Ts':Ts,'Tcmb':Tcmb}")
            sys.exit()

        if xlog==True:
            ax.set_xscale('log')
        if ylog==True:
            ax.set_yscale('log')
        
        ax.set_xlabel(r'$1+z$',fontsize=20)    
        ax.tick_params(axis='both', which='major', length=5, width=1, labelsize=20,direction='in')
        ax.tick_params(axis='both', which='minor', length=3, width=1, direction='in')
        ax.minorticks_on()
        ax.invert_xaxis()
        ax.yaxis.set_ticks_position('both')
        if add_edges==False:
            ax.xaxis.set_ticks_position('both')
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.savefig(self.path+quant_name+'.pdf')
        return
#End of class pipeline
#========================================================================================================

class extras():
    def __init__(self):
        return None

    def to_array(self,params):
        for keys in params.keys():
            if type(params[keys])==list:
                params[keys]=np.array(params[keys])
            elif type(params[keys])==float or type(params[keys])==int:
                params[keys]=np.array([params[keys]])
        return params

    def to_float(self,params):
        for keys in params.keys():
            if type(params[keys])==list:
                [params[keys]]=params[keys]
            elif type(params[keys])==np.ndarray:
                params[keys]=params[keys][0]
        return params
        
    def no_of_mdls(self,params):
        prod=1
        for keys in params.keys():
            if type(params[keys])==np.ndarray:
                prod=prod*len(params[keys])
        return prod

    def clump(self,Z):
        '''
        Clumping factor. Taken from Madau & Fragos (2017)
        '''
        return 20.81*Z**-1.1

    def recoil(self,Tk):
        '''
        The recoil parameter. Eq.(15) in Mittal & Kulkarni (2021).
        '''
        return 0.02542/np.sqrt(Tk)

    def dopp(self,Tk):
        '''
        Doppler width for Lya-HI interaction. Eq.(14) in Mittal & Kulkarni (2021).
        '''
        return nu_alpha*np.sqrt(2*kB*Tk/(mP*cE**2))

    def a_tau(self,Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp):
        '''
        Returns the product a*tau, since all the relevant formulae require the product only.
        a is the Voigt parameter and tau is the optical depth of Lya photons.
        '''
        tau = 3/(8*np.pi)*A_alpha/basic_cosmo().H(Z,Ho,Om_m,Tcmbo)*basic_cosmo().nH(Z,Ho,Om_b,Yp)*(1-xe)*lam_alpha**3
        a = A_alpha/(4*np.pi*self.dopp(Tk))
        return a*tau
        
    def zeta(self,Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp):
        '''
        A dimensionless number. See below Eq.(12) in Chuzhoy & Shapiro (2006).
        '''
        return 4/3*np.sqrt(self.a_tau(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)*self.recoil(Tk)**3/np.pi)

    def xi2(self,Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp):
        '''
        A dimensionless number. Eq.(39) in Mittal & Kulkarni (2021).
        '''
        return (4*self.a_tau(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)*self.recoil(Tk)**3/np.pi)**(1/3)

    def scatter_corr(self,Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp):
        '''
        This is the scattering correction, S. I am using the approximate version from Chuzhoy & Shapiro (2006).
        '''
        return np.exp(-1.69*self.zeta(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)**0.667)
        
    def _eps_alpha_beta(self,Z,E, Ho,Om_m,Om_b,Tcmbo,Tmin_vir):
        phi = hP/eC*2902.91*(E/13.6)**-0.86
        return 1/(1.22*mP)*phi*hmf().SFRD(Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir)

    def _eps_above_beta(self,Z,E, Ho,Om_m,Om_b,Tcmbo,Tmin_vir):
        '''
        Comoving emissivity in units of number per unit time per unit frequency per unit volume (s^-1.m^-3.Hz^-1)
        Valid only for photons of energy above Ly beta, i.e., E > 12.089 eV
        '''
        phi = hP/eC*1303.34*(E/13.6)**-7.658 #this is the SED in units of number per baryon per unit frequency (Hz^-1)
        return 1/(1.22*mP)*phi*hmf().SFRD(Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir)

    def lya_spec_inten(self,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,falp=1,Tmin_vir=1e4,cosmo=None,astro=None):
        '''
        Specific intensity of Ly-alpha photons in terms of number per unit time per unit area per unit frequency per unit solid angle
        (m^-2.s^-1.Hz^-1.sr^-1)
        '''
        if cosmo!=None and astro!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            Tcmbo = cosmo['Tcmbo']
            
            falp = astro['falp']
            Tmin_vir = astro['Tmin_vir']
            
        loc=0
        flag=False
        integ=0
        if type(Z)==float or type(Z)==int:
            if Z>Zstar:
                return 0
            Zmax = 32/27*Z
            temp = np.linspace(Z,Zmax,10)
            integ = scint.trapz(self._eps_alpha_beta(temp,10.2*temp/Z, Ho,Om_m,Om_b,Tcmbo,Tmin_vir)/basic_cosmo().H(temp, Ho, Om_m,Tcmbo),temp)
            for ni in np.arange(4,24):
                Zmax = (1-1/(ni+1)**2)/(1-1/ni**2)*Z
                temp = np.linspace(Z,Zmax,5)
                integ = integ+Pn[ni-4]*scint.trapz(self._eps_above_beta(temp,13.6*(1-1/ni**2)*temp/Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir)/basic_cosmo().H(temp, Ho, Om_m,Tcmbo),temp)
        
        elif type(Z)==np.ndarray or type(Z)==list:
            if Z[0]>Zstar:
                flag=True
                loc = np.where(Z<Zstar)[0][0]
                Z=Z[loc:]
            
            counter=0
            numofZ = len(Z)
            integ=np.zeros(numofZ)
            for Z_value in Z:
                Zmax = 32/27*Z_value
                temp = np.linspace(Z_value,Zmax,10)
                integ[counter] = scint.trapz(self._eps_alpha_beta(temp,10.2*temp/Z_value, Ho,Om_m,Om_b,Tcmbo,Tmin_vir)/basic_cosmo().H(temp, Ho, Om_m,Tcmbo),temp)

                for ni in np.arange(4,24):
                    Zmax = (1-1/(ni+1)**2)/(1-1/ni**2)*Z_value
                    temp = np.linspace(Z_value,Zmax,5)
                    integ[counter] = integ[counter]+Pn[ni-4]*scint.trapz(self._eps_above_beta(temp,13.6*(1-1/ni**2)*temp/Z_value,Ho,Om_m,Om_b,Tcmbo,Tmin_vir)/basic_cosmo().H(temp, Ho, Om_m,Tcmbo),temp)
                
                counter=counter+1
        

        J_temp = falp*cE/(4*np.pi)*Z**2*integ
        if flag == True:
            J_before_CD = np.zeros(loc)
            J_after_CD = J_temp
            return np.concatenate((J_before_CD,J_after_CD))
        else:
            return J_temp
#End of class extras
#========================================================================================================
