import scipy.special as scsp
import scipy.integrate as scint
from scipy.interpolate import CubicSpline
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
from tqdm import tqdm

#========================================================================================================
#Universal constants in SI units
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
year = 365*86400
Msolar_by_Mpc3_year_to_kg_by_m3_sec = Msolar*(1000*Mpc2km)**-3*year**-1

#-------------------------------------------------------------
#Hardcoded but later we want to change some of these

fstar = 0.1
Nion = 7253

Ji_to_Jc = 1 #Ratio of intensities of injected photons to continuum photons

Zstar = 60 #redshift of the beginning of star formation

Z_start = 1501
Z_end = 6

Z_default = np.concatenate((np.linspace(Z_start,Zstar+1,2000),1/np.linspace(1/Zstar,1/Z_end,200)))
#-------------------------------------------------------------
#Recombination related
Lam_H = 8.22458 #The H 2s–1s two photon rate in s^−1
A_rec,b_rec,c_rec,d_rec = 4.309, -0.6166, 0.6703, 0.53
Feff = 1.14 #This extra factor gives the effective 3-level recombination model
lam_alpha = 121.5682e-9 #Wavelength of Lya photon in m
nu_alpha = cE/lam_alpha #Frequency in Hz
B2 = 3.4*eC #Bind energy of level 2 in J
B1 = 13.6*eC #Bind energy of level 1 in J
Ea = B1-B2  #Energy of Lya photon in J
A_alpha = 6.25e8 #Spontaneous emission coeffecient in Hz
alpha_B = 1.43e-19 #Case-B recombination coefficient (m^3/s) at T=2 X 10^4 K
#-------------------------------------------------------------
#Others
T_se = 0.4 #Spin exchange correction (in Kelvin; Chuzhoy & Shapiro 2006)
Tstar = 0.068 #Hyperfine energy difference in temperature (K)
A10 = 2.85e-15 # Einstein's spontaneous emission rate, sec^-1
Pn=np.array([0.2609,0.3078,0.3259,0.3353,0.3410,0.3448,0.3476,0.3496,0.3512,0.3524,0.3535,0.3543,0.355,0.3556,
    0.3561,0.3565,0.3569,0.3572,0.3575,0.3578])
            
#========================================================================================================

def _print_banner():
    banner = """\n\033[94m
    ███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  ██╗
    ██╔════╝██╔════╝██║  ██║██╔═══██╗╚════██╗███║
    █████╗  ██║     ███████║██║   ██║ █████╔╝╚██║
    ██╔══╝  ██║     ██╔══██║██║   ██║██╔═══╝  ██║
    ███████╗╚██████╗██║  ██║╚██████╔╝███████╗ ██║
    ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═╝
    Copyright 2024, Shikhar Mittal.                                     
    \033[00m\n"""
    print(banner)
    return None

def _to_array(params):
        for keys in params.keys():
            if type(params[keys])==list:
                params[keys]=np.array(params[keys])
            elif type(params[keys])==float or type(params[keys])==int:
                params[keys]=np.array([params[keys]])
        return params

def _to_float(params):
    for keys in params.keys():
        if type(params[keys])==list:
            [params[keys]]=params[keys]
        elif type(params[keys])==np.ndarray:
            params[keys]=params[keys][0]
    return params
    
def _no_of_mdls(params):
    prod=1
    for keys in params.keys():
        if type(params[keys])==np.ndarray:
            prod=prod*len(params[keys])
    return prod

def _gaif(xe,Q):
    '''
    Computes the globally-averaged ionisation factor.
    
    Arguments
    ---------
    xe: float
        Is the electron fraction of the bulk IGM.

    Q: float
        Volume-filling factor.
    
    Return
    ------
    float
        :math:`x_{\\mathrm{i}}=Q+(1-Q)x_{\\mathrm{e}}`.

    '''
    return Q+(1-Q)*xe


class main():
    '''
    Function names starting with 'basic_cosmo' include the basic :math:`\\Lambda`CDM-cosmology-related functions, such as Hubble function, CMB temperature, etc.

    Function names starting with 'recomb' include recombination-physics-related functions.

    Function names starting with 'hmf' include HMF-related functions, i.e., :math:`\\mathrm{d}n/\\mathrm{d\\,ln}M`, :math:`\\mathrm{d}n/\\mathrm{d}M`, :math:`m_{\\mathrm{min}}`, :math:`f_{\\mathrm{coll}}`, :math:`\\mathrm{d}f_{\\mathrm{coll}}/\\mathrm{d}z`, and :math:`\\dot{\\rho}_{\\star}`. Use this function to set your choice of HMF model name and also the choice of star formation efficiency (SFE) model name. Eg. ``hmf.hmf_name = 'press74'`` and ``hmf.sfe_name = 'const'``. Available HMF model names:
        - press74 (default, for Press & Schechter 1974),
        - sheth99 (for Sheth & Tormen 1999)

    Function names starting with 'heating' include all the heating (or cooling) terms. All the terms are in the form of :math:`-(1+z)\\mathrm{d}T_{\\mathrm{k}}/\\mathrm{d}z` and hence in units of temperature.
    
    Function names starting with 'hyfi' include all the functions related to computation of 21-cm signal. These are
    :math:`\\kappa_{\\mathrm{HH}}, \\kappa_{\\mathrm{eH}}, x_{\\mathrm{k}}, x_{\\alpha}, T_{\\mathrm{s}}` and :math:`T_{21}`.

    Methods
    ~~~~~~~
    '''
    def __init__(self,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1.0,fX=0.1,fesc=0.1,Tmin_vir=1e4,cosmo=None,astro=None, hmf_name='press74',sfe_name='const'):
        '''
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
        
        fX : float, optional
            :math:`f_{\\mathrm{X}}`, a dimensionless parameter which controls the emissivity of the X-ray photons. Default value 0.1.
        
        fesc : float, optional
            :math:`f_{\\mathrm{esc}}`, a dimensionless parameter which controls the escape fraction of the ionising photons. Default value 0.1.

        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.

        Note: cosmological and astrophysical parameters can also be supplied through dictionaries ``cosmo`` and ``astro``.

        hmf_name : str, option
            HMF model name. Default 'press74' for Press & Schechter (1974).
        '''
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
        
        self.Ho = Ho
        self.Om_m = Om_m
        self.Om_b = Om_b
        self.Tcmbo = Tcmbo
        self.Yp = Yp
        
        self.falp = falp
        self.fX = fX
        self.fesc = fesc
        self.Tmin_vir = Tmin_vir

        self.hmf_name=hmf_name
        self.sfe_name=sfe_name

        self.my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
        cosmology.setCosmology('my_cosmo', self.my_cosmo)
        self.h100 = self.Ho/100

        return None

    def basic_cosmo_mu(self,xe):
        '''
        Average baryon mass in amu

        Arguments
        ---------

        xe : float
            Electron fraction (dimensionless)
           
        Returns
        -------

        float
            Average baryon mass in amu
        
        '''
        return 4/(4-3*self.Yp+4*xe*(1-self.Yp))

    def basic_cosmo_xHe(self):
        '''
        Ratio of helium number density to hydrogen number density

        Arguments
        ---------

        No arguments required.

        Returns
        -------

        float
            :math:`n_{\\mathrm{He}}/n_{\\mathrm{H}}`
        
        '''
        return 0.25*self.Yp/(1-self.Yp)

    def basic_cosmo_Tcmb(self,Z):
        '''
        CMB temperature at a given redshift

        Arguments
        ---------

        Z : float
            1+z

        Returns
        -------

        float
            CMB temperature at the given redshift in kelvin
        '''
        return self.Tcmbo*Z

    def basic_cosmo_rho_crit(self):
        '''
        Critical density of the Universe today

        Arguments
        ---------

        No arguments required.

        Returns
        -------
        
        float
            Critical density today, :math:`\\rho_{\\mathrm{crit}}=\\frac{3H_0^2}{8\\pi G_{\\mathrm{N}}}` in units of :math:`\\mathrm{kg}\\,\\mathrm{m}^{-3}`
        '''
        return 3*self.Ho**2/(8*np.pi*GN*Mpc2km**2)

    def basic_cosmo_nH(self,Z):
        '''
        Hydrogen number density (proper)
        
        Arguments
        ---------
        
        Z : float
            1+z
        
        Returns
        -------
        
        float
            Proper hydrogen number density at given redshift in units of :math:`\\mathrm{m}^{-3}`
        '''
        return self.basic_cosmo_rho_crit()*self.Om_b*(1-self.Yp)*Z**3/mP

    def basic_cosmo_H(self,Z):
        '''
        Hubble factor in SI units
        
        Arguments
        ---------
        
        Z : float
            1+z
        
        Returns
        -------
        
        float
            Hubble parameter at a given redshift in units of :math:`\\mathrm{s}^{-1}`
        '''
        Om_lam = 1-self.Om_m
        Om_r = (1+fnu)*aS*self.Tcmbo**4/(cE**2*self.basic_cosmo_rho_crit())
        
        return self.Ho*(Om_r*Z**4+self.Om_m*Z**3+Om_lam)**0.5/Mpc2km

    #End of functions related to basic cosmology.
    #========================================================================================================
           
    def recomb_alpha(self, Tk):
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
        return (1e-19)*Feff*A_rec*t**b_rec/(1+c_rec*t**d_rec)

    def recomb_beta(self, Tk):
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
        return self.recomb_alpha(Tk)*(2*np.pi*me*kB*Tk/hP**2)**1.5*np.exp(-B2/(kB*Tk))

    def recomb_Krr(self, Z):
        '''
        Redshifting rate appearing in the Peebles' 'C' factor
        
        Arguments
        ---------
        
        Z : float
            1+z
        
        Returns
        -------
        
        float
            Redshifting rate in units of :math:`\\mathrm{m^3s}`
        
        '''
        return lam_alpha**3/(8*np.pi*self.basic_cosmo_H(Z))

    def recomb_Peebles_C(self,Z,xe,Tk):
        '''
        :math:`C_{\\mathrm{P}}`
        
        Arguments
        ---------
        
        Z : float
            1 + redshift, dimensionless
        
        xe : float
            Electron fraction, dimensionless
            
        Tk : float
            Gas temperature in units of kelvin
        
        Returns
        -------
        
        float
            Peebles 'C' factor appearing in Eq. (71) from `Seager et al (2000) <https://iopscience.iop.org/article/10.1086/313388>`__, dimensionless.
        '''
        
        return (1+self.recomb_Krr(Z)*Lam_H*self.basic_cosmo_nH(Z)*(1-xe))/(1+self.recomb_Krr(Z)*(Lam_H+self.recomb_beta(Tk))*self.basic_cosmo_nH(Z)*(1-xe))

    def recomb_Saha_xe(self,Z,Tk):
        '''
        Electron fraction predicted by the Saha's equation. This is important to initialise the differential equation for :math:`x_{\\mathrm{e}}`. At high redshift such as :math:`z\\approx1500`, Saha's equation gives accurate estimate of :math:`x_{\\mathrm{e}}`.
        
        Arguments
        ---------
        
        Z : float
            1 + redshift, dimensionless
          
        Tk : float
            Gas temperature in units of kelvin
        
        Returns
        -------
        
        float
            Electron fraction predicted by Saha's equation.
        '''
        Saha=1/self.basic_cosmo_nH(Z)*(2*np.pi*me*kB*Tk/hP**2)**1.5*np.exp(-B1/(kB*Tk))
        return (np.sqrt(Saha**2+4*Saha)-Saha)/2

    #End of functions related to recombination
    #========================================================================================================
        
    def hmf_dndlnM(self, M,Z):
        '''
        The halo mass function (HMF) in the form of :math:`\\mathrm{d}n/\\mathrm{d\\,ln}M`. Note the natural logarithm.
        
        Arguments
        ---------
        
        M : float
            The desired halo mass at which you want to evaluate HMF. Input M in units of solar mass.
        
        Z : float
            1 + redshift, dimensionless.
        
        Returns
        -------
        
        float
            HMF, :math:`\\mathrm{d}n/\\mathrm{d\\,ln}M=M\\mathrm{d}n/\\mathrm{d}M`, in units of :math:`\\mathrm{cMpc}^{-3}`, where 'cMpc' represents comoving Mega parsec.
        '''

        M_by_h = M*self.h100 #M in units of solar mass/h
        return self.h100**3*mass_function.massFunction(M_by_h, Z-1, q_in='M', q_out='dndlnM', model = self.hmf_name)

    def hmf_dndM(self,M,Z):
        '''
        The halo mass function (HMF) in a different form, i.e., :math:`\\mathrm{d}n/\\mathrm{d}M`.
        
        Arguments
        ---------
        
        M : float
            The desired halo mass at which you want to evaluate HMF. Input M in units of solar mass (:math:`\\mathrm{M}_{\\odot}`).
        
        Z : float
            1 + z, dimensionless.
        
        Returns
        -------
        
        float
            HMF in a different form, :math:`\\mathrm{d}n/\\mathrm{d}M`, in units of :math:`\\mathrm{cMpc}^{-3}\\mathrm{M}_{\\odot}^{-1}`, where 'cMpc' represents comoving Mega parsec and :math:`\\mathrm{M}_{\\odot}` represents the solar mass.
        '''

        return 1/M*self.hmf_dndlnM(M,Z)

    #For details see eq.(50),(52) and (53) from Mittal & Kulkarni (2021), MNRAS
    def hmf_m_min(self,Z):
        '''
        The minimum halo mass for which star formation is possible.
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless. It can be a single number or an array.
        
        Returns
        -------
        
        float
            The mass returned is in units of :math:`\\mathrm{M}_{\\odot}/h`.
        '''
        return 1e8*self.Om_m**(-0.5)*(10/Z*0.6/1.22*self.Tmin_vir/1.98e4)**1.5

    def hmf_f_coll(self,Z):
        '''
        Collapse fraction -- fraction of total matter that collapsed into the haloes. See definition below.
        :math:`F_{\\mathrm{coll}}=\\frac{1}{\\bar{\\rho}^0_{\\mathrm{m}}}\\int_{M_{\\mathrm{min}}}^{\\infty} M\\frac{\\mathrm{d}n}{\\mathrm{d} M}\\,\\mathrm{d} M\\,,`
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless. Can be a single quantity or an array.
        
        Returns
        -------
        
        float 
            Collapse fraction. Single number or an array accordingly as ``Z`` is single number or an array.
        '''

        if self.hmf_name=='press74':
            return scsp.erfc(peaks.peakHeight(self.hmf_m_min(Z),Z-1)/np.sqrt(2))
        else:
            numofZ = np.size(Z)
                
            if numofZ == 1:
                if type(Z)==np.ndarray: Z=Z[0]
                M_space = np.logspace(np.log10(self.hmf_m_min(Z)/self.h100),18,1500)    #These masses are in solar mass. 
                hmf_space = self.hmf_dndlnM(M=M_space,Z=Z)    #Corresponding HMF values are in cMpc^-3 
                rho_halo = Msolar_by_Mpc3_to_kg_by_m3*np.trapezoid(hmf_space,M_space)    #matter density collapsed as haloes (in kg/m^3, comoving)
            else:    
                rho_halo = np.zeros(numofZ)
                counter=0
                for i in Z:
                    M_space = np.logspace(np.log10(self.hmf_m_min(i)/self.h100),18,1500)    #These masses are in solar mass. 
                    hmf_space = self.hmf_dndlnM(M=M_space,Z=i)    #Corresponding HMF values are in cMpc^-3 
                    rho_halo[counter] = Msolar_by_Mpc3_to_kg_by_m3*np.trapezoid(hmf_space,M_space)    #matter density collapsed as haloes (in kg/m^3, comoving)
                    counter=counter+1
            return rho_halo/(self.Om_m*self.basic_cosmo_rho_crit())

    def hmf_dfcoll_dz(self,Z):
        '''
        Redshift derivative of the collapse fraction, i.e., :math:`\\mathrm{d}F_{\\mathrm{coll}}/\\mathrm{d}z`
        '''
        return (self.hmf_f_coll(Z+1e-3)-self.hmf_f_coll(Z))*1e3

     
    def hmf_SFRD(self,Z):
        '''
        This function returns the comoving star formation rate density (SFRD, :math:`\\dot{\\rho}_{\\star}`).
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless. Can be a single quantity or an array.
        
        Note: cosmological and astrophysical parameters can also be supplied through dictionaries ``cosmo`` and ``astro``.
        
        Returns
        -------
        
        float 
            Comoving SFRD in units of :math:`\\mathrm{kgs^{-1}m^{-3}}`. Single number or an array accordingly as ``Z`` is single number or an array.
        '''

        return -Z*fstar*self.Om_b*self.basic_cosmo_rho_crit()*self.hmf_dfcoll_dz(Z)*self.basic_cosmo_H(Z)

    #End of functions related to HMF
    #========================================================================================================

    def _recoil(self,Tk):
        '''
        The recoil parameter. Eq.(15) in Mittal & Kulkarni (2021).
        '''
        return 0.02542/np.sqrt(Tk)

    def _dopp(self,Tk):
        '''
        Doppler width for Lya-HI interaction. Eq.(14) in Mittal & Kulkarni (2021).
        '''
        return nu_alpha*np.sqrt(2*kB*Tk/(mP*cE**2))

    def _a_tau(self,Z,xe,Tk):
        '''
        Returns the product a*tau, since all the relevant formulae require the product only.
        a is the Voigt parameter and tau is the optical depth of Lya photons.
        '''
        tau = 3/(8*np.pi)*A_alpha/self.basic_cosmo_H(Z)*self.basic_cosmo_nH(Z)*(1-xe)*lam_alpha**3
        a = A_alpha/(4*np.pi*self._dopp(Tk))
        return a*tau
        
    def _zeta(self,Z,xe,Tk):
        '''
        A dimensionless number. See below Eq.(12) in Chuzhoy & Shapiro (2006).
        '''
        return 4/3*np.sqrt(self._a_tau(Z,xe,Tk)*self._recoil(Tk)**3/np.pi)

    def _xi2(self,Z,xe,Tk):
        '''
        A dimensionless number. Eq.(39) in Mittal & Kulkarni (2021).
        '''
        return (4*self._a_tau(Z,xe,Tk)*self._recoil(Tk)**3/np.pi)**(1/3)

    def _scatter_corr(self,Z,xe,Tk):
        '''
        This is the scattering correction, S. I am using the approximate version from Chuzhoy & Shapiro (2006).
        '''
        return np.exp(-1.69*self._zeta(Z,xe,Tk)**0.667)
        
    def _eps_alpha_beta(self,Z,E):
        phi = hP/eC*2902.91*(E/13.6)**-0.86
        return 1/(1.22*mP)*phi*self.hmf_SFRD(Z)

    def _eps_above_beta(self,Z,E):
        '''
        Comoving emissivity in units of number per unit time per unit frequency per unit volume (s^-1.m^-3.Hz^-1)
        Valid only for photons of energy above Ly beta, i.e., E > 12.089 eV
        '''
        phi = hP/eC*1303.34*(E/13.6)**-7.658 #this is the SED in units of number per baryon per unit frequency (Hz^-1)
        return 1/(1.22*mP)*phi*self.hmf_SFRD(Z)

    def _lya_spec_inten(self,Z):
        '''
        Specific intensity of Ly:math:`\\alpha` photons in terms of number per unit time per unit area per unit frequency per unit solid angle (:math:`\\mathrm{m^{-2}.s^{-1}.Hz^{-1}.sr^{-1}}`)
        '''
        loc=0
        flag=False
        integ=0
        if type(Z)==float or type(Z)==int:
            if Z>Zstar:
                return 0
            Zmax = 32/27*Z
            temp = np.linspace(Z,Zmax,10)
            integ = scint.trapezoid(self._eps_alpha_beta(temp,10.2*temp/Z)/self.basic_cosmo_H(temp),temp)
            for ni in np.arange(4,24):
                Zmax = (1-1/(ni+1)**2)/(1-1/ni**2)*Z
                temp = np.linspace(Z,Zmax,5)
                integ = integ+Pn[ni-4]*scint.trapezoid(self._eps_above_beta(temp,13.6*(1-1/ni**2)*temp/Z)/self.basic_cosmo_H(temp),temp)
        
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
                integ[counter] = scint.trapezoid(self._eps_alpha_beta(temp,10.2*temp/Z_value)/self.basic_cosmo_H(temp),temp)

                for ni in np.arange(4,24):
                    Zmax = (1-1/(ni+1)**2)/(1-1/ni**2)*Z_value
                    temp = np.linspace(Z_value,Zmax,5)
                    integ[counter] = integ[counter]+Pn[ni-4]*scint.trapezoid(self._eps_above_beta(temp,13.6*(1-1/ni**2)*temp/Z_value)/self.basic_cosmo_H(temp),temp)
                
                counter=counter+1
        

        J_temp = self.falp*cE/(4*np.pi)*Z**2*integ
        if flag == True:
            J_before_CD = np.zeros(loc)
            J_after_CD = J_temp
            return np.concatenate((J_before_CD,J_after_CD))
        else:
            return J_temp
    
    #End of extra functions.
    #========================================================================================================


    def heating_Ecomp(self,Z,xe,Tk):
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
        
        Returns
        -------    
        
        float
            Compton heating. Units kelvin.

        '''        
        return (8*sigT*aS)/(3*me*cE)*self.basic_cosmo_Tcmb(Z)**4*xe*(self.basic_cosmo_Tcmb(Z)-Tk)/(self.basic_cosmo_H(Z)*(1+self.basic_cosmo_xHe()+xe))


    def heating_Elya(self,Z,xe,Tk):
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
        
        Returns
        -------    
        
        float
            Net heating by the Lyman series photons. Units kelvin.
        '''
            
        eta = self._recoil(Tk)
        Scat = self._scatter_corr(Z,xe,Tk)
        atau = self._a_tau(Z,xe,Tk)
        arr = scsp.airy(-self._xi2(Z,xe,Tk))
        
        Ic = eta*(2*np.pi**4*atau**2)**(1/3)*(arr[0]**2+arr[2]**2)
        Ii = eta*np.sqrt(atau/2)*scint.quad(lambda y:y**(-1/2)*np.exp(-2*eta*y-np.pi*y**3/(6*atau))*scsp.erfc(np.sqrt(np.pi*y**3/(2*atau))),0,np.inf)[0]-Scat*(1-Scat)/(2*eta)
        J_alp = self._lya_spec_inten(Z)
        nbary = (1+self.basic_cosmo_xHe())*self.basic_cosmo_nH(Z)
        return 8*np.pi/3 * hP/(kB*lam_alpha) * J_alp*self._dopp(Tk)/nbary * (Ic+Ji_to_Jc*Ii)
       
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

    def heating_Ex(self,Z,xe):
        '''
        See Eq.(11) from `Furlanetto (2006) <https://academic.oup.com/mnras/article/371/2/867/1033021>`__
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Returns
        -------    
        
        float
            Net heating by the X-ray photons. Units kelvin.
            
        '''
        def fXh(xe):
            return 1-(1-xe**0.2663)**1.3163
        
        return 5e5*self.fX*fstar*fXh(xe)*Z*np.abs(self.hmf_dfcoll_dz(Z))

    #End of functions related to heating.
    #========================================================================================================

    def reion_clump(self,Z):
        '''
        Clumping factor for the ionisation of hydrogen. From `Shull et al. (2012) <https://iopscience.iop.org/article/10.1088/0004-637X/747/2/100>`__.
        '''
        return 20.81*Z**-1.1

    def reion_tau(self,Z,Q):
        '''
        Compute the Thomson-scattering optical depth up to a 1+redshift=Z.

        Arguments
        ---------
        Z : float
            The 1+redshift to which you want to calculate :math:`\\tau_{\\mathrm{e}}}`.
        
        Q : float
            The volume-filling factor. This should be the solution for default redshift range. Saved as ``Q_default``.

        Returns
        -------

        float
            :math:`\\tau_{\\mathrm{e}}}`

        '''
        idx1 = np.where(Q>=0.98)[0][0]
        Zreion = Z_default[idx1]

        if type(Z) == int or type(Z)==float:
            idx2 = np.argmin(np.abs(Z_default-Z))
            if Z>Zreion:
                Z_int = Z_default[idx2:idx1][::-1]
                Q_int = Q[idx2:idx1][::-1]
                tau1 = cE*sigT*self.basic_cosmo_nH(1)*np.trapezoid(Q_int*(1+self.basic_cosmo_xHe())*Z_int**2/self.basic_cosmo_H(Z_int),Z_int)
                tau2 = Mpc2km*(cE*sigT/self.Ho)*(2/3*1/self.Om_m)*self.basic_cosmo_nH(1)*(1+self.basic_cosmo_xHe())*(np.sqrt(1-self.Om_m+self.Om_m*Zreion**3)-1)
                tau = tau1 + tau2
            else:
                tau = Mpc2km*(cE*sigT/self.Ho)*(2/3*1/self.Om_m)*self.basic_cosmo_nH(1)*(1+self.basic_cosmo_xHe())*(np.sqrt(1-self.Om_m+self.Om_m*Z**3)-1)
            return tau
        elif type(Z)==np.ndarray or type(Z)==list:
            i = 0
            numofZ = len(Z)
            tau=np.zeros(numofZ)
            for X in Z:
                idx2 = np.argmin(np.abs(Z_default-X))
                if X>Zreion:
                    Z_int = Z_default[idx2:idx1][::-1]
                    Q_int = Q[idx2:idx1][::-1]
                    tau1 = cE*sigT*self.basic_cosmo_nH(1)*np.trapezoid(Q_int*(1+self.basic_cosmo_xHe())*Z_int**2/self.basic_cosmo_H(Z_int),Z_int)
                    tau2 = Mpc2km*(cE*sigT/self.Ho)*(2/3*1/self.Om_m)*self.basic_cosmo_nH(1)*(1+self.basic_cosmo_xHe())*(np.sqrt(1-self.Om_m+self.Om_m*Zreion**3)-1)
                    tau[i] = tau1 + tau2
                else:
                    tau[i] = Mpc2km*(cE*sigT/self.Ho)*(2/3*1/self.Om_m)*self.basic_cosmo_nH(1)*(1+self.basic_cosmo_xHe())*(np.sqrt(1-self.Om_m+self.Om_m*X**3)-1)
                i=i+1
            return tau
    #End of functions related to reionization.
    #========================================================================================================
    
    def history_eqns(self, Z,V):
        '''
        This function has the differential equations governing the ionisation and thermal history.
        When solving upto the end of dark ages, only cosmological parameters will be used.
        Beyond Zstar, i.e., beginning of cosmic dawn astrophysical will also be used.
        '''
        xe = V[0]
        QHII = V[1]
        Tk = V[2]

        #eq1 is (1+z)d(xe)/dz; see Weinberg's Cosmology book or eq.(71) from Seager et al (2000), ApJSS
        eq1 = 1/self.basic_cosmo_H(Z)*self.recomb_Peebles_C(Z,xe,Tk)*(xe**2*self.basic_cosmo_nH(Z)*self.recomb_alpha(Tk)-self.recomb_beta(Tk)*(1-xe)*np.exp(-Ea/(kB*Tk)))

        #eq2 is (1+z)dQ/dz; Pritchard & Furlanetto (2007) eq.(11)
        #eq3 is (1+z)dT/dz; see eq.(2.31) from Mittal et al (2022), JCAP
        
        if Z>Zstar:
            eq2 = 0
            eq3 = 2*Tk-Tk*eq1/(1+self.basic_cosmo_xHe()+xe)-self.heating_Ecomp(Z,xe,Tk)
        else:
            if QHII<0.99:
                eq2 = (1/(1-self.Yp))*fstar*self.fesc*Nion*Z*self.hmf_dfcoll_dz(Z) + (1+self.basic_cosmo_xHe())*1/self.basic_cosmo_H(Z)*(alpha_B*self.reion_clump(Z)*self.basic_cosmo_nH(Z)*QHII)
            else:
                eq2 = 0
            eq3 = 2*Tk-Tk*eq1/(1+self.basic_cosmo_xHe()+xe)-self.heating_Ecomp(Z,xe,Tk)-self.heating_Ex(Z,xe)-self.heating_Elya(Z,xe,Tk)
        
        return np.array([eq1,eq2,eq3])

    def history_solver(self, Z_eval, xe_init = None, Tk_init = None):

        #Assuming Z_eval is in decreasing order.
        Z_start = Z_eval[0]
        Z_end = Z_eval[-1]

        if Z_start == 1501:
            Tk_init = self.basic_cosmo_Tcmb(Z_start)
            xe_init = self.recomb_Saha_xe(Z_start,Tk_init)
            
        Sol = scint.solve_ivp(lambda a, Var: -self.history_eqns(1/a,Var)/a, [1/Z_start, 1/Z_end],[xe_init,0,Tk_init],method='Radau',t_eval=1/Z_eval) 
        
        #Obtaining the solutions ...
        xe=Sol.y[0]
        QHII=Sol.y[1]
        Tk=Sol.y[2]

        return [xe,QHII,Tk]

    #End of functions related to history equations.
    #========================================================================================================

    def hyfi_kHH(self,Tk):
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

    def hyfi_keH(self,Tk):
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


    def hyfi_kpH(self,Tk):
        '''
        Volumetric spin flip rate for electron-proton collision. Fit taken from `Mittal et al. (2022) <https://iopscience.iop.org/article/10.1088/1475-7516/2022/03/030>`__.
        
        Arguments
        ---------
        
        Tk : float
            Gas kinetic temperature.
            
        Returns
        -------
        
        float
            :math:`k_{\\mathrm{pH}}` in units of :math:`\\mathrm{m^3s^{-1}}`.
        '''
        lnT=np.log(Tk)
        return (4.28+0.1023*lnT-0.2586*lnT**2+0.04321*lnT**3)*1e-16
    
    def hyfi_col_coup(self,Z,xe,Tk):
        '''
        Collisional coupling.
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature.
        
        Returns
        -------
        
        float 
            :math:`x_{\\mathrm{k}}`, dimensionless.
        '''

        return Tstar*self.basic_cosmo_nH(Z)*((1-xe)*self.hyfi_kHH(Tk)+xe*self.hyfi_keH(Tk)+xe*self.hyfi_kpH(Tk))/(A10*self.basic_cosmo_Tcmb(Z))

    def hyfi_lya_coup(self,Z,xe,Tk):
        '''
        Ly:math:`\\alpha` coupling or the Wouthuysen--Field coupling.
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature.
        
        Returns
        -------
        
        float
            :math:`x_{\\alpha}`, dimensionless.
        '''
    
        Scat = self._scatter_corr(Z,xe,Tk)
        J_alp = self._lya_spec_inten(Z)    #'undistorted' background Spec. Inte. of Lya photons.
        Jo = 5.54e-8*Z         #eq.(24) in Mittal & Kulkarni (2021)
        return Scat*J_alp/Jo

    def hyfi_spin_temp(self,Z,xe,Tk):
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
        
        Returns
        -------
        
        float
            :math:`T_{\\mathrm{s}}`, K.
        '''

        xa = self.hyfi_lya_coup(Z,xe,Tk)
        xk = self.hyfi_col_coup(Z,xe,Tk)
        Ts = ( 1  + xa + xk*Tk/(Tk+T_se))/(1/self.basic_cosmo_Tcmb(Z) +  xk/Tk + xa/(Tk+T_se) )
        return Ts

    def hyfi_twentyone_cm(self,Z,xe, Q,Tk):
        '''
        The global (sky-averaged) 21-cm signal.
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.

        Q : float
            Volume-filling factor.
        
        Tk : float
            Gas kinetic temperature.
        
        Returns
        -------
        
        float
            :math:`T_{21}`, mK.
        '''
        #Get the two-zone model averaged ionisation fraction.
        xHI = 1-_gaif(xe,Q)
        Ts = self.hyfi_spin_temp(Z,xe,Tk)
        return 27*xHI*((1-self.Yp)/0.76)*(self.Om_b*self.h100**2/0.023)*np.sqrt(0.15*Z/(10*self.Om_m*self.h100**2))*(1-self.basic_cosmo_Tcmb(Z)/Ts)

#End of class main.
#========================================================================================================
#========================================================================================================


class pipeline():
    '''
    This class runs the cosmic history solver and produces the global signal and the corresponding redshifts.
    
    Methods
    ~~~~~~~
    '''
    def __init__(self,cosmo={'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'Tcmbo':2.725,'Yp':0.245},astro= {'falp':1,'fX':0.1,'fesc':0.1,'Tmin_vir':1e4},Z_eval=None,path='', hmf_name='press74'):

        self.comm = MPI.COMM_WORLD
        self.cpu_ind = self.comm.Get_rank()
        self.n_cpu = self.comm.Get_size()  

        self.cosmo=cosmo
        self.astro=astro

        self.model = 0
        for keys in self.astro.keys():
            if np.size(self.astro[keys])>1:
                self.model = self.model+1
                break
                
        for keys in self.cosmo.keys():
            if np.size(self.cosmo[keys])>1:
                self.model = self.model+2
                break
        
        if self.model==0:
            self.astro=_to_float(self.astro)
            self.cosmo=_to_float(self.cosmo)
        elif self.model==1:
            self.astro=_to_array(self.astro)
            self.cosmo=_to_float(self.cosmo)
        elif self.model==3:
            self.astro=_to_array(self.astro)
            self.cosmo=_to_array(self.cosmo)
        else:
            print('Currently not designed to work with varying cosmological parameters only!')
            sys.exit()
        
        self.Z_eval = Z_eval

        if type(self.Z_eval)==np.ndarray or type(self.Z_eval)==list:
            self.Z_eval=np.array(self.Z_eval)
            if self.Z_eval[1]>self.Z_eval[0]:
                # Arranging redshifts from ascending to descending
                self.Z_eval = self.Z_eval[::-1]

        self.Ho = cosmo['Ho']
        self.Om_m = cosmo['Om_m']
        self.Om_b = cosmo['Om_b']
        self.Tcmbo = cosmo['Tcmbo']
        self.Yp = cosmo['Yp']
        
        self.falp = astro['falp']
        self.fX = astro['fX']
        self.fesc = astro['fesc']
        self.Tmin_vir = astro['Tmin_vir']
        
        self.hmf_name = hmf_name

        self.path=path
        if self.cpu_ind==0:
            if os.path.isdir(self.path)==False:
                print('The requested directory does not exist. Creating ',self.path)
                os.mkdir(self.path)
            
            self.timestamp = strftime("%Y%m%d%H%M%S", localtime())
            self.path = self.path + 'output_'+self.timestamp+'/'
            os.mkdir(self.path)

            self.formatted_timestamp = self.timestamp[8:10]+':'+self.timestamp[10:12]+':'+self.timestamp[12:14]+' '+self.timestamp[6:8]+'/'+self.timestamp[4:6]+'/'+ self.timestamp[:4]
        return None

    def glob_sig(self):      
                    
        if self.model==0:
        #Cosmological and astrophysical parameters are fixed.
            if self.cpu_ind==0:
                _print_banner()
                print('Both cosmological and astrophysical parameters are fixed.\n')
                
                st = time.process_time()
                
                myobj = main(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,Tcmbo=self.Tcmbo,Yp=self.Yp,falp=self.falp,fX=self.fX,fesc=self.fesc,Tmin_vir=self.Tmin_vir, hmf_name=self.hmf_name)

                Z_temp = Z_default

                if self.Z_eval is not None:
                    if (self.Z_eval[0]>1501 or self.Z_eval[-1]<Z_end):
                        print('\033[31mYour requested redshift values should satisfy ',1501,'>1+z>',Z_end)
                        print('Terminating ...\033[00m')
                        sys.exit()
                    else:
                        Z_temp = self.Z_eval
                
                print('Obtaining the thermal and ionisation history ...')
                sol = myobj.history_solver(Z_eval=Z_default)
                
                xe = sol[0]
                Q_Hii = sol[1]
                Tk = sol[2]
                
                Q_Hii_default = Q_Hii  #We need this for computing CMB optical depth

                if self.Z_eval is not None:
                    splxe = CubicSpline(np.flip(Z_default), np.flip(xe))
                    xe = splxe(self.Z_eval)
                    Q_Hii = np.interp(self.Z_eval, np.flip(Z_default), np.flip(Q_Hii))
                    splTk = CubicSpline(np.flip(Z_default), np.flip(Tk))
                    Tk = splTk(self.Z_eval)

                print('Obtaining spin temperature ...')
                Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)

                print('Computing the 21-cm signal ...')
                T21_mod1 = myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Tk=Tk)
                
                print('Done.')

                xe_save_name = self.path+'xe'
                Q_save_name = self.path+'Q'
                Q_default_save_name = self.path+'Q_default'
                Tk_save_name = self.path+'Tk'
                Ts_save_name = self.path+'Ts'
                Tcmb_save_name = self.path+'Tcmb'
                T21_save_name = self.path+'T21'
                z_save_name = self.path+'one_plus_z'
                
                np.save(xe_save_name,xe)
                np.save(Q_save_name,Q_Hii)
                np.save(Q_default_save_name,Q_Hii_default)
                np.save(Tk_save_name,Tk)
                np.save(Ts_save_name,Ts)
                np.save(Tcmb_save_name,myobj.basic_cosmo_Tcmb(Z_temp))
                np.save(T21_save_name,T21_mod1)
                np.save(z_save_name,Z_temp)
                
                print('\033[32mYour outputs have been saved into folder:',self.path,'\033[00m')
                
                et = time.process_time()
                # get the execution time
                elapsed_time = et - st
                print('\nExecution time: %.2f seconds' %elapsed_time)

                #========================================================
                #Writing to a summary file
                max_T21 = np.min(T21_mod1)
                max_ind = np.where(T21_mod1==max_T21)
                [max_z] = Z_temp[max_ind]

                idx = np.argmin(np.abs(Q_Hii_default-0.5))
                z50 = Z_default[idx]-1
                z100 = None
                try:
                    idx = np.where(Q_Hii_default>=0.98)[0][0]
                    z100 = Z_default[idx]-1
                    tau_e = myobj.reion_tau(60,Q_Hii_default)
                except:
                    print('\nReionisation did not complete by z=5.')
                
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
                myfile.write('-----------------')
                myfile.write('\nHo = {}'.format(self.Ho))
                myfile.write('\nOm_m = {}'.format(self.Om_m))
                myfile.write('\nOm_b = {}'.format(self.Om_b))
                myfile.write('\nTcmbo = {}'.format(self.Tcmbo))
                myfile.write('\nYp = {}'.format(self.Yp))
                myfile.write('\n\nfalp = {}'.format(self.falp))
                myfile.write('\nfX = {}'.format(self.fX))
                myfile.write('\nfesc = {}'.format(self.fesc))
                myfile.write('\nmin(T_vir) = {}'.format(self.Tmin_vir))
                myfile.write('\n')
                myfile.write('\n50% reionisation complete at z = {:.2f}'.format(z50))
                if z100!=None:
                    myfile.write("\nReionisation complete at z = {:.2f}".format(z100))
                    myfile.write("\nTotal Thomson-scattering optical depth = {:.4f}".format(tau_e))

                myfile.write('\n\nStrongest 21-cm signal is {:.2f} mK, observed at z = {:.2f}'.format(max_T21,max_z-1))
                myfile.write('\n')
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
                return None
            
        elif self.model==1:
        #Cosmological parameters are fixed so dark ages is solved only once.
            if self.cpu_ind==0:
                _print_banner()
                print('Cosmological parameters are fixed. Astrophysical parameters are varied.')
                print('\nGenerating once the thermal and ionisation history for dark ages ...')
            
            myobj_da = main(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,Tcmbo=self.Tcmbo,Yp=self.Yp,falp=self.falp[0],fX=self.fX[0],fesc=self.fesc[0],Tmin_vir=self.Tmin_vir[0], hmf_name=self.hmf_name)

            Z_da = np.linspace(1501,Zstar,1400)
            sol_da = myobj_da.history_solver(Z_eval=Z_da)
            xe_da = sol_da[0]
            Tk_da = sol_da[2]

            Z_cd = np.linspace(Zstar,Z_end,100)
            Z_temp = Z_cd
            if self.Z_eval is not None:
                if (self.Z_eval[0]>Zstar or self.Z_eval[-1]<Z_end):
                    print('\033[31mYour requested redshift values should satisfy ',Zstar,'>1+z>',Z_end)
                    print('Terminating ...\033[00m')
                    sys.exit()
                else:
                    Z_temp = self.Z_eval

            n_values = len(Z_temp)
            
            n_mod = _no_of_mdls(self.astro)
            arr = np.arange(n_mod)
            arr = np.reshape(arr,[np.size(self.falp),np.size(self.fX),np.size(self.fesc),np.size(self.Tmin_vir)])
            T21_cd = np.zeros((np.size(self.falp),np.size(self.fX),np.size(self.fesc),np.size(self.Tmin_vir),n_values))
            
            if self.cpu_ind==0: print('Done.\n\nGenerating',n_mod,'models ...\n')

            st = time.process_time()
            for i in range(n_mod):
                if (self.cpu_ind == int(i/int(n_mod/self.n_cpu))%self.n_cpu):
                    ind=np.where(arr==i)

                    myobj_cd = main(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,Tcmbo=self.Tcmbo,Yp=self.Yp,falp=self.falp[ind[0][0]],fX=self.fX[ind[1][0]],fesc=self.fesc[ind[2][0]],Tmin_vir=self.Tmin_vir[ind[3][0]], hmf_name=self.hmf_name)
                    sol_cd = myobj_cd.history_solver(Z_eval=Z_cd,xe_init=xe_da[-1],Tk_init=Tk_da[-1])
                    
                    xe_cd = sol_cd[0]
                    Q_cd = sol_cd[1]
                    Tk_cd = sol_cd[2]

                    if np.any(self.Z_eval)!= None:
                        splxe = CubicSpline(np.flip(Z_cd), np.flip(xe_cd))
                        xe_cd = splxe(self.Z_eval)
                        Q_cd = np.interp(self.Z_eval, np.flip(Z_cd), np.flip(Q_cd))
                        splTk = CubicSpline(np.flip(Z_cd), np.flip(Tk_cd))
                        Tk_cd = splTk(self.Z_eval)

                    T21_cd[ind[0][0],ind[1][0],ind[2][0],ind[3][0],:]= myobj_cd.hyfi_twentyone_cm(Z=Z_temp,xe=xe_cd,Q=Q_cd,Tk=Tk_cd)
            
            self.comm.Barrier()
            if self.cpu_ind!=0:
                self.comm.send(T21_cd, dest=0)
            else:
                print('Done.')
                for j in range(1,self.n_cpu):
                    T21_cd = T21_cd + self.comm.recv(source=j)
                
                T21_save_name = self.path+'T21_'+str(np.size(self.falp))+str(np.size(self.fX))+str(np.size(self.fesc))+str(np.size(self.Tmin_vir))
                z_save_name = self.path+'one_plus_z'
                
                np.save(T21_save_name,T21_cd)
                np.save(z_save_name,Z_temp)
                print('\033[32m\nOutput saved into folder:',self.path,'\033[00m')
                
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
                myfile.write('-----------------')
                myfile.write('\nHo = {}'.format(self.Ho))
                myfile.write('\nOm_m = {}'.format(self.Om_m))
                myfile.write('\nOm_b = {}'.format(self.Om_b))
                myfile.write('\nTcmbo = {}'.format(self.Tcmbo))
                myfile.write('\nYp = {}'.format(self.Yp))
                myfile.write('\n\nfalp = {}'.format(self.falp))
                myfile.write('\nfX = {}'.format(self.fX))
                myfile.write('\nfesc = {}'.format(self.fesc))
                myfile.write('\nmin(T_vir) = {}'.format(self.Tmin_vir))
                myfile.write('\n\n{} models generated'.format(n_mod))
                myfile.write('\nNumber of CPU(s) = \n{}'.format(self.n_cpu))
                myfile.write('\n')
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')

        elif self.model==3:

            if self.cpu_ind==0:
                _print_banner()
                print('Both cosmological and astrophysical parameters are varied.')
            

            Z_temp = Z_default
            if self.Z_eval is not None:
                if (self.Z_eval[0]>1501 or self.Z_eval[-1]<Z_end):
                    print('\033[31mYour requested redshift values should satisfy ',1501,'>1+z>',Z_end)
                    print('Terminating ...\033[00m')
                    sys.exit()
                else:
                    Z_temp = self.Z_eval

            n_values = len(Z_temp)
            
            n_mod = _no_of_mdls(self.astro)*_no_of_mdls(self.cosmo)
            arr = np.arange(n_mod)
            arr = np.reshape(arr,[np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.falp),np.size(self.fX),np.size(self.fesc),np.size(self.Tmin_vir)])
            T21_mod3 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.falp),np.size(self.fX),np.size(self.fesc),np.size(self.Tmin_vir),n_values))
            
            if self.cpu_ind==0: print('\nGenerating',n_mod,'models ...')
            st = time.process_time()
            
            for i in range(n_mod):
                if (self.cpu_ind == int(i/int(n_mod/self.n_cpu))%self.n_cpu):
                    ind=np.where(arr==i)

                    myobj = main(Ho=self.Ho[ind[0][0]],Om_m=self.Om_m[ind[1][0]],Om_b=self.Om_b[ind[2][0]],Tcmbo=self.Tcmbo[ind[3][0]],Yp=self.Yp[ind[4][0]],falp=self.falp[ind[5][0]],fX=self.fX[ind[6][0]],fesc=self.fesc[ind[7][0]],Tmin_vir=self.Tmin_vir[ind[8][0]], hmf_name=self.hmf_name)
                    sol = myobj.history_solver(Z_eval=Z_default)

                    xe = sol[0]
                    Q_Hii = sol[1]
                    Tk = sol[2]

                    if np.any(self.Z_eval)!= None:
                        splxe = CubicSpline(np.flip(Z_default), np.flip(xe))
                        xe = splxe(self.Z_eval)
                        Q_Hii = np.interp(self.Z_eval, np.flip(Z_default), np.flip(Q_Hii))
                        splTk = CubicSpline(np.flip(Z_default), np.flip(Tk))
                        Tk = splTk(self.Z_eval)

                    T21_mod3[ind[0][0],ind[1][0],ind[2][0],ind[3][0],[ind[4][0]],[ind[5][0]],[ind[6][0]],[ind[7][0]],[ind[8][0]],:] = myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Tk=Tk)
            
            self.comm.Barrier()
            if self.cpu_ind!=0:
                self.comm.send(T21_mod3, dest=0)
            else:
                print('Done.\n')
                for j in range(1,self.n_cpu):
                    T21_mod3 = T21_mod3 + self.comm.recv(source=j)
                
                save_name = self.path+'T21_'+str(np.size(self.Ho))+str(np.size(self.Om_m))+str(np.size(self.Om_b))+str(np.size(self.Tcmbo))+str(np.size(self.Yp))+str(np.size(self.falp))+str(np.size(self.fX))+str(np.size(self.fesc))+str(np.size(self.Tmin_vir))+'.npy'
                np.save(save_name,T21_mod3)
                
                print('\033[32m\nOutput saved into folder:',self.path,'\033[00m')
                
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
                myfile.write('-----------------')
                myfile.write('\nHo = {}'.format(self.Ho))
                myfile.write('\nOm_m = {}'.format(self.Om_m))
                myfile.write('\nOm_b = {}'.format(self.Om_b))
                myfile.write('\nTcmbo = {}'.format(self.Tcmbo))
                myfile.write('\nYp = {}'.format(self.Yp))
                myfile.write('\n\nfalp = {}'.format(self.falp))
                myfile.write('\nfX = {}'.format(self.fX))
                myfile.write('\nfesc = {}'.format(self.fesc))
                myfile.write('\nmin(T_vir) = {}'.format(self.Tmin_vir))
                myfile.write('\n\n{} models generated'.format(n_mod))
                myfile.write('\nNumber of CPU(s) = \n{}'.format(self.n_cpu))
                myfile.write('\n')
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
        return None
    #End of function glob_sig               

#End of class pipeline
#========================================================================================================
