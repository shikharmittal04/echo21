import scipy.special as scsp
import scipy.integrate as scint
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.lss import mass_function
import warnings

import os
import sys
from .const import *
warnings.filterwarnings('ignore')

def _gaif(xe,Q):
    '''
    Computes the globally-averaged ionisation factor for a two-zone IGM model.
    
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

class funcs():
    '''
    Function names starting with 'basic_cosmo' include the basic :math:`\\Lambda` CDM-cosmology-related functions, such as Hubble function, CMB temperature, etc.

    Function names starting with 'recomb' include recombination-physics-related functions.

    Function names starting with 'heating' include all the heating terms. All the terms are in the form of :math:`-(1+z)\\mathrm{d}T_{\\mathrm{k}}/\\mathrm{d}z` and hence, in units of temperature.
    
    Function names starting with 'hyfi' include all the functions related to the computation of 21-cm signal. These are
    :math:`\\kappa_{\\mathrm{HH}}, \\kappa_{\\mathrm{eH}}, x_{\\mathrm{k}}, x_{\\mathrm{Ly}}, T_{\\mathrm{s}}` and :math:`T_{21}`.

    Methods
    ~~~~~~~
    '''
    def __init__(self,Ho=67.4,Om_m=0.315,Om_b=0.049,sig8=0.811,ns=0.965,Tcmbo=2.725,Yp=0.245,fLy=1.0,sLy=2.64,fX=1,wX=1.5,fesc=0.0106,cosmo=None,astro=None,**kwargs):
        '''
        
        '''
        if cosmo!=None:
            Ho = cosmo['Ho']
            Om_m = cosmo['Om_m']
            Om_b = cosmo['Om_b']
            sig8 = cosmo['sig8']
            ns = cosmo['ns']
            Tcmbo = cosmo['Tcmbo']
            Yp = cosmo['Yp']
        if astro!=None:
            fLy = astro['fLy']
            sLy = astro['sLy']
            fX = astro['fX']
            wX = astro['wX']
            fesc = astro['fesc']
        
        self.Ho = Ho
        self.Om_m = Om_m
        self.Om_b = Om_b
        self.sig8 = sig8
        self.ns = ns
        self.Tcmbo = Tcmbo
        self.Yp = Yp
        
        self.fLy = fLy
        self.sLy = sLy
        self.fX = fX
        self.wX = wX
        self.fesc = fesc
        

        self.sfrd_type = kwargs.pop('type', 'phy')
        self.mdef = kwargs.pop('mdef','fof')
        self.hmf = kwargs.pop('hmf','press74')
        self.Tmin_vir = kwargs.pop('Tmin_vir',1e4)
        self.a_sfrd = kwargs.pop('a',0.257)
        self.b_sfrd = kwargs.pop('b',4)

        self.cosmo_par = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': sig8, 'ns': ns,'relspecies': True,'Tcmb0': Tcmbo}
        self.my_cosmo = cosmology.setCosmology('cosmo_par', self.cosmo_par)
        self.h100 = self.Ho/100

        return None

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
        Hydrogen number density (proper).
        
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
        Hubble factor in SI units.
        
        Arguments
        ---------
        
        Z : float
            1+z
        
        Returns
        -------
        
        float
            Hubble parameter at a given redshift in units of :math:`\\mathrm{s}^{-1}`.
        '''
        Om_lam = 1-self.Om_m
        Om_r = (1+fnu)*aS*self.Tcmbo**4/(cE**2*self.basic_cosmo_rho_crit())
        
        return self.Ho*(Om_r*Z**4+self.Om_m*Z**3+Om_lam)**0.5/Mpc2km

    #End of functions related to basic cosmology.
    #========================================================================================================



    def recomb_alpha(self, T):
        '''
        :math:`\\alpha_{\\mathrm{B}}=\\alpha_{\\mathrm{B}}(T)`
        
        The effective case-B recombination coefficient for hydrogen. See Eq. (70) from `Seager et al (2000) <https://iopscience.iop.org/article/10.1086/313388>`__.
        
        Arguments
        ---------
        
        T : float
            Temperature in units of kelvin.
        
        Returns
        -------
        
        float
            The effective case-B recombination coefficient for hydrogen :math:`(\\mathrm{m}^3\\mathrm{s}^{-1})`.
            
        '''
        t=T/10000
        return (1e-19)*Feff*A_rec*t**b_rec/(1+c_rec*t**d_rec)

    def recomb_beta(self, T):
        '''
        :math:`\\beta=\\beta(T)`
        
        The total photoionization rate. See description below Eq. (71) from `Seager et al (2000) <https://iopscience.iop.org/article/10.1086/313388>`__. Relation between :math:`\\alpha_{\\mathrm{B}}` and :math:`\\beta`:
        
        :math:`\\beta=\\alpha_{\\mathrm{B}}\\left(\\frac{2\\pi m_{\\mathrm{e}}k_{\\mathrm{B}}T}{h_{\\mathrm{P}}^2}\\right)^{3/2}\\exp\\left(-\\frac{B_2}{k_{\\mathrm{B}}T}\\right)`
        
        Arguments
        ---------
        
        T : float
            Temperature in units of kelvin.
        
        Returns
        -------
        
        float
            The total photoionization rate in :math:`(\\mathrm{s}^{-1})`.
            
        '''
        beta = self.recomb_alpha(T)*(2*np.pi*me*kB*T/hP**2)**1.5*np.exp(-B2/(kB*T))
        return beta

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

    def recomb_Peebles_C(self,Z,xe,T):
        '''
        :math:`C_{\\mathrm{P}}`
        
        Arguments
        ---------
        
        Z : float
            1 + redshift, dimensionless
        
        xe : float
            Electron fraction, dimensionless
            
        Tk : float
            Temperature in units of kelvin.
        
        Returns
        -------
        
        float
            Peebles 'C' factor appearing in Eq. (71) from `Seager et al (2000) <https://iopscience.iop.org/article/10.1086/313388>`__, dimensionless.
        '''
        
        return (1+self.recomb_Krr(Z)*Lam_H*self.basic_cosmo_nH(Z)*(1-xe))/(1+self.recomb_Krr(Z)*(Lam_H+self.recomb_beta(T))*self.basic_cosmo_nH(Z)*(1-xe))

    def recomb_Saha_xe(self,Z,T):
        '''
        Electron fraction predicted by the Saha's equation. This is important to initialize the differential equation for :math:`x_{\\mathrm{e}}`. At high redshift such as :math:`z=1500`, Saha's equation gives accurate estimate of :math:`x_{\\mathrm{e}}`.
        
        Arguments
        ---------
        
        Z : float
            1 + redshift, dimensionless
          
        T : float
            Temperature in units of kelvin
        
        Returns
        -------
        
        float
            Electron fraction predicted by Saha's equation. Dimensionless.
        '''
        Saha=1/self.basic_cosmo_nH(Z)*(2*np.pi*me*kB*T/hP**2)**1.5*np.exp(-B1/(kB*T))
        return (np.sqrt(Saha**2+4*Saha)-Saha)/2

    #End of functions related to recombination
    #========================================================================================================
    
    
    def dndlnM(self, M,Z):
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
            HMF, :math:`\\mathrm{d}n/\\mathrm{d\\,ln}M=M\\mathrm{d}n/\\mathrm{d}M`, in units of :math:`\\mathrm{cMpc}^{-3}`, where 'cMpc' represents comoving mega parsec.
        '''
        M_by_h = M*self.h100 #M in units of solar mass/h
        return self.h100**3*mass_function.massFunction(M_by_h, Z-1, q_in='M', q_out='dndlnM', mdef = self.mdef, model = self.hmf)

    def dndM(self,M,Z):
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
            HMF in a different form, :math:`\\mathrm{d}n/\\mathrm{d}M`, in units of :math:`\\mathrm{cMpc}^{-3}\\mathrm{M}_{\\odot}^{-1}`, where 'cMpc' represents comoving mega parsec and :math:`\\mathrm{M}_{\\odot}` represents the solar mass.
        '''

        return 1/M*self.dndlnM(M,Z)

    
    def m_min(self,Z):
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

    def f_coll(self,Z):
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

        def rho_dm_coll(Z):
            #DM collapsed as haloes (in kg/m^3, comoving)
            M_space = np.logspace(np.log10(self.m_min(Z)/self.h100),18,1500)    #These masses are in solar mass. Strictly speaking we should integrate up to infinity but for numerical purposes 10^18.Msun is sufficient.
            hmf_space = self.dndlnM(M=M_space,Z=Z)    #Corresponding HMF values are in cMpc^-3 
            return Msolar_by_Mpc3_to_kg_by_m3*np.trapezoid(hmf_space,M_space)
        
        if self.hmf=='press74':
            return scsp.erfc(peaks.peakHeight(self.m_min(Z),Z-1)/np.sqrt(2))
        else:
            numofZ = np.size(Z)
                
            if numofZ == 1:
                if type(Z)==np.ndarray: Z=Z[0]
                rho_halo = rho_dm_coll(Z)
            else:    
                rho_halo = np.zeros(numofZ)
                counter=0
                for i in Z:
                    rho_halo[counter]=rho_dm_coll(i)
                    counter=counter+1
            return rho_halo/(self.Om_m*self.basic_cosmo_rho_crit())

    def dfcoll_dz(self,Z):
        '''
        Redshift derivative of the collapse fraction, i.e., :math:`\\mathrm{d}F_{\\mathrm{coll}}/\\mathrm{d}z`
        '''
        return (self.f_coll(Z+1e-3)-self.f_coll(Z))*1e3

     
    def sfrd(self,Z):
        '''
        This function returns the comoving star formation rate density (SFRD, :math:`\\dot{\\rho}_{\\star}`).
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless. Can be a single quantity or an array.
        
        Returns
        -------
        
        float 
            Comoving SFRD in units of :math:`\\mathrm{kgs^{-1}m^{-3}}`. Single number or an array accordingly as ``Z`` is single number or an array.
        '''
        def _sfrd(Z):
            if Z<1+self.b_sfrd: return 0.015*Z**2.73/(1+(Z/3)**6.2)
            else: return 0.015*(1+self.b_sfrd)**2.73/(1+((1+self.b_sfrd)/3)**6.2)*10**(self.a_sfrd*(1+self.b_sfrd-Z))
        
        if self.sfrd_type=='phy':
            mysfrd = -Z*fstar*self.Om_b*self.basic_cosmo_rho_crit()*self.dfcoll_dz(Z)*self.basic_cosmo_H(Z)
        elif self.sfrd_type=='emp':
            NumZ = np.size(Z)
            
            if NumZ>1:
                mysfrd = np.zeros(NumZ)
                for i in range(NumZ):
                    mysfrd[i] = _sfrd(Z[i])
            else: mysfrd = _sfrd(Z)

            mysfrd = mysfrd*Msolar_by_Mpc3_year_to_kg_by_m3_sec

        return mysfrd

    #========================================================================================================

    def _fXh(self,xe):
        return 1-(1-xe**0.2663)**1.3163

    def _fXion(self,xe):
        return 0.3908*(1-xe**0.4092)**1.7592
    
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
        Returns the product :math:`a\\tau`, since all the relevant formulae require the product only.
        :math:`a` is the Voigt parameter and :math:`\\tau` is the optical depth of Lya photons.
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
    
    def phi_Ly(self,E):
        '''
        Spectral energy distribution (SED) of Lyman series photons in units of number of photons per unit frequency per stellar baryon.
        
        Arguments
        ---------
        
        E : float
            Energy in eV.
        
        Returns
        -------

        float
            SED in dimensions :math:`\\mathrm{Hz^{-1}}`. 
        '''
        if self.sLy!=0:
            return self.fLy*hP/eC*1/13.6*self.sLy*N_alpha_infty/(1.33**self.sLy-1)*(E/13.6)**(-self.sLy-1)
        else:
            return self.fLy*hP/eC*N_alpha_infty/np.log(4/3)*E**-1

    def eps_Ly(self,Z,E):
        '''
        Emissivity of Lyman series photons in units of number of photons per unit frequency per unit comoving volume per unit time. Construction:
        
        :math:`\\epsilon_{\\mathrm{Ly}}=\\frac{1}{m_{\\mathrm{b}}}\\phi_{\\mathrm{Ly}}\\dot{\\rho}_{\\star}`

        Arguments
        ---------
        Z : float
            1+z
        
        E : float
            Energy in eV.
        
        Returns
        -------

        float
            Emissivity in dimensions :math:`\\mathrm{m^{-3}Hz^{-1}s^{-1}}`. 

        '''
        return 1/(1.22*mP)*self.phi_Ly(E)*self.sfrd(Z)
    
    def lya_spec_inten(self,Z):
        '''
        Specific intensity of Ly :math:`\\alpha` photons, :math:`J_{\\mathrm{Ly}}`, due to continuum and injected photons.
        
        Arguments
        ---------
        Z : float
            1 + z, dimensionless. Can be array.
        
        Returns
        -------
        
        float
            Specific intensity in terms of number per unit time per unit area per unit frequency per unit solid angle (:math:`\\mathrm{m^{-2}s^{-1}Hz^{-1}sr^{-1}}`). Two values are returned, namely intensity due to continuum and injected photons, respectively.
        '''
        def _lya_spec_inten(Z):
            prefac = cE/(4*np.pi)*Z**2

            Zmax = 32/27*Z
            temp = np.linspace(Z,Zmax,10)
            continuum = scint.trapezoid(self.eps_Ly(temp,10.2*temp/Z)/self.basic_cosmo_H(temp),temp)

            injected = 0
            for ni in np.arange(4,24):
                Zmax = (1-1/(ni+1)**2)/(1-1/ni**2)*Z
                temp = np.linspace(Z,Zmax,5)
                injected = injected+Pn[ni-4]*scint.trapezoid(self.eps_Ly(temp,13.6*(1-1/ni**2)*temp/Z)/self.basic_cosmo_H(temp),temp)
            return prefac*continuum,prefac*injected

        if type(Z)==np.float64 or type(Z)==float or type(Z)==int:
            if Z>Zstar:
                return 0.0,0.0
            else:
                return _lya_spec_inten(Z)

        
        elif type(Z)==np.ndarray or type(Z)==list:
            loc=0
            flag=False
            #First check if the provided Z is in descending order or not.
            if Z[1]>Z[0]:
                # Arranging redshifts from ascending to descending
                Z = Z[::-1]

            if Z[0]>Zstar:
                flag=True
                loc = np.where(Z<Zstar)[0][0]
                Z=Z[loc:]
            
            counter=0
            numofZ = len(Z)
            J_temp=np.zeros((numofZ,2))
            for Z_value in Z:
                J_temp[counter,:]=_lya_spec_inten(Z_value)
                counter=counter+1
        
            if flag == True:
                J_before_CD = np.zeros((loc,2))
                J_after_CD = J_temp
                return np.concatenate((J_before_CD,J_after_CD))
            else:
                return J_temp
    
    #End of functions related to Lyman-alpha photons.
    #========================================================================================================



    def heating_Ecomp(self,Z,xe,Tk):
        '''
        See Eq.(2.32) from Mittal et al (2022), JCAP.
        (However, there is a typo in that equation; numerator has an :math:`x_{\\mathrm{e}}` missing.)
        
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
        compterm = (8*sigT*aS)/(3*me*cE)*self.basic_cosmo_Tcmb(Z)**4*xe*(self.basic_cosmo_Tcmb(Z)-Tk)/(self.basic_cosmo_H(Z)*(1+self.basic_cosmo_xHe()+xe))
        return compterm


    def heating_Elya(self,Z,xe,Tk):
        '''
        Ly :math:`\\alpha` heating rate. For details see `Mittal & Kulkarni (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.4264M/abstract>`__
        
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
        Jc,Ji = self.lya_spec_inten(Z)
        nbary = (1+self.basic_cosmo_xHe()+xe)*self.basic_cosmo_nH(Z)
        return 8*np.pi/3 * hP/(kB*lam_alpha) * self._dopp(Tk)/nbary * (Jc*Ic+Ji*Ii)

    def heating_Ex(self,Z,xe):
        '''
        We use the parametric approach for X-ray heating as in `Furlanetto (2006) <https://academic.oup.com/mnras/article/371/2/867/1033021>`__. However, our normalisation is smaller by a factor of 0.14 as we adopt the :math:`L_{\\mathrm{X}}/\\mathrm{SFR}` relation from `Lehmer et al. (2024) <https://iopscience.iop.org/article/10.3847/1538-4357/ad8de7>`__.
        
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

        if self.wX!=1: CX_modifier=(tilda_E1**(1-self.wX)-tilda_E0**(1-self.wX))/(E1**(1-self.wX)-E0**(1-self.wX))
        else: CX_modifier= np.log(tilda_E1/tilda_E0)/np.log(E1/E0)
        prefactor = 2/(3*self.basic_cosmo_nH(Z)*(1+self.basic_cosmo_xHe()+xe)*kB*self.basic_cosmo_H(Z))
        return prefactor*self.fX*self._fXh(xe)*self.sfrd(Z)*CX_fid*CX_modifier

    #End of functions related to heating.
    #========================================================================================================
    def Gamma_x(self,Z,xe):
        '''
        Ionization (of bulk IGM) rate due to X-ray photons.
        
        Z : float
            1 + z, dimensionless.
        
        xe : float
            Electron fraction.
        
        Returns
        -------    
        
        float
            Ionization due to X-ray photons in units of :math:`\\mathrm{s}^{-1}`.
        '''
        prefactor = 2/(3*self.basic_cosmo_nH(Z)*(1+self.basic_cosmo_xHe()+xe)*kB*self.basic_cosmo_H(Z))
        qX = self.heating_Ex(Z,xe)/prefactor
        HX = qX/(self._fXh(xe)*(1-xe)*self.basic_cosmo_nH(Z))
        Ew = 1e3*((tilda_E1**(-self.wX-2.4)-tilda_E0**(-self.wX-2.4))/(tilda_E1**(-self.wX-3.4)-tilda_E0**(-self.wX-3.4)))*(self.wX+3.4)/(self.wX+2.4)-13.6
        secondary_ionization = self._fXion(xe)/13.6
        ionization_rate = HX*(1/Ew+secondary_ionization)/eC
        return ionization_rate

    #========================================================================================================
    def reion_clump(self,Z):
        '''
        Clumping factor for the ionization of hydrogen. From `Shull et al. (2012) <https://iopscience.iop.org/article/10.1088/0004-637X/747/2/100>`__.
        '''
        return 20.81*Z**-1.1

    def reion_tau(self,Z,Q):
        '''
        Compute the Thomson-scattering optical depth up to a 1+redshift=Z.

        Arguments
        ---------
        Z : float
            1+z to which you want to calculate :math:`\\tau_{\\mathrm{e}}`.
        
        Q : float
            The volume-filling factor. This should be the solution for default redshift range; in the output folder it would be saved as ``Q_default``.

        Returns
        -------

        float
            :math:`\\tau_{\\mathrm{e}}`

        '''
        prefac = cE*sigT*self.basic_cosmo_nH(1)
        xHe = self.basic_cosmo_xHe()

        def _Ez(Z):
            return np.sqrt(1-self.Om_m+self.Om_m*Z**3)

        def _reion(Z):
            if Z>Zreion:
                idx2 = np.argmin(np.abs(Z_default-Z))
                Z_int = Z_default[idx2:idx1][::-1]
                Q_int = Q[idx2:idx1][::-1]
                tau1 = prefac*(1+xHe)*np.trapezoid(Q_int*Z_int**2/self.basic_cosmo_H(Z_int),Z_int)
                tau2 = prefac*(Mpc2km/self.Ho)*(2/3*1/self.Om_m)*((1+2*xHe)*(_Ez(5)-1)+(1+xHe)*(_Ez(Zreion)-_Ez(5)))
                tau = tau1 + tau2
            else:
                if Z>5:
                    tau = prefac*(Mpc2km/self.Ho)*(2/3*1/self.Om_m)*((1+2*xHe)*(_Ez(5)-1)+(1+xHe)*(_Ez(Z)-_Ez(5)))
                else:
                    tau = prefac*(Mpc2km/self.Ho)*(2/3*1/self.Om_m)*(1+2*xHe)*(_Ez(Z)-1)

            return tau
        
        idx1 = np.where(Q>=0.98)[0][0]
        Zreion = Z_default[idx1]
        
        
        if type(Z) == int or type(Z)==float or type(Z)==np.float64:
            return _reion(Z)
        elif type(Z)==np.ndarray or type(Z)==list:
            i = 0
            numofZ = len(Z)
            tau=np.zeros(numofZ)
            for X in Z:
                tau[i]=_reion(X)
                i=i+1
            return tau
    #End of functions related to reionization.
    #========================================================================================================
    
    def igm_eqns(self, Z,V):
        '''
        This function has the differential equations governing the ionisation and thermal history of the bulk of IGM. When solving upto the end of dark ages, only cosmological parameters will be used.
        Beyond ``Zstar``, i.e., beginning of cosmic dawn astrophysical will also be used.
        '''
        xe = V[0]
        Tk = V[1]

        #eq1 is (1+z)d(xe)/dz; see Weinberg's Cosmology book or eq.(71) from Seager et al (2000), ApJSS. Addtional correction based on Chluba et al (2015).            

        #eq2 is (1+z)dT/dz; see eq.(2.31) from Mittal et al (2022), JCAP
        
        if Z>Zstar:
            eq1 = 1/self.basic_cosmo_H(Z)*self.recomb_Peebles_C(Z,xe,self.basic_cosmo_Tcmb(Z))*(xe**2*self.basic_cosmo_nH(Z)*self.recomb_alpha(Tk)-self.recomb_beta(self.basic_cosmo_Tcmb(Z))*(1-xe)*np.exp(-Ea/(kB*self.basic_cosmo_Tcmb(Z))))

            eq2 = 2*Tk-Tk*eq1/(1+self.basic_cosmo_xHe()+xe)-self.heating_Ecomp(Z,xe,Tk)
        else:
            if xe<0.99:
                eq1 = 1/self.basic_cosmo_H(Z)*self.recomb_Peebles_C(Z,xe,self.basic_cosmo_Tcmb(Z))*(xe**2*self.basic_cosmo_nH(Z)*self.recomb_alpha(Tk)-self.recomb_beta(self.basic_cosmo_Tcmb(Z))*(1-xe)*np.exp(-Ea/(kB*self.basic_cosmo_Tcmb(Z))))-1/self.basic_cosmo_H(Z)*self.Gamma_x(Z,xe)*(1-xe)
            else:
                eq1 = 0.0
            eq2 = 2*Tk-Tk*eq1/(1+self.basic_cosmo_xHe()+xe)-self.heating_Ecomp(Z,xe,Tk)-self.heating_Elya(Z,xe,Tk)-self.heating_Ex(Z,xe)

        return np.array([eq1,eq2])

    def igm_solver(self, Z_eval, xe_init = None, Tk_init = None):

        #Assuming Z_eval is in decreasing order.
        Z_start = Z_eval[0]
        Z_end = Z_eval[-1]

        if Z_start == 1501:
            Tk_init = self.basic_cosmo_Tcmb(Z_start)
            xe_init = self.recomb_Saha_xe(Z_start,Tk_init)
            
        Sol = scint.solve_ivp(lambda a, Var: -self.igm_eqns(1/a,Var)/a, [1/Z_start, 1/Z_end],[xe_init,Tk_init],method='Radau',t_eval=1/Z_eval)

        #Obtaining the solutions ...
        xe = Sol.y[0]
        Tk = Sol.y[1]

        return [xe,Tk]

    def reion_eqn(self,Z,QHii):
        #eq is (1+z)dQ/dz; eq.(17) from Madau & Fragos (2007)

        if QHii<0.999:
            eq = -1/self.basic_cosmo_H(Z)*(self.fesc*Iion*self.sfrd(Z)/self.basic_cosmo_nH(1) - (1+self.basic_cosmo_xHe())*alpha_B*self.reion_clump(Z)*self.basic_cosmo_nH(Z)*QHii)
        else:
            eq = np.array([0.0])
        return eq
    
    def reion_solver(self):
        Sol = scint.solve_ivp(lambda a, Var: -self.reion_eqn(1/a,Var)/a, [1/Zstar, 1/Z_end],[0],method='Radau',t_eval=1/Z_cd)
        QHii = Sol.y[0]
        return QHii
    
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
        Ly :math:`\\alpha` coupling or the Wouthuysen--Field coupling.
        
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
            :math:`x_{\\mathrm{Ly}}`, dimensionless.
        '''
    
        Scat = self._scatter_corr(Z,xe,Tk)
        J_Ly = self.lya_spec_inten(Z)    #'undistorted' background Spec. Inte. of Lya photons.
        Jo = 5.54e-8*Z         #eq.(24) in Mittal & Kulkarni (2021)
        return Scat*(J_Ly[:,0]+J_Ly[:,1])/Jo

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

    def hyfi_twentyone_cm(self,Z,xe, Q,Ts):
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
        
        Ts : float
            Spin temperature.
        
        Returns
        -------
        
        float
            :math:`T_{21}`, mK.
        '''
        #Get the two-zone model averaged ionisation fraction.
        xHI = 1-_gaif(xe,Q)
        return 27*xHI*((1-self.Yp)/0.76)*(self.Om_b*self.h100**2/0.023)*np.sqrt(0.15*Z/(10*self.Om_m*self.h100**2))*(1-self.basic_cosmo_Tcmb(Z)/Ts)

#End of class main.
#========================================================================================================
#========================================================================================================
