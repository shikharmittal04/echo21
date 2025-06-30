import scipy.special as scsp
import scipy.integrate as scint
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import CubicSpline
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.lss import mass_function
import warnings
from pathlib import Path
from .const import *

warnings.filterwarnings('ignore')
home_path = str(Path.home())

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
    def __init__(self,Ho=67.4,Om_m=0.315,Om_b=0.049,sig8=0.811,ns=0.965,Tcmbo=2.725,Yp=0.245,mx_gev=None,sigma45=None,fLy=1.0,sLy=2.64,fX=1,wX=1.5,fesc=0.0106,cosmo=None,astro=None,**kwargs):
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
            try:
                mx_gev = cosmo['mx_gev']
                sigma45 = cosmo['sigma45']
            except:
                pass
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
        
        self.mx_gev = mx_gev
        self.sigma45 = sigma45

        self.fLy = fLy
        self.sLy = sLy
        self.fX = fX
        self.wX = wX
        self.fesc = fesc
        
        ############################################################################
        #Setting up cosmology for COLOSSUS package
        self.cosmo_par = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': sig8, 'ns': ns,'relspecies': True,'Tcmb0': Tcmbo}
        self.my_cosmo = cosmology.setCosmology('cosmo_par', self.cosmo_par)
        self.h100 = self.Ho/100
        
        ############################################################################
        #Setting up the star formation rate density related parameters and functions
        self.sfrd_type = kwargs.pop('type', 'phy')

        if self.sfrd_type == 'phy':
            self.mdef = kwargs.pop('mdef','fof')
            self.hmf = kwargs.pop('hmf','press74')
            self.Tmin_vir = kwargs.pop('Tmin_vir',1e4)

            if self.hmf == 'press74':
                self._f_coll = self._f_coll_press74
            else:
                self._f_coll = self._f_coll_nonpress74
            
            self._sfrd = self._sfrd_phy

        elif self.sfrd_type == 'emp':
            self._sfrd = self._sfrd_emp
            self.a_sfrd = kwargs.pop('a',0.257)
            self.b_sfrd = kwargs.pop('b',4)

        elif self.sfrd_type == 'semi-emp':
            self.mdef = kwargs.pop('mdef','fof')
            self.hmf = kwargs.pop('hmf','press74')
            self.Tmin_vir = kwargs.pop('Tmin_vir',1e4)
            self.t_star = kwargs.pop('t_star',0.5)

            if self.hmf == 'press74':
                self._f_coll = self._f_coll_press74
            else:
                self._f_coll = self._f_coll_nonpress74
            
            self._sfrd = self._sfrd_semi_emp

        else:
            raise ValueError(f"Unknown SFRD type: {self.sfrd_type}")
        ############################################################################
        #Checking if the DM model is interacting or cold        
        
        self.is_idm = False
        if all(x is not None for x in [self.mx_gev, self.sigma45]):
            self.is_idm = True

        if self.is_idm:
            self.mx = mx_gev*GeV2kg #Now mx is in kg
            self.sigma0 = sigma45*sig_ten45m2   #Now sigma0 is in m^2

            npz_file = f'{home_path}/.echo21/f_coll_idm.npz'
            # Load the compressed grid
            data = np.load(npz_file)
            
            f_coll = data['fcoll']            # Shape: (Nmdm, Nsigma, Nz, Nmass)

            mdmeff_vals = data['mdmeff']
            sigma0_vals = data['sigma0']
            zvals = data['zvals']
            halomass_vals = data['halomass']

            i_mdm = np.argmin(np.abs(mdmeff_vals - mx_gev))
            i_sigma = np.argmin(np.abs(sigma0_vals - self.sigma0))

            fcoll_slice = f_coll[i_mdm, i_sigma, :, :]
            self.rbs = RectBivariateSpline(zvals, halomass_vals, fcoll_slice)

            self._f_coll = self._f_coll_idm
            self._igm_eqns = self._igm_eqns_idm
            self._igm_solver = self._igm_solver_idm
        else:
            self._igm_eqns = self._igm_eqns_cdm
            self._igm_solver = self._igm_solver_cdm
        ############################################################################        
        #Solve reionization at initialization itself        

        self.QHii = self.reion_solver()
        return None

    def basic_cosmo_mu(self,xe):
        '''
        The average baryon mass.

        Arguments
        ---------

        xe : float
            Electron fraction, dimensionless

        Returns
        -------
            :math:`\\mu`, dimensionless
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
        M_by_h = M*self.h100 #M is in solar mass units and M_by_h is in units of solar mass/h.
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

    def _f_coll_press74(self, Z):
        return scsp.erfc(peaks.peakHeight(self.m_min(Z),Z-1)/np.sqrt(2))
    
    def _f_coll_nonpress74(self,Z):
        Z = np.atleast_1d(Z)
        single_value = Z.size == 1
        rho_halo_arr = np.zeros_like(Z)

        for idx, z in enumerate(Z):
            M_space = np.logspace(np.log10(self.m_min(z)/self.h100),18,1500)    #These masses are in solar mass. Strictly speaking we should integrate up to infinity but for numerical purposes 10^18.Msun is sufficient.
            hmf_space = self.dndlnM(M=M_space,Z=z)    #Corresponding HMF values are in cMpc^-3
            rho_halo_arr[idx]=scint.simpson(hmf_space,x=M_space)

        F_coll = rho_halo_arr *Msolar_by_Mpc3_to_kg_by_m3/(self.Om_m*self.basic_cosmo_rho_crit())
        return F_coll[0] if single_value else F_coll
    
    def _f_coll_idm(self, Z):
        scalar_input = np.isscalar(Z)
        Z = np.atleast_1d(Z)

        results = np.zeros_like(Z, dtype=float)  # Initialize all results to 0

        valid = Z <= 61  # Boolean mask for Z values that are <= 60

        if np.any(valid):
            Z_valid = Z[valid]
            mmin = self.m_min(Z_valid) / self.h100
            results[valid] = self.rbs.ev(Z_valid - 1,mmin)
            
        return results[0] if scalar_input else results
    
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
        return self._f_coll(Z)

    def dfcoll_dz(self,Z):
        '''
        Redshift derivative of the collapse fraction, i.e., :math:`\\mathrm{d}F_{\\mathrm{coll}}/\\mathrm{d}z`
        '''
        return (self.f_coll(Z+1e-3)-self.f_coll(Z))*1e3

    def _sfrd_phy(self,Z):
        Z = np.atleast_1d(Z)
        mysfrd = -Z*fstar*self.Om_b*self.basic_cosmo_rho_crit()*self.dfcoll_dz(Z)*self.basic_cosmo_H(Z)
        return mysfrd if mysfrd.size > 1 else mysfrd[0]
    
    def _sfrd_semi_emp(self,Z):
        return fstar*self.Om_b*self.basic_cosmo_rho_crit()*self.basic_cosmo_H(Z)*self.f_coll(Z)/self.t_star

    def _sfrd_emp(self,Z):
        Z = np.atleast_1d(Z)
        Zcut = 1 + self.b_sfrd

        lowz = 0.015 * Z**2.73 / (1 + (Z / 3)**6.2)
        highz = 0.015 * Zcut**2.73 / (1 + (Zcut / 3)**6.2) * 10**(self.a_sfrd * (Zcut - Z))

        mysfrd = np.where(Z < Zcut, lowz, highz)
        mysfrd *= Msolar_by_Mpc3_year_to_kg_by_m3_sec

        return mysfrd if mysfrd.size > 1 else mysfrd[0]
     
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
            Comoving SFRD in units of :math:`\\mathrm{kgs^{-1}m^{-3}}`. Single number or an array accordingly as ``Z`` is single number or an array. To convert to solar mass per year per cubic Mpc, use the factor ``Msolar_by_Mpc3_year_to_kg_by_m3_sec`` available in the module ``const``.
        '''
        return self._sfrd(Z)
    
    #End of functions related to HMF and SFRD.
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
        Z = np.atleast_1d(Z).astype(float)
        J = np.zeros((len(Z), 2))  # [continuum, injected]
        
        valid = Z <= Zstar
        Z_valid = Z[valid]

        for i, z in enumerate(Z_valid):
            prefac = cE / (4 * np.pi) * z ** 2

            # Continuum contribution
            Zmax_cont = 32 / 27 * z
            zgrid_cont = np.linspace(z, Zmax_cont, 10)
            eps_cont = self.eps_Ly(zgrid_cont, 10.2 * zgrid_cont / z)
            H_cont = self.basic_cosmo_H(zgrid_cont)
            continuum = scint.trapezoid(eps_cont / H_cont, zgrid_cont)

            # Injected contribution
            injected = 0.0
            for ni, pn in zip(range(4, 24), Pn):
                Zmax_inj = (1 - 1 / (ni + 1) ** 2) / (1 - 1 / ni ** 2) * z
                zgrid_inj = np.linspace(z, Zmax_inj, 5)
                eps_inj = self.eps_Ly(zgrid_inj, 13.6 * (1 - 1 / ni ** 2) * zgrid_inj / z)
                H_inj = self.basic_cosmo_H(zgrid_inj)
                injected += pn * scint.trapezoid(eps_inj / H_inj, zgrid_inj)

            J[i if not valid.any() else np.where(valid)[0][i]] = prefac * continuum, prefac * injected

        # Return scalar if input was scalar
        return tuple(J[0]) if np.isscalar(Z) else J

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
        Ly :math:`\\alpha` heating rate. For details see `Mittal & Kulkarni (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.4264M/abstract>`__ or the ECHO21 paper `Mittal et al (2025) <https://arxiv.org/abs/2503.11762>`__
        
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
        Jc_Ji = self.lya_spec_inten(Z)
        nbary = (1+self.basic_cosmo_xHe()+xe)*self.basic_cosmo_nH(Z)
        [heat]=8*np.pi/3*hP/(kB*lam_alpha) * self._dopp(Tk)/nbary *(Jc_Ji[:,0]*Ic+Jc_Ji[:,1]*Ii)
        return heat

    def heating_Ex(self,Z,xe):
        '''
        We use the parametric approach for X-ray heating as in `Furlanetto (2006) <https://academic.oup.com/mnras/article/371/2/867/1033021>`__. We adopt the :math:`L_{\\mathrm{X}}/\\mathrm{SFR}` relation from `Lehmer et al. (2024) <https://iopscience.iop.org/article/10.3847/1538-4357/ad8de7>`__.
        
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
        prefactor = 2/(3*self.basic_cosmo_nH(1)*(1+self.basic_cosmo_xHe()+xe)*kB*self.basic_cosmo_H(Z))
        return prefactor*self.fX*self._fXh(xe)*self.sfrd(Z)*CX_fid*CX_modifier

    #End of functions related to heating.
    #========================================================================================================

    # The following functions are relevant to Coloumb-like DM particles.
 
    def u_t(self, xe,Tk,Tx, target='p'):
        '''
        The characteristic thermal sound speed of the DM-baryon fluid.
        
        Arguments
        ---------
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature (K).
        
        Tx : float
            DM temperature (K).

        Returns
        -------    
        
        float
            :math:`u_{\\mathrm{th}} (\\mathrm{m\\,s^{-1}})`.
        '''
        
        if (target == 'e'):
            return np.sqrt(kB*Tk/me+kB*Tx/self.mx)
        if (target == 'p'):
            return np.sqrt(kB*Tk/(self.basic_cosmo_mu(xe)*mP)+kB*Tx/self.mx)

    def r_t(self,xe,Tk,Tx,v_bx,target='p'):
        '''
        Ratio of relative velocity of DM and baryons to the characteristic thermal sound speed.
         
        Arguments
        ---------
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature (K).
        
        Tx : float
            DM temperature (K).

        v_bx : float
            Relative velocity of DM and baryons (m/s).
        
        Returns
        -------    
        
        float
            :math:`v_{\\mathrm{b}\\chi}/u_{\\mathrm{th}}`, dimensionless.
        '''
        if (target == 'e'):
            return v_bx/self.u_t(xe,Tk,Tx, 'e')
        if (target == 'p'):
            return v_bx/self.u_t(xe,Tk,Tx, 'p')

    def F(self, x):
        return scsp.erf(x/np.sqrt(2))- np.sqrt(2/np.pi)*x*np.exp(-x**2/2)

    def Drag(self,Z,xe,Tk,Tx,v_bx):
        '''
        Drag due to DM baryon interaction.

        Arguments
        ---------
        
        Z : float
            1+z
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature (K).
        
        Tx : float
            DM temperature (K).

        v_bx : float
            Relative velocity of DM and baryons (m/s).
        
        Returns
        -------    
        
        float
            :math:`D (\\mathrm{m\\,s^{-2}})`.
        '''        
        rho_b = Z**3*self.basic_cosmo_rho_crit()*self.Om_b
        rho_x = Z**3*self.basic_cosmo_rho_crit()*(self.Om_m-self.Om_b)

        return cE**4*self.sigma0*(rho_x+rho_b)/(self.mx+self.basic_cosmo_mu(xe)*mP) * self.F(self.r_t(xe,Tk,Tx,v_bx,'p'))/v_bx**2
        
    def mu_bx(self,xe):
        '''
        Reduced mass for DM-baryon system.

        Arguments
        ---------
        
        xe : float
            Electron fraction.
        
        Returns
        -------

        float
            :math:`\\mu_{\\mathrm{b}\\chi} (\\mathrm{kg})`
        '''
        return self.basic_cosmo_mu(xe)*mP*self.mx/(self.basic_cosmo_mu(xe)*mP+self.mx)

    def Ex2b(self,Z,xe,Tk,Tx,v_bx):
        '''
        This corresponds to the heat that flows into the baryonic system from the DM.
        
        Arguments
        ---------
        
        Z : float
            1+z
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature (K).
        
        Tx : float
            DM temperature (K).

        v_bx : float
            Relative velocity of DM and baryons :math:`(\\mathrm{m\\,s^{-1}})`.
        
        Returns
        -------    
        
        float
            :math:`\\dot{Q}_{\\mathrm{k}} (\\mathrm{K})`.
        '''
        # fraction of DM which is coloumb-like (in kg/m^3 proper)
        rho_x = self.basic_cosmo_rho_crit()*(self.Om_m-self.Om_b)*Z**3	

        #mass density of baryons only (in kg/m^3 proper)
        rho_b = self.basic_cosmo_rho_crit()*self.Om_b*Z**3 
        
        rp = self.r_t(xe,Tk,Tx,v_bx,'p')
        
        up = self.u_t(xe,Tk,Tx, 'p')
        term1 = cE**4*2*self.basic_cosmo_mu(xe)*mP*rho_x*self.sigma0*np.exp(-rp**2/2)*(Tx-Tk)/((self.mx+self.basic_cosmo_mu(xe)*mP)**2*np.sqrt(2*np.pi)*up**3)
        term2 = 1/kB*rho_x/(rho_x+rho_b)*self.mu_bx(xe)*v_bx*self.Drag(Z,xe,Tk,Tx,v_bx)
        return 2/(3*self.basic_cosmo_H(Z))*(term1+term2)

    def Eb2x(self,Z,xe,Tk,Tx,v_bx):
        '''
        This corresponds to the heat that flows into the DM from baryons.
        
        Arguments
        ---------
        
        Z : float
            1+z
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature (K).
        
        Tx : float
            DM temperature (K).

        v_bx : float
            Relative velocity of DM and baryons (m/s).
        
        Returns
        -------    
        
        float
            :math:`\\dot{Q}_{\\chi}` (K).
        '''
        # fraction of DM which is coloumb-like (in kg/m^3 proper)
        rho_x = self.basic_cosmo_rho_crit()*(self.Om_m-self.Om_b)*Z**3

        #mass density of baryons only (in kg/m^3 proper)
        rho_b = self.basic_cosmo_rho_crit()*self.Om_b*Z**3 

        rp = self.r_t(xe,Tk,Tx,v_bx, 'p')
        
        up = self.u_t(xe,Tk,Tx, 'p')
        term1 = cE**4*2*self.mx*rho_b*self.sigma0*np.exp(-rp**2/2)*(Tk-Tx)/((self.mx+self.basic_cosmo_mu(xe)*mP)**2*np.sqrt(2*np.pi)*up**3)
        term2 = 1/kB*rho_b/(rho_x+rho_b)*self.mu_bx(xe)*v_bx*self.Drag(Z,xe,Tk,Tx,v_bx)
        return 2/(3*self.basic_cosmo_H(Z))*(term1+term2)
    
    #End of functions related to IDM.
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
        prefactor = 2/(3*self.basic_cosmo_nH(1)*(1+self.basic_cosmo_xHe()+xe)*kB*self.basic_cosmo_H(Z))
        qX = self.heating_Ex(Z,xe)/prefactor
        HX = qX/(self._fXh(xe)*(1-xe)*self.basic_cosmo_nH(1))
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

    def reion_tau(self,Z):
        '''
        Compute the Thomson-scattering optical depth up to a 1+redshift=Z.

        Arguments
        ---------
        Z : float
            1+z to which you want to calculate :math:`\\tau_{\\mathrm{e}}`.
        
        Returns
        -------

        float
            :math:`\\tau_{\\mathrm{e}}` (dimensionless).

        '''
        prefac = cE*sigT*self.basic_cosmo_nH(1)
        xHe = self.basic_cosmo_xHe()
        
        QHii = self.QHii
        spl = CubicSpline(flipped_Z_cd, np.flip(QHii))

        def dtaudz(Z):
            he_factor = (1 + np.where(Z < 5, 2 * xHe, xHe))
            return prefac*he_factor*spl(Z)*Z**2/self.basic_cosmo_H(Z)

        Z_int = np.linspace(1,Z,100)
        tau = scint.trapezoid(dtaudz(Z_int),Z_int,axis=0)

        return tau
    #End of functions related to reionization.
    #========================================================================================================
    
    def _igm_eqns_cdm(self, Z,V):
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
            eq2 = 2*Tk-Tk*eq1/(1+self.basic_cosmo_xHe()+xe)-self.heating_Ecomp(Z,xe,Tk)-self.heating_Ex(Z,xe)-self.heating_Elya(Z,xe,Tk)
        
        return np.array([eq1,eq2])

    def _igm_eqns_idm(self, Z,V):
        xe = V[0]
        Tk = V[1]
        Tx = V[2]
        v_bx= V[3]
        
        #eq1 is (1+z)d(xe)/dz; see Weinberg's Cosmology book or eq.(71) from Seager et al (2000), ApJSS. Addtional correction based on Chluba et al (2015).            

        #eq2 is (1+z)dT/dz; see eq.(2.31) from Mittal et al (2022), JCAP
        
        if Z>Zstar:
            eq1 = 1/self.basic_cosmo_H(Z)*self.recomb_Peebles_C(Z,xe,self.basic_cosmo_Tcmb(Z))*(xe**2*self.basic_cosmo_nH(Z)*self.recomb_alpha(Tk)-self.recomb_beta(self.basic_cosmo_Tcmb(Z))*(1-xe)*np.exp(-Ea/(kB*self.basic_cosmo_Tcmb(Z))))

            eq2 = 2*Tk-Tk*eq1/(1+self.basic_cosmo_xHe()+xe)-self.heating_Ecomp(Z,xe,Tk)-self.Ex2b(Z,xe,Tk,Tx,v_bx)
        else:
            if xe<0.99:
                eq1 = 1/self.basic_cosmo_H(Z)*self.recomb_Peebles_C(Z,xe,self.basic_cosmo_Tcmb(Z))*(xe**2*self.basic_cosmo_nH(Z)*self.recomb_alpha(Tk)-self.recomb_beta(self.basic_cosmo_Tcmb(Z))*(1-xe)*np.exp(-Ea/(kB*self.basic_cosmo_Tcmb(Z))))-1/self.basic_cosmo_H(Z)*self.Gamma_x(Z,xe)*(1-xe)
            else:
                eq1 = 0.0
            eq2 = 2*Tk-Tk*eq1/(1+self.basic_cosmo_xHe()+xe)-self.heating_Ecomp(Z,xe,Tk)-self.Ex2b(Z,xe,Tk,Tx,v_bx)-self.heating_Elya(Z,xe,Tk)-self.heating_Ex(Z,xe)

        #eq3 is (1+z)dTx/dz;
        eq3 = 2*Tx-self.Eb2x(Z,xe,Tk,Tx,v_bx)
        
        #eq2 is (1+z)dv_bx/dz;
        eq4 = v_bx + self.Drag(Z,xe,Tk,Tx,v_bx)/self.basic_cosmo_H(Z)
        return np.array([eq1,eq2,eq3,eq4])
    
    def igm_eqns(self, Z,V):
        '''
        This function has the differential equations governing the ionization and thermal history of the bulk of IGM. When solving upto the end of dark ages, only cosmological parameters will be used. Beyond ``Zstar``, i.e., after the beginning of cosmic dawn astrophysical will also be used.
        '''
        return self._igm_eqns(Z,V)

    def _igm_solver_cdm(self, Z_eval, xe_init = None, Tk_init = None):

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

    def _igm_solver_idm(self, Z_eval, xe_init = None, Tk_init = None, Tx_init = None, v_bx_init = None):

        #Assuming Z_eval is in decreasing order.
        Z_start = Z_eval[0]
        Z_end = Z_eval[-1]

        if Z_start == 1501:
            Tk_init = self.basic_cosmo_Tcmb(Z_start)
            xe_init = self.recomb_Saha_xe(Z_start,Tk_init)
            Tx_init = 0
            v_bx_init = 43500
            
        Sol = scint.solve_ivp(lambda a, Var: -self.igm_eqns(1/a,Var)/a, [1/Z_start, 1/Z_end],[xe_init,Tk_init, Tx_init, v_bx_init],method='Radau',t_eval=1/Z_eval)

        #Obtaining the solutions ...
        xe = Sol.y[0]
        Tk = Sol.y[1]
        Tx = Sol.y[2]
        v_bx = Sol.y[3]

        return [xe,Tk,Tx,v_bx]

    def igm_solver(self,Z_eval,**kwargs):
        '''
        This function solves the coupled IGM differential equations. In case of CDM it is just electron fraction and gas temperature. When IDM is involed DM temperature and relative DM-baryon velocity is also solved.
        '''
        return self._igm_solver(Z_eval,**kwargs)
    
    def reion_eqn(self,Z,QHii):
        '''
        The RHS of the differential equation governing the evolution of Q. Equation is (1+z)dQ/dz; eq.(17) from Madau & Fragos (2017).

        Arguments
        ---------
        Z : float
            1+z.

        Q : float
            The volume filling factor of the ionized regions.
            
        Returns
        -------
        float
            (1+z)dQ/dz
        '''

        if QHii<0.999:
            eq = -1/self.basic_cosmo_H(Z)*(self.fesc*Iion*self.sfrd(Z)/self.basic_cosmo_nH(1) - (1+self.basic_cosmo_xHe())*alpha_B*self.reion_clump(Z)*self.basic_cosmo_nH(Z)*QHii)
        else:
            eq = np.array([0.0])
        return eq
    
    def reion_solver(self):
        '''
        Solves the reionization equation.

        Arguments
        ---------
        None

        Returns
        -------

        float array
            Q for the cosmic dawn redshifts, ``Z_cd``. The redshifts can be access from the module ``const``. 
        '''
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
        return np.where(Tk>0.0423,(4.28+0.1023*lnT-0.2586*lnT**2+0.04321*lnT**3)*1e-16, 1.12e-19)
    
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
        Jc_Ji = self.lya_spec_inten(Z)    #'undistorted' background Spec. Inte. of Lya photons.
        Jo = 5.54e-8*Z         #eq.(24) in Mittal & Kulkarni (2021)
        return Scat*(Jc_Ji[:,0]+Jc_Ji[:,1])/Jo

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

#End of class echofuncs.
#========================================================================================================
#========================================================================================================
