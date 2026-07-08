import scipy.special as scsp
import scipy.integrate as scint
from ..const import *


def _fXh(xe):
    return 1-(1-xe**0.2663)**1.3163

def _fXion(xe):
    return 0.3908*(1-xe**0.4092)**1.7592


class heating():
    '''
    Class of all the *standard* heating terms (Compton, Ly-:math:`\\alpha`, X-ray). Exotic heating terms, such as those for IDM live, in their own module. Also, note that the return value is in the form of
     
    :math:`\\frac{2q}{3n_{\\mathrm{b}}k_{\\mathrm{B}}H}`,
    
    where :math:`q, n_{\\mathrm{b}}, k_{\\mathrm{B}}`, and :math:`H` are the volumetric heating rate, baryon number density, Boltzman constant, and Hubble factor, respectively.

    Within this class I have also included the ionization rate due to X-ray photons.

    Methods
    ^^^^^^^
    '''
    def __init__(self, config, basic, halo, lya):
        self.config = config
        self.basic = basic
        self.halo = halo
        self.lya = lya

    def Ecomp(self,Z,xe,Tk):
        '''
        See Eq.(2.32) from Mittal et al (2022), JCAP.
        (However, there is a typo in that equation; numerator has an :math:`x_{\\mathrm{e}}` missing.)
        
        Arguments
        ---------
        
        Z : float
            :math:`1+z`, dimensionless.
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature.
        
        Returns
        -------    
        
        float
            Compton heating. Units kelvin.

        '''
        compterm = (8*sigT*aS)/(3*me*cE)*self.basic.Tcmb(Z)**4*xe*(self.basic.Tcmb(Z)-Tk)/(self.basic.Hubble(Z)*(1+self.basic.xHe()+xe))
        return compterm


    def Elya(self,Z,xe,Tk):
        '''
        Ly-:math:`\\alpha` heating rate. For details see `Mittal & Kulkarni (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.4264M/abstract>`__ or the ``ECHO21`` paper `Mittal et al (2025) <https://arxiv.org/abs/2503.11762>`__
        
        Arguments
        ---------
        
        Z : float
            :math:`1+z`, dimensionless.
        
        xe : float
            Electron fraction.
        
        Tk : float
            Gas kinetic temperature.
        
        Returns
        -------    
        
        float
            Net heating by the Lyman series photons. Units kelvin.
        '''
            
        eta = self.lya._recoil(Tk)
        Scat = self.lya._scatter_corr(Z,xe,Tk)
        atau = self.lya._a_tau(Z,xe,Tk)
        arr = scsp.airy(-self.lya._xi2(Z,xe,Tk))
        
        Ic = eta*(2*np.pi**4*atau**2)**(1/3)*(arr[0]**2+arr[2]**2)
        Ii = eta*np.sqrt(atau/2)*scint.quad(lambda y:y**(-1/2)*np.exp(-2*eta*y-np.pi*y**3/(6*atau))*scsp.erfc(np.sqrt(np.pi*y**3/(2*atau))),0,np.inf)[0]-Scat*(1-Scat)/(2*eta)
        Jc_Ji = self.lya.lya_spec_inten(Z)
        nbary = (1+self.basic.xHe()+xe)*self.basic.nH(Z)
        [heat]=8*np.pi/3*hP/(kB*lam_alpha) * self.lya._dopp(Tk)/nbary *(Jc_Ji[:,0]*Ic+Jc_Ji[:,1]*Ii)

        return heat

    def Ex(self,Z,xe):
        '''
        We use the parametric approach for X-ray heating as in `Furlanetto (2006) <https://academic.oup.com/mnras/article/371/2/867/1033021>`__. We adopt the :math:`L_{\\mathrm{X}}/\\mathrm{SFR}` relation from `Lehmer et al. (2024) <https://iopscience.iop.org/article/10.3847/1538-4357/ad8de7>`__.
        
        Arguments
        ---------
        
        Z : float
            :math:`1+z`, dimensionless.
        
        xe : float
            Electron fraction.
        
        Returns
        -------    
        
        float
            Net heating by the X-ray photons. Units kelvin.
           
        '''

        if self.config.wX!=1: CX_modifier=(tilda_E1**(1-self.config.wX)-tilda_E0**(1-self.config.wX))/(E1**(1-self.config.wX)-E0**(1-self.config.wX))
        else: CX_modifier= np.log(tilda_E1/tilda_E0)/np.log(E1/E0)
        prefactor = 2/(3*self.basic.nH(1)*(1+self.basic.xHe()+xe)*kB*self.basic.Hubble(Z))
        return prefactor*self.config.fX*_fXh(xe)*self.halo.sfrd(Z)*CX_fid*CX_modifier

    def Gamma_x(self,Z,xe):
        '''
        Ionization (of bulk IGM) rate due to X-ray photons.
        
        Z : float
            :math:`1+z`, dimensionless.
        
        xe : float
            Electron fraction.
        
        Returns
        -------    
        
        float
            Ionization due to X-ray photons in units of :math:`\\mathrm{s}^{-1}`.
        '''
        prefactor = 2/(3*self.basic.nH(1)*(1+self.basic.xHe()+xe)*kB*self.basic.Hubble(Z))
        qX = self.Ex(Z,xe)/prefactor
        HX = qX/(_fXh(xe)*(1-xe)*self.basic.nH(1))
        Ew = 1e3*((tilda_E1**(-self.config.wX-2.4)-tilda_E0**(-self.config.wX-2.4))/(tilda_E1**(-self.config.wX-3.4)-tilda_E0**(-self.config.wX-3.4)))*(self.config.wX+3.4)/(self.config.wX+2.4)-13.6
        secondary_ionization = _fXion(xe)/13.6
        ionization_rate = HX*(1/Ew+secondary_ionization)/eC
        return ionization_rate
    
    #End of functions related to heating.