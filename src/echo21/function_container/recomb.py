import numpy as np
from ..const import *

class recomb():
    '''
    Class of all the recombination-physics-related functions.

    Methods
    ^^^^^^^
    '''
    def __init__(self, config, basic):
        self.config = config
        self.basic = basic
        pass

    def alpha(self, T):
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

    def beta(self, T):
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
        beta = self.alpha(T)*(2*np.pi*me*kB*T/hP**2)**1.5*np.exp(-B2/(kB*T))
        return beta

    def Krr(self, Z):
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
        return lam_alpha**3/(8*np.pi*self.basic.Hubble(Z))

    def Peebles_C(self,Z,xe,T):
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
        
        return (1+self.Krr(Z)*Lam_H*self.basic.nH(Z)*(1-xe))/(1+self.Krr(Z)*(Lam_H+self.beta(T))*self.basic.nH(Z)*(1-xe))

    def Saha_xe(self,Z,T):
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
        Saha=1/self.basic.nH(Z)*(2*np.pi*me*kB*T/hP**2)**1.5*np.exp(-B1/(kB*T))
        return (np.sqrt(Saha**2+4*Saha)-Saha)/2