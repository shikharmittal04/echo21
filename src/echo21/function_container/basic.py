import numpy as np
from ..const import *

class basic():
    def __init__(self, config):
        self.config = config
        return None

    def mu(self,xe):
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
        
        return 4/(4-3*self.config.Yp+4*xe*(1-self.config.Yp))
    
    def xHe(self):
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
        return 0.25*self.config.Yp/(1-self.config.Yp)

    def Tcmb(self,Z):
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
        return self.config.Tcmbo*Z

    def rho_crit(self):
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
        return 3*self.config.Ho**2/(8*np.pi*GN*Mpc2km**2)

    def nH(self,Z):
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
        return self.rho_crit()*self.config.Om_b*(1-self.config.Yp)*Z**3/mP

    def Hubble(self,Z):
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
        Om_lam = 1-self.config.Om_m
        Om_r = (1+fnu)*aS*self.config.Tcmbo**4/(cE**2*self.rho_crit())
        
        return self.config.Ho*(Om_r*Z**4+self.config.Om_m*Z**3+Om_lam)**0.5/Mpc2km