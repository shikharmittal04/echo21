import numpy as np
import scipy.integrate as scint
from scipy.interpolate import CubicSpline

from ..const import *

class eor():
    '''
    Class of all the functions related to reionization - clumping factor, CMB optical depth, analytical equation that governs the reionization, and the ODE solver.

    Methods
    ^^^^^^^
    '''
    def __init__(self, config, basic, halo):
        self.config = config
        self.basic = basic
        self.halo = halo
        
        self.QHii = self.solve_dQdlna()
        return None
    
    def clumping_factor(self,Z):
        '''
        Clumping factor for the ionization of hydrogen. From `Shull et al. (2012) <https://iopscience.iop.org/article/10.1088/0004-637X/747/2/100>`__.

        Arguments
        ---------
        Z : float
            :math:`1+z`
        '''
        return 20.81*Z**-1.1

    def cmb_tau(self,Z):
        '''
        Compute the Thomson-scattering optical depth up to a given redshift.

        Arguments
        ---------
        Z : float
            :math:`1+z` to which you want to calculate :math:`\\tau_{\\mathrm{e}}`.
        
        Returns
        -------

        float
            :math:`\\tau_{\\mathrm{e}}` (dimensionless).

        '''
        prefac = cE*sigT*self.basic.nH(1)
        xHe = self.basic.xHe()
        
        spl = CubicSpline(flipped_Z_CD, np.flip(self.QHii))

        def dtaudz(Z):
            he_factor = (1 + np.where(Z < 5, 2 * xHe, xHe))
            return prefac*he_factor*spl(Z)*Z**2/self.basic.Hubble(Z)

        Z_int = np.linspace(1,Z,60)
        tau = scint.trapezoid(dtaudz(Z_int),Z_int,axis=0)

        return np.atleast_1d(tau)
    
    def dQdlna(self,Z,QHii):
        '''
        Analytical equation of reionization (`Madau et al 1999 <https://iopscience.iop.org/article/10.1086/306975>`_).

        Arguments
        ---------
        Z : float
            :math:`1+z`

        QHii : float
            The volume filling factor of the ionized regions.
            
        Returns
        -------
        float
            :math:`\mathrm{d}Q/\mathrm{d}\ln(a)`
        '''

        if QHii<0.999:
            n_ion = self.config.fesc*Iion*self.halo.sfrd(Z)
            nH0 = self.basic.nH(1)
            AHe = 1+self.basic.xHe()
            Cf = self.clumping_factor(Z)
            Hub = self.basic.Hubble(Z)
            nH = self.basic.nH(Z)

            eq = 1/Hub*(n_ion/nH0 - AHe*alpha_B*Cf*nH*QHii)
        else:
            eq = np.array([0.0])
        return eq
    
    def solve_dQdlna(self):
        '''
        Solves the reionization equation.

        Returns
        -------

        float array
            QHii for the cosmic dawn redshifts, ``Z_CD``. 
        '''
        Sol = scint.solve_ivp(lambda lna, Var: self.dQdlna(np.exp(-lna),Var), [-np.log(Z_STAR), -np.log(Z_END)],[0],method='Radau',t_eval=-np.log(Z_CD))
        QHii = Sol.y[0]
        
        return QHii