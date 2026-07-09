import numpy as np
import scipy.integrate as scint
from ..const import *

class lyman_alpha():
    '''
    Class of all the functions required to construct the specific intensity of the Ly-:math:`\\alpha` photons.

    Methods
    ^^^^^^^
    '''
    def __init__(self, config, basic, halo):
        self.config = config
        self.basic = basic
        self.halo = halo
        
        if self.config.sLy!=0:
            self._phi_Ly = self._phi_Ly_non0_sLy
        else:
            self._phi_Ly = self._phi_Ly_0_sLy

    def _recoil(self, Tk):
        '''
        The recoil parameter. Eq.(15) in Mittal & Kulkarni (2021).
        '''
        return 0.02542/np.sqrt(Tk)

    def _dopp(self, Tk):
        '''
        Doppler width for Lya-HI interaction. Eq.(14) in Mittal & Kulkarni (2021).
        '''
        return nu_alpha*np.sqrt(2*kB*Tk/(mP*cE**2))

    def _a_tau(self,Z,xe,Tk):
        '''
        Returns the product :math:`a\\tau`, since all the relevant formulae require the product only.
        :math:`a` is the Voigt parameter and :math:`\\tau` is the optical depth of Lya photons.
        '''
        tau = 3/(8*np.pi)*A_alpha/self.basic.Hubble(Z)*self.basic.nH(Z)*(1-xe)*lam_alpha**3
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

    def _phi_Ly_non0_sLy(self,E):
        return self.config.fLy*hP/eC*1/13.6*self.config.sLy*N_alpha_infty/(1.33**self.config.sLy-1)*(E/13.6)**(-self.config.sLy-1)

    def _phi_Ly_0_sLy(self,E):
        return self.config.fLy*hP/eC*N_alpha_infty/np.log(4/3)*E**-1
        
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
        return self._phi_Ly(E)

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
        return 1/(1.22*mP)*self.phi_Ly(E)*self.halo.sfrd(Z)
    
    def lya_spec_inten(self,Z):
        '''
        Specific intensity of Ly-:math:`\\alpha` photons, :math:`J_{\\mathrm{Ly}}`, due to continuum and injected photons.
        
        Arguments
        ---------
        Z : float
            :math:`1 + z`, dimensionless. Can be array.
        
        Returns
        -------
        
        float
            Specific intensity in terms of number per unit time per unit area per unit frequency per unit solid angle (:math:`\\mathrm{m^{-2}s^{-1}Hz^{-1}sr^{-1}}`). Two values are returned, namely intensity due to continuum and injected photons, respectively.
        '''
        Z = np.atleast_1d(Z).astype(float)
        J = np.zeros((len(Z), 2))  # [continuum, injected]
        
        valid = Z <= Z_STAR
        Z_valid = Z[valid]

        for i, z in enumerate(Z_valid):
            prefac = cE / (4 * np.pi) * z ** 2

            # Continuum contribution
            Zmax_cont = 32 / 27 * z
            zgrid_cont = np.linspace(z, Zmax_cont, 10)
            eps_cont = self.eps_Ly(zgrid_cont, 10.2 * zgrid_cont / z)
            H_cont = self.basic.Hubble(zgrid_cont)
            continuum = scint.trapezoid(eps_cont / H_cont, zgrid_cont)

            # Injected contribution
            injected = 0.0
            for ni, pn in zip(range(4, 24), Pn):
                Zmax_inj = (1 - 1 / (ni + 1) ** 2) / (1 - 1 / ni ** 2) * z
                zgrid_inj = np.linspace(z, Zmax_inj, 5)
                eps_inj = self.eps_Ly(zgrid_inj, 13.6 * (1 - 1 / ni ** 2) * zgrid_inj / z)
                H_inj = self.basic.Hubble(zgrid_inj)
                injected += pn * scint.trapezoid(eps_inj / H_inj, zgrid_inj)

            J[i if not valid.any() else np.where(valid)[0][i]] = prefac * continuum, prefac * injected

        # Return scalar if input was scalar
        return tuple(J[0]) if np.isscalar(Z) else J
