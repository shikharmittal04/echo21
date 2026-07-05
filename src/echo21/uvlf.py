'''
``uvlf``
========
This module contains all functions related to the UV LF and the number of galaxies seen by a survey.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import numpy as np
import scipy.integrate as scint
from .const import *

class uvlf():
    '''
    Class of functions relevant to galaxy surveys, such as luminosity functions and galaxy count for given limiting magnitude. If you use this module please consider citing `Mittal & Kulkarni (2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.2901M/abstract>`_

    However, note that there are some differences in the implementation of the UV LF and galaxy count functions in this module and the ones I used previously. The main difference is in the star formation rate. Previously, 

    :math:`\\dot{M}_{\\star} = f_{\\star} \\dot{M}_{\\star0} \\left(\\frac{M}{10^{10}M_{\\odot}}\\right)^{a} \\left(\\frac{1+z}{7}\\right)^{b}` \\,

    whereas in this module, we have used

    :math:`\\dot{M}_{\\star} = f_{\\star} \\frac{\\Omega_b}{\\Omega_m} \\frac{M}{t_{\\star}H(z)^{-1}}`

    *Accordingly, the following calculations are relevant only for semi-empirical star formation models.*

    Methods
    ~~~~~~~
    '''
    def __init__(self, funcs):
        self.funcs = funcs
        return None
    
    def dLUV_dM(self, Z):
        '''
        Rate of change of luminosity w.r.t. halo mass. (For our assumed constant star formation efficiency, this is independent of halo mass.)

        Arguments
        ---------
        Z : float
            :math:`1+z`
        
        Return
        ------
        float
            :math:`\\left(\\frac{\\mathrm{d}L}{\\mathrm{d}M}\\right)_{z}` in units of :math:`\\mathrm{W Hz^{-1}M_{\\odot}^{-1}}`
        '''
        return year * fstar * (self.funcs.Om_b/self.funcs.Om_m) * self.funcs.basic_cosmo_H(Z)*UVlum_by_SFR/self.funcs.tstar

    def luminosity(self, M,Z):
        '''
        Compute the UV luminosity for given halo mass and redshift.

        Arguments
        ---------
        
        M : float
            The desired halo mass at which you want to evaluate absolute magnitude. Input ``M`` in units of solar mass.
        
        Z : float
            1 + redshift, dimensionless.
        
        Return
        ------

        float
            UV luminosity in units of W/Hz.
        '''
        return self.dLUV_dM(Z) * M

    def halomass_to_absmag(self, M,Z):
        '''
        Absolute AB magnitude.

        Arguments
        ---------
        
        M : float
            The desired halo mass at which you want to evaluate absolute magnitude. Input ``M`` in units of solar mass.
        
        Z : float
            1 + redshift, dimensionless.
        
        Return
        ------

        float
            Absolute AB magnitude `(Oke 1974) <https://ui.adsabs.harvard.edu/abs/10.1086/190287>`_
        '''
        return -2.5*np.log10(self.luminosity(M,Z)/(4*np.pi*d10**2))-56.1

    def absmag_to_halomass(self, MAB,Z):
        '''
        Compute the halo mass which produces the given absolute magnitude at the given redshift.

        Arguments
        ---------
        
        MAB : float
            Absolute AB magnitude `(Oke 1974) <https://ui.adsabs.harvard.edu/abs/10.1086/190287>`_
        
        Z : float
            1 + redshift, dimensionless.
        
        Return
        ------

        float
            Halo mass in units of solar mass.
        '''
        lum = 4*np.pi*d10**2 * 10**(-0.4*(MAB+56.1)) #in units of W/Hz
        return lum/self.dLUV_dM(Z)

    def appmag_to_halomass(self, mAB,Z):
        '''
        Compute the halo mass which produces the given apparent magnitude at the given redshift.

        Arguments
        ---------
        
        mAB : float
            Apparent AB magnitude `(Oke 1974) <https://ui.adsabs.harvard.edu/abs/10.1086/190287>`_
        
        Z : float
            1 + redshift, dimensionless.
        
        Return
        ------

        float
            Halo mass in units of solar mass.
        '''
        d_L = Mpc2km*1e3 * self.funcs.my_cosmo.luminosityDistance(Z-1)/self.funcs.h100 #luminosity distance in meters.
        lum = 4*np.pi*d_L**2 * 10**(-0.4*(mAB+56.1)) #in units of W/Hz
        return lum/self.dLUV_dM(Z)
    
    def lum_func(self, MAB, Z):
        '''
        For given absolute magnitude and redshift, get UV luminosity function (LF). LF is defined as :math:`\\mathrm{d}\phi/\\mathrm{d}M_{\\mathrm{UV}}`, which represents number density (comoving) per unit absolute AB magnitude. Important note: valid only when star formation efficiency is a constant.

        Arguments
        ---------

        MAB : float or array_like
            Absolute AB magnitude.

        Z : float or array_like
            1 + redshift, dimensionless.

        Return
        ------

        float or ndarray
            Luminosity function in units of :math:`\\mathrm{cMpc}^{-3}`, where 'cMpc' represents comoving mega parsec.
            Shape is ``(len(MAB), len(Z))`` when both inputs are arrays; 1-D when one is scalar; scalar when both are scalar.
        '''
        MAB = np.asarray(MAB)
        Z = np.asarray(Z)
        scalar_MAB = MAB.ndim == 0
        scalar_Z = Z.ndim == 0
        MAB = np.atleast_1d(MAB)
        Z = np.atleast_1d(Z)

        result = np.empty((len(MAB), len(Z)))

        for j, Zj in enumerate(Z):
            M_halo = self.absmag_to_halomass(MAB, Zj)                          # shape (nMAB,)
            dLUV_dMAB = 0.4 * np.log(10) * self.luminosity(M_halo, Zj)       # shape (nMAB,)
            result[:, j] = self.funcs.dndM(M_halo, Zj) * dLUV_dMAB / self.dLUV_dM(Zj)

        if scalar_MAB and scalar_Z:
            return result[0, 0]
        elif scalar_MAB:
            return result[0, :]
        elif scalar_Z:
            return result[:, 0]
        return result
    
    def dNdz(self, mAB, Z, area = 1.0):
        '''
        Gradient of the number of galaxies seen at a given redshift and a limiting apparent magnitude of the survey and the survey area.

        Arguments
        ---------

        mAB : float
            Apparent AB magnitude `(Oke 1974) <https://ui.adsabs.harvard.edu/abs/10.1086/190287>`_ for for the faintest object the survey can see.

        Z : float
            :math:`1+z`; this can be an array as well

        area : float, optional
            Survey area in sq. deg.; default is 1.0.

        Returns
        -------

        float
            :math:`\\left(\\frac{\\mathrm{d}N}{\\mathrm{d}z}\\right)_{z}`; dimensionless
        '''
        Z = np.asarray(Z)
        scalar_input = Z.ndim == 0
        Z = np.atleast_1d(Z)

        area_sr = area * (np.pi/180)**2
        result = np.empty(Z.shape)

        for i, Zi in enumerate(Z.flat):
            Mh_lim = self.appmag_to_halomass(mAB, Zi)
            halo_masses = np.logspace(np.log10(Mh_lim), 16, 200)
            integral = scint.simpson(self.funcs.dndM(halo_masses, Zi), x=halo_masses)    #number per unit cMpc^3
            d_L = self.funcs.my_cosmo.luminosityDistance(Zi - 1) / self.funcs.h100      #luminosity distance in Mpc
            result.flat[i] = (1/(1e3*Mpc2km) * cE / self.funcs.basic_cosmo_H(Zi)
                              * (d_L/Zi)**2 * integral * area_sr)

        return result.squeeze() if scalar_input else result
    
    def num_gal(self, mAB, Z, area=1.0):
        '''
        Cumulative number of galaxies brighter than the limiting apparent magnitude from today to given redshift.

        Arguments
        ---------

        mAB : float
            Apparent AB magnitude `(Oke 1974) <https://ui.adsabs.harvard.edu/abs/10.1086/190287>`_ for for the faintest object the survey can see.
        
        Z : float
            :math:`1+z`; this can be an array as well

        area : float, optional
            Survey area in sq. deg.; default is 1.0.
        
        Returns
        -------

        float
            :math:`\\int_0^{z} \\left(\\frac{\\mathrm{d}N}{\\mathrm{d}z}\\right)_{z'} \\mathrm{d}z'`; dimensionless
        '''
        Z_int = np.linspace(1,Z,10*int(Z-1))
        num_gal_upto_Z = scint.simpson(self.dNdz(mAB, Z_int, area), x=Z_int)
        
        return num_gal_upto_Z