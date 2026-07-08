import numpy as np
from ..const import *

class hyfi():
    def __init__(self, config, basic, lyman_alpha):
        self.config = config
        self.basic = basic
        self.lya = lyman_alpha
    
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


    def kpH(self,Tk):
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
    
    def col_coup(self,Z,xe,Tk):
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
        return Tstar*self.basic.nH(Z)*((1-xe)*self.kHH(Tk)+xe*self.keH(Tk)+xe*self.kpH(Tk))/(A10*self.basic.Tcmb(Z))

    def lya_coup(self,Z,xe,Tk):
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
    
        Scat = self.lya._scatter_corr(Z,xe,Tk)
        Jc_Ji = self.lya.lya_spec_inten(Z)    #'undistorted' background Spec. Inte. of Lya photons.
        Jo = 5.54e-8*Z         #eq.(24) in Mittal & Kulkarni (2021)
        return Scat*(Jc_Ji[:,0]+Jc_Ji[:,1])/Jo

    def spin_temp(self,Z,xe,Tk):
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

        xa = self.lya_coup(Z,xe,Tk)
        xk = self.col_coup(Z,xe,Tk)
        Ts = ( 1  + xa + xk*Tk/(Tk+T_se))/(1/self.basic.Tcmb(Z) +  xk/Tk + xa/(Tk+T_se) )
        return Ts

    def twentyone_cm(self, Z, xHI, Ts):
        '''
        The global (sky-averaged) 21-cm signal.
        
        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        xHI : float
            Two-zone-model-averaged neutral hydrogen fraction.
        
        Ts : float
            Spin temperature.
        
        Returns
        -------
        
        float
            :math:`T_{21}`, mK.
        '''

        return 27*xHI*((1-self.config.Yp)/0.76)*(self.config.Om_b*self.config.h100**2/0.023)*np.sqrt(0.15*Z/(10*self.config.Om_m*self.config.h100**2))*(1-self.basic.Tcmb(Z)/Ts)