import numpy as np
import scipy.special as scsp
from ..const import *

class idm():
    def __init__(self, config, basic):
        self.config = config
        self.basic = basic
    
    def u_t(self, xe,Tk,Tx):
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
        
        ut = np.sqrt(kB*Tk/(self.basic.mu(xe)*mP)+kB*Tx/self.config.mx)
        return ut

    def r_t(self,xe,Tk,Tx,v_bx):
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
        return v_bx/self.u_t(xe,Tk,Tx)

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
        rho_b = Z**3*self.basic.rho_crit()*self.config.Om_b
        rho_x = Z**3*self.basic.rho_crit()*(self.config.Om_m-self.config.Om_b)
        rp = self.r_t(xe,Tk,Tx,v_bx)
        up = self.u_t(xe,Tk,Tx)
        prefactor = cE**4*self.config.sigma0*(rho_x+rho_b)/(self.config.mx+self.basic.mu(xe)*mP) * 1/up**2
        if rp>=0.001:
            D = prefactor * self.F(rp)/rp**2
        else:
            D = prefactor * np.sqrt(2/np.pi)*(rp/3 - rp**3/10 + rp**5/56)
        
        return D

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
        return self.basic.mu(xe)*mP*self.config.mx/(self.basic.mu(xe)*mP+self.config.mx)

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
        temp_diff = Tk - Tx

        # fraction of DM which is coloumb-like (in kg/m^3 proper)
        rho_x = self.basic.rho_crit()*(self.config.Om_m-self.config.Om_b)*Z**3	

        #mass density of baryons only (in kg/m^3 proper)
        rho_b = self.basic.rho_crit()*self.config.Om_b*Z**3 
        
        rp = self.r_t(xe,Tk,Tx,v_bx)
        
        up = self.u_t(xe,Tk,Tx)
        term1 = cE**4*2*self.basic.mu(xe)*mP*rho_x*self.config.sigma0*np.exp(-rp**2/2)*(-temp_diff)/((self.config.mx+self.basic.mu(xe)*mP)**2*np.sqrt(2*np.pi)*up**3)
        term2 = 1/kB*rho_x/(rho_x+rho_b)*self.mu_bx(xe)*v_bx*self.Drag(Z,xe,Tk,Tx,v_bx)
        
        retval = 2/(3*self.basic.Hubble(Z))*(term1+term2)
        return retval

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
        temp_diff = Tk - Tx

        # fraction of DM which is coloumb-like (in kg/m^3 proper)
        rho_x = self.basic.rho_crit()*(self.config.Om_m-self.config.Om_b)*Z**3

        #mass density of baryons only (in kg/m^3 proper)
        rho_b = self.basic.rho_crit()*self.config.Om_b*Z**3 

        rp = self.r_t(xe,Tk,Tx,v_bx)
        
        up = self.u_t(xe,Tk,Tx)
        term1 = cE**4*2*self.config.mx*rho_b*self.config.sigma0*np.exp(-rp**2/2)*(temp_diff)/((self.config.mx+self.basic.mu(xe)*mP)**2*np.sqrt(2*np.pi)*up**3)
        term2 = 1/kB*rho_b/(rho_x+rho_b)*self.mu_bx(xe)*v_bx*self.Drag(Z,xe,Tk,Tx,v_bx)

        return 2/(3*self.basic.Hubble(Z))*(term1+term2)
    
    #End of functions related to IDM.