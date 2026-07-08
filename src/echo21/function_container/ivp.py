import numpy as np
import scipy.integrate as scint

from ..const import *

class ivp():
    '''
    Class of all the functions required to solve the initial value problem of bulk of IGM. EoR related functions are in the class :class:`eor`.

    Methods
    ^^^^^^^
    '''
    def __init__(self, config, basic, recomb, halo, idm, heating):
        self.config = config
        self.basic = basic
        self.recomb = recomb
        self.halo = halo
        self.idm = idm
        self.heating = heating

        # Select the RHS functions for the ODE system by dark-matter model.
        if config.dm_model == 'IDM':
            self.igm_eqns_da = self._igm_eqns_idm_da
            self.igm_eqns_cd = self._igm_eqns_idm_cd
        else:
            self.igm_eqns_da = self._igm_eqns_cdm_da
            self.igm_eqns_cd = self._igm_eqns_cdm_cd

    def _logratio_to_temp(self, Z, y):
        '''
        Given the evolved log-ratio variable :math:`y = \\ln(T_{\\mathrm{k}}/T_{\\gamma})`,
        compute :math:`T_{\\mathrm{k}}`.

        Arguments
        ---------
        
        Z : float
            redshift, :math:`1+z`

        y : float
            :math:`y = \\ln(T_{\\mathrm{k}}/T_{\\gamma})`
        
        Returns
        -------
            :math:`T_{\\mathrm{k}} = e^{y}\\,T_{\\gamma}(z)`
        '''
        Tgamma = self.basic.Tcmb(Z)
        Tk = np.exp(y)*Tgamma
        return Tk

    def initial_conditions(self):
        '''
        Initial conditions for the IGM equations at :math:`z=1500`. For CDM, we need electron fraction and gas kinetic temperature. For IDM, we also need DM temperature and relative velocity of DM and baryons.

        Also, note that for gas temperature it is a transformed variable. Instead of :math:`T_{\\mathrm{k}}` we evolve :math:`y = \\ln(T_{\\mathrm{k}}/T_{\\gamma})`. At :math:`z=1500`, :math:`T_{\\mathrm{k}}=T_{\\gamma}` so the initial value is :math:`y=0`.
        
        For IDM case instead of velocity we have :math:`\\ln(v_{\\mathrm{b}\\chi})`.

        Arguments
        ---------
        
        Z : float
            1 + z, dimensionless.
        
        Returns
        -------
        tuple
            Initial conditions. For CDM, the tuple is (xe_init, yT_init). For IDM, the tuple is (xe_init, yT_init, Tx_init, ln_vbx_init).
        '''
        Tgamma_init = self.basic.Tcmb(Z_START)
        xe_init = self.recomb.Saha_xe(Z_START,Tgamma_init)
        yT_init = 0.0
        Tx_init = 0.0
        ln_vbx_init = np.log(43500)
        ic = (xe_init,yT_init,Tx_init,ln_vbx_init) if self.config.dm_model == 'IDM' else (xe_init,yT_init)
        return ic

    def _igm_eqns_cdm_da(self, Z, V):
        '''
        Differential equations for the IGM in the CDM model during the dark ages phase.
        The thermal state variable is y = ln(Tk/Tgamma); eq2 returns d(y)/dlnZ.
        '''
        xe, yT = V
        Tk = self._logratio_to_temp(Z, yT)

        Tgamma = self.basic.Tcmb(Z)
        xe = np.clip(xe, 0.0, 1.0)  # Ensure xe stays within physical bounds

        eq1 = 1/self.basic.Hubble(Z)*self.recomb.Peebles_C(Z,xe,Tgamma)*(xe**2*self.basic.nH(Z)*self.recomb.alpha(Tk)-self.recomb.beta(Tgamma)*(1-xe)*np.exp(-Ea/(kB*Tgamma)))

        eq2 = 1 - eq1/(1+self.basic.xHe()+xe) - 1/Tk*self.heating.Ecomp(Z,xe,Tk)

        return np.array([eq1,eq2])

    def _igm_eqns_cdm_cd(self, Z, V):
        '''
        Differential equations for the IGM in the CDM model during the cosmic dawn phase.
        For cosmic dawn we keep the original variable, Tk.
        '''
        xe, Tk = V

        Tgamma = self.basic.Tcmb(Z)

        if xe<0.99:
            eq1 = 1/self.basic.Hubble(Z)*self.recomb.Peebles_C(Z,xe,Tgamma)*(xe**2*self.basic.nH(Z)*self.recomb.alpha(Tk)-self.recomb.beta(Tgamma)*(1-xe)*np.exp(-Ea/(kB*Tgamma)))-1/self.basic.Hubble(Z)*self.heating.Gamma_x(Z,xe)*(1-xe)
        else:
            eq1 = 0.0

        THETA_k = self.heating.Ecomp(Z,xe,Tk) + self.heating.Ex(Z,xe) + self.heating.Elya(Z,xe,Tk)
        eq2 = 2*Tk - Tk*eq1/(1+self.basic.xHe()+xe) - THETA_k
        
        return np.array([eq1,eq2])

    def _igm_eqns_idm_da(self, Z, V):
        '''
        Differential equations for the IGM in the IDM model during the dark ages phase.
        The thermal state variable is y = ln(Tk/Tgamma); eq2 returns d(y)/dlnZ.
        '''
        xe, yT, Tx, ln_v_bx = V
        Tk = self._logratio_to_temp(Z, yT)

        v_bx = np.exp(np.clip(ln_v_bx, -30, None))
        xe = np.clip(xe, 0.0, 1.0)
        
        Tgamma = self.basic.Tcmb(Z)

        eq1 = 1/self.basic.Hubble(Z)*self.recomb.Peebles_C(Z,xe,Tgamma)*(xe**2*self.basic.nH(Z)*self.recomb.alpha(Tk)-self.recomb.beta(Tgamma)*(1-xe)*np.exp(-Ea/(kB*Tgamma)))

        THETA_k = self.heating.Ecomp(Z,xe,Tk) + self.idm.Ex2b(Z,xe,Tk,Tx,v_bx)
        THETA_x = self.idm.Eb2x(Z,xe,Tk,Tx,v_bx)
        
        eq2 = 1 - eq1/(1+self.basic.xHe()+xe) - 1/Tk*THETA_k

        eq3 = 2*Tx-THETA_x
        
        eq4 = 1 + 1/v_bx*self.idm.Drag(Z,xe,Tk,Tx,v_bx)/self.basic.Hubble(Z)
        
        return np.array([eq1,eq2,eq3,eq4])

    def _igm_eqns_idm_cd(self, Z, V):
        '''
        Differential equations for the IGM in the IDM model during the cosmic dawn phase.
        '''
        xe, Tk, Tx, ln_v_bx = V
        
        v_bx = np.exp(np.clip(ln_v_bx, -30, None))

        Tgamma = self.basic.Tcmb(Z)

        if xe<0.99:
            eq1 = 1/self.basic.Hubble(Z)*self.recomb.Peebles_C(Z,xe,Tgamma)*(xe**2*self.basic.nH(Z)*self.recomb.alpha(Tk)-self.recomb.beta(Tgamma)*(1-xe)*np.exp(-Ea/(kB*Tgamma)))-1/self.basic.Hubble(Z)*self.heating.Gamma_x(Z,xe)*(1-xe)
        else:
            eq1 = 0.0

        THETA_k = self.heating.Ecomp(Z,xe,Tk) + self.heating.Ex(Z,xe) + self.heating.Elya(Z,xe,Tk) + self.idm.Ex2b(Z,xe,Tk,Tx,v_bx)
        THETA_x = self.idm.Eb2x(Z,xe,Tk,Tx,v_bx)

        eq2 = 2*Tk  - Tk*eq1/(1+self.basic.xHe()+xe) - THETA_k

        eq3 = 2*Tx-THETA_x
        
        eq4 = 1 + 1/v_bx*self.idm.Drag(Z,xe,Tk,Tx,v_bx)/self.basic.Hubble(Z)
        
        return np.array([eq1,eq2,eq3,eq4])


    def igm_solver(self, Z_solver, *initial_conditions, eqns_func):
        '''
        This function solves the coupled IGM differential equations. In case of CDM it is just electron fraction and gas temperature. When IDM is involed DM temperature and relative DM-baryon velocity is also solved.
        Note the following two points:
        
            1. For thermal evolution, I don't solve for :math:`T_{\\mathrm{k}}` but rather :math:`y = \\ln(T_{\\mathrm{k}}/T_{\\gamma})`. Recover the temperature with :meth:`_logratio_to_temp`, i.e. :math:`T_{\\mathrm{k}} = e^{y}\\,T_{\\gamma}`.
        
            2. In case of IDM, the last value of the solution array is :math:`\\ln v_{\\mathrm{b}\\chi}` and not :math:`v_{\\mathrm{b}\\chi}` itself.

        Arguments
        ---------
        Z_solver: array
            Redshift array (decreasing) over which to solve. Use Z_DA for dark ages, Z_CD for cosmic dawn, or Z_default for the full range.

        initial_conditions: tuple
            Initial conditions for the ODE solver. For CDM, the tuple is (xe_init, yT_init). For IDM, the tuple is (xe_init, yT_init, Tx_init, ln_vbx_init). For DA the second variable is yT but for CD-EoR it is Tk. Use :func:`initial_conditions()` to get the initial conditions when the starting redshift is :math:`z=1500`.
        
        eqns_func: callable
            The RHS function to pass to the ODE solver. Either dark ages or cosmic dawn.
        
        Returns
        -------
        array
            :math:`x_{\\mathrm{e}}`, :math:`y=\\ln(T_{\\mathrm{k}}/T_{\\gamma})` or :math:`T_{\\mathrm{k}}`, :math:`T_{\\chi}`, :math:`\\ln v_{\\mathrm{b}\\chi}`
        '''
        local_Z_START = Z_solver[0]
        local_Z_END   = Z_solver[-1]

        Sol = scint.solve_ivp(
            lambda lna, Var: -eqns_func(np.exp(-lna), Var),
            [-np.log(local_Z_START), -np.log(local_Z_END)],
            list(initial_conditions),
            method='Radau',
            t_eval= -np.log(Z_solver),
            rtol=1e-4, atol=1e-7
        )
        
        results = [y for y in Sol.y]
        
        return results