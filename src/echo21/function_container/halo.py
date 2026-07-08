import numpy as np
import scipy.special as scsp
from scipy.interpolate import CubicSpline
import scipy.integrate as scint
from colossus.lss import peaks
from colossus.lss import mass_function
import os, tempfile
try:
    import classy
except ImportError:
    print("")
    
from ..const import *

def _get_As_for_sig8(class_set):
    '''
    For the given cosmological parameters (including sigma8), compute the primordial power spectrum amplitude.

    Arguments
    ---------
    class_set: dict
        A dictionary of cosmological parameters including user provided sigma8
    
    Returns
    -------
    
    float
        A_s

    '''
    class_obj = classy.Class()
    class_obj.set(class_set)
    class_obj.compute()

    A_s = class_obj.get_current_derived_parameters(['A_s'])['A_s']
    class_obj.struct_cleanup()
    class_obj.empty()

    return A_s

class halo():
    def __init__(self, config, basic):
        self.config = config
        self.basic = basic

        # Select the SFRD implementation from the sfrd_type flag on config.
        _sfrd_by_type = {
            'phy': self._sfrd_phy,
            'semi-emp': self._sfrd_semi_emp,
            'emp': self._sfrd_emp,
        }
        self._sfrd = _sfrd_by_type[config.sfrd_type]

        # Select the collapse-fraction implementation. IDM (or any non-press74
        # HMF) needs a precomputed spline; CDM press74 has a closed form.
        # The 'emp' SFRD does not use f_coll at all, so guard on sfrd_type.
        if config.sfrd_type in ('phy', 'semi-emp'):
            if config.dm_model == 'IDM':
                self._f_coll_spline = self._build_idm_fcoll_spline()
                self._f_coll = self._f_coll_not_cdm_press74
            elif config.hmf == 'press74':
                self._f_coll = self._f_coll_cdm_press74
            else:
                self._f_coll_spline = self._build_cdm_fcoll_spline()
                self._f_coll = self._f_coll_not_cdm_press74

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
        M_by_h = M*self.config.h100 #M is in solar mass units and M_by_h is in units of solar mass/h.
        return self.config.h100**3*mass_function.massFunction(M_by_h, Z-1, q_in='M', q_out='dndlnM', mdef = self.config.mdef, model = self.config.hmf)

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
        
        return 1e8*self.config.Om_m**(-0.5)*(10/Z*0.6/1.22*self.config.Tmin_vir/1.98e4)**1.5

    def _f_coll_cdm_press74(self, Z):
        '''
        Collapse fraction only for the Press-Schechter HMF and CDM.
        '''
        return scsp.erfc(peaks.peakHeight(self.m_min(Z),Z-1)/np.sqrt(2))

    def _f_coll_not_cdm_press74(self, Z):
        '''
        Collapse fraction for all other cases. E.g. CDM Tinker08, IDM Press74, or IDM Tinker08.
        '''
        scalar = np.isscalar(Z)
        Z = np.atleast_1d(np.asarray(Z, dtype=float))
        result = np.where(Z <= Z_STAR, np.maximum(self._f_coll_spline(Z), 0.0), 0.0)
        return float(result[0]) if scalar else result
    
    def _build_cdm_fcoll_spline(self, n_points=60):
        '''
        Precompute the collapse fraction on a redshift grid and return a CubicSpline.
        Called once per funcs() instantiation for non-press74 HMFs, replacing the
        per-ODE-step numerical integration with a cheap spline evaluation.

        Arguments
        ---------
        funcs_obj : funcs
            An initialised funcs instance whose cosmology and HMF are already set up.

        n_points : int
            Number of redshift grid points. Default 60.

        Returns
        -------
        CubicSpline
            Spline of f_coll(Z) over Z in [1, Z_STAR], with Z = 1+z.
        '''
        Z_grid = np.linspace(1, Z_STAR, n_points)
        f_grid = np.zeros(n_points)
        norm = Msolar_by_Mpc3_to_kg_by_m3 / (self.config.Om_m * self.basic.rho_crit())
        for i, Zv in enumerate(Z_grid):
            M_space = np.logspace(np.log10(self.m_min(Zv) / self.config.h100), 16, 200)
            f_grid[i] = scint.simpson(self.dndlnM(M=M_space, Z=Zv), x=M_space)
        f_grid *= norm

        return CubicSpline(Z_grid, f_grid)

    def _build_idm_fcoll_spline(self, n_points=60):
        '''
        Collapse fraction for Coulomb-like IDM HMFs.

        Arguments
        ---------
        n_points : int
            Number of redshift grid points. Default 60.

        Returns
        -------
        CubicSpline
            Spline of f_coll(Z) over Z in [1, Z_STAR], with `Z` :math:`= 1+z`.
        '''
        Z_grid = np.linspace(1, Z_STAR, n_points)
        f_grid = np.zeros(n_points)
        k = np.logspace(-6,3,50) #in h/Mpc

        #---------------------------------------------------------------------------------
        #First get As for the given cosmological parameters assuming CDM
        class_set_cdm = {'h':self.config.h100, 'Omega_b':self.config.Om_b, 'Omega_cdm':self.config.Om_m-self.config.Om_b, 'Omega_dmeff':0.0, 'YHe':self.config.Yp, 'n_s':self.config.ns, 'sigma8': self.config.sig8, 'output':'mPk','P_k_max_1/Mpc':1, 'z_max_pk':0.1}

        class_set_idm = {'h':self.config.h100, 'Omega_b':self.config.Om_b, 'Omega_cdm':0.0, 'Omega_dmeff':self.config.Om_m-self.config.Om_b, 'YHe':self.config.Yp, 'n_s':self.config.ns, 'm_dmeff': self.config.mx_gev, 'N_dmeff': 1, 'sigma_dmeff': 1e4*self.config.sigma0, 'npow_dmeff':-4, 'dmeff_target':'baryon', 'Vrel_dmeff': 30, 'output':'mPk','P_k_max_1/Mpc':k.max(), 'z_max_pk':0.1}

        class_set = class_set_idm | {'A_s':_get_As_for_sig8(class_set_cdm)}

        #Initialize CLASS for IDM power spectrum generation.
        class_obj = classy.Class()
        class_obj.set(class_set)
        class_obj.compute()
        
        #Now run CLASS and generate matter power spectrum. This will be fed to COLOSSUS.
        Pk_0 = np.array([self.config.h100**3*class_obj.pk(self.config.h100*kk,0.0) for kk in k]) #in (Mpc/h)**3

        fd, pk_path = tempfile.mkstemp(prefix='Pk_idm_', suffix='.txt')   # unique per process
        os.close(fd)
        np.savetxt(pk_path, np.vstack((np.log10(k),np.log10(Pk_0))).T)
        
        #CLASS's job is done.
        #---------------------------------------------------------------------------------
        #Now compute the collapse fraction by feeding CLASS's matter power spectrum into COLOSSUS.
        ps_idm_dict = dict(model = f'idm_m{self.config.mx_gev:.4e}_s{self.config.sigma0:.4e}', path = pk_path)


        norm = Msolar_by_Mpc3_to_kg_by_m3 / (self.config.Om_m * self.basic.rho_crit())

        for i, Zv in enumerate(Z_grid):
            M_space = np.logspace(np.log10(self.m_min(Zv) / self.config.h100), 16, 200) #solar mass units
            dn_dlnM = self.config.h100**3*mass_function.massFunction(self.config.h100*M_space, Zv-1, q_in='M',
            q_out='dndlnM', mdef = self.config.mdef, model = self.config.hmf, ps_args = ps_idm_dict)
            f_grid[i] = scint.simpson(dn_dlnM, x=M_space)

        f_grid *= norm

        # cleanup
        os.remove(pk_path)

        return CubicSpline(Z_grid, f_grid)

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
        mysfrd = -Z*fstar*self.config.Om_b*self.basic.rho_crit()*self.dfcoll_dz(Z)*self.basic.Hubble(Z)
        return mysfrd if mysfrd.size > 1 else mysfrd[0]
    
    def _sfrd_semi_emp(self,Z):
        Z = np.atleast_1d(Z)
        mysfrd = fstar*self.config.Om_b*self.basic.rho_crit()*self.basic.Hubble(Z)*self.f_coll(Z)/self.config.tstar
        return mysfrd if mysfrd.size > 1 else mysfrd[0]

    def _sfrd_emp(self,Z):
        Z = np.atleast_1d(Z)
        Zcut = 1 + self.config.b_sfrd

        lowz = 0.015 * Z**2.73 / (1 + (Z / 3)**6.2)
        highz = 0.015 * Zcut**2.73 / (1 + (Zcut / 3)**6.2) * 10**(self.config.a_sfrd * (Zcut - Z))

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