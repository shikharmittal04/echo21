"""
``config``
==========
This module contains class config.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from colossus.cosmology import cosmology
import warnings
from .const import *
from .utils import _ensure_scalar_dict

warnings.filterwarnings('ignore')

class config():
    '''
    Function names starting with 'basic_cosmo' include the basic :math:`\\Lambda` CDM-cosmology-related functions, such as Hubble function, CMB temperature, etc.

    Function names starting with 'recomb' include recombination-physics-related functions.

    Function names starting with 'heating' include all the heating terms. All the terms are in the form of :math:`\\mathrm{d}T_{\\mathrm{k}}/\\mathrm{d}\\ln(a)` and hence, in units of temperature. (:math:`a` is the scale factor.)
    
    Function names starting with 'hyfi' include all the functions related to the computation of 21-cm signal. These are
    :math:`\\kappa_{\\mathrm{HH}}, \\kappa_{\\mathrm{eH}}, x_{\\mathrm{k}}, x_{\\mathrm{Ly}}, T_{\\mathrm{s}}` and :math:`T_{21}`.

    Arguments
    ~~~~~~~~~
    params: dict
        A dictionary containing all the cosmological and astrophysical parameters.
    
    Methods
    ~~~~~~~
    '''
    def __init__(self,params=None, dm_model='CDM'):
        '''
        
        '''
        params = {} if params is None else params
        params = _ensure_scalar_dict(params)

        ############################################################################
        self.Ho = params.get('Ho', 67.4)
        self.Om_m = params.get('Om_m', 0.315)
        self.Om_b = params.get('Om_b', 0.049)
        self.sig8 = params.get('sig8', 0.844)
        self.ns = params.get('ns', 0.965)
        self.Tcmbo = params.get('Tcmbo', 2.726)
        self.Yp = params.get('Yp', 0.245)

        self.fLy = params.get('fLy',1.0)
        self.sLy = params.get('sLy',2.64)
        self.fX = params.get('fX',1.0)
        self.wX = params.get('wX',1.5)
        self.fesc = params.get('fesc',0.01)

        ############################################################################
        #Setting up cosmology for COLOSSUS package
        self.cosmo_par = {'flat': True, 'H0': self.Ho, 'Om0': self.Om_m, 'Ob0': self.Om_b, 'sigma8': self.sig8, 'ns': self.ns,'relspecies': True,'Tcmb0': self.Tcmbo}
        self.my_cosmo = cosmology.setCosmology('cosmo_par', self.cosmo_par, persistence = 'r')
        self.h100 = self.Ho/100
        
        ############################################################################
        #Setting up the star formation rate density related parameters and functions
        self.sfrd_type = params.get('type','phy')

        # config only records the SFRD-related *parameters*. The choice of which
        # f_coll / sfrd implementation to use (and building the f_coll spline) is
        # the halo class's responsibility -- see echo21.funcs.halo.halo.__init__,
        # which reads these flags.
        if self.sfrd_type == 'phy':
            params = {**phy_sfrd_default_model, **params}
            self.hmf = params['hmf']
            self.mdef = params['mdef']
            self.Tmin_vir = params['Tmin_vir']

        elif self.sfrd_type == 'semi-emp':
            params = {**semi_emp_sfrd_default_model, **params}
            self.hmf = params['hmf']
            self.mdef = params['mdef']
            self.Tmin_vir = params['Tmin_vir']
            self.tstar = params['tstar']

        elif self.sfrd_type == 'emp':
            params = {**emp_sfrd_default_model, **params}
            self.a_sfrd = params['a']
            self.b_sfrd = 4.0#params['b']

        else:
            raise ValueError(f"Unknown SFRD type: {self.sfrd_type}")
        
        ############################################################################
        # NOTE: the IGM equation RHS functions and f_coll splines that depend on
        # halo/solver physics now live on their respective concern classes
        # (echo21.funcs.solver, echo21.funcs.halo). config only records the
        # parameters they need (dm_model, mx, sigma0, ...).
        self.dm_model = dm_model
        if dm_model == 'IDM':
            self.mx_gev = params['mx_gev']
            sigma45 = params['sigma45']
            self.mx = self.mx_gev*GeV2kg #Now mx is in kg
            self.sigma0 = sigma45*sig_ten45m2   #Now sigma0 is in m^2

        return None

#End of class config.
#======================================================================================================