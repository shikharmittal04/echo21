"""
``pipeline``
============
This module contains the class pipeline.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import pandas as pd
import numpy as np
from mpi4py import MPI
from mpi4py.util import pkl5
import time
import warnings
from tqdm import tqdm

from .const import *
from .funcs import funcs
from .single_set_solver import *
from .utils import *
#--------------------------------------------------------------------------------------------

class pipeline():
    '''
    This class runs the cosmic history solver and produces the global signal, globally-averaged nuetral hydrogen fraction, optical depth and the corresponding redshifts. There are three inputs required for a complete specification -- cosmological parameters, astrophysical parameter, and star formation related parameters. They are supplied through arguments, ``cosmo``, ``astro``, and ``sfrd``, respectively. The notation for the parameters is as follows. All of these need to be dictionaries. For example:

    .. code:: python
        
        cosmo = {'Ho':67.4, 'Om_m':0.315, 'Om_b':0.049, 'sig8':0.811, 'ns':0.965, 'Tcmbo':2.725, 'Yp':0.245},
        astro = {'fLy':1, 'sLy' : 2.64, 'fX':1, 'wX':1.5, 'fesc':0.01},
        sfrd = {'type':'phy', 'hmf':'press74', 'mdef':'fof', 'Tmin_vir':1e4}
    
    Arguments
    ---------
    cosmo: dict
        Dictionary of cosmological parameters. They are:

        Ho : float, optional
            Hubble parameter today in units of :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`. Default value ``67.4``.
        
        Om_m : float, optional
            Relative matter density. Default value ``0.315``.
        
        Om_b : float, optional
            Relative baryon density. Default value ``0.049``.            
        
        sig8 : float, optional
            Amplitude of density fluctuations. Default value ``0.811``.
        
        ns : float, optional
            Spectral index of the primordial scalar spectrum. Default value ``0.965``. 

        Tcmbo : float, optional
            CMB temperature today in kelvin. Default value ``2.725``.
        
        Yp : float, optional
            Primordial helium fraction by mass. Default value ``0.245``.
    
    astro: dict
        Dictionary of cosmological parameters. They are:

        fLy : float, optional
            :math:`f_{\\mathrm{Ly}}`, a dimensionless parameter which controls the emissivity of the Lyman series photons. Default value ``1.0``.
        
        sLy : float, optional
            :math:`s`, spectral index of Lyman series SED, when expressed as :math:`\\epsilon\\propto E^{-s}`. :math:`\\epsilon` is energy emitted per unit energy range and per unit volume. Default value ``2.64``.

        fX : float, optional
            :math:`f_{\\mathrm{X}}`, a dimensionless parameter which controls the emissivity of the X-ray photons. Default value ``1.0``.
        
        wX : float, optional
            :math:`w`, spectral index of X-ray SED, when expressed as :math:`\\epsilon\\propto E^{-w}`. :math:`\\epsilon` is energy emitted per unit energy range and per unit volume. Default value ``1.5``.

        fesc : float, optional
            :math:`f_{\\mathrm{esc}}`, a dimensionless parameter which controls the escape fraction of the ionizing photons. Default value ``0.01``.

    sfrd : dict
        This should be a dictionary containing all the details of SFRD.
        
        type : str, optional
            Available types are 'phy' (default), 'semi-emp', and 'emp', for a physically-motivated, semi-empirical, and an empiricaly-motivated SFRD, respectively.

        hmf : str, optional
            HMF model to use. Default value ``press74``. Other commonly used HMFs are
            
            - sheth99 (for Sheth & Tormen 1999)
            
            - tinker08 (for Tinker et al 2008)

        For the full list see `colossus <https://bdiemer.bitbucket.io/colossus/lss_mass_function.html#mass-function-models>`__ page.

        mdef: str, optional
            Definition for halo mass. Default is ``fof``. For most HMFs such as Press-Schechter or Sheth-Tormen, friends-of-friends (``fof``) algorithm is used. For Tinker08, it is an integer times mean matter density (``<int>m``). See the ``colossus`` documentation for definition `page <https://bdiemer.bitbucket.io/colossus/halo_mass.html>`_
            
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.

        t_star : float, optional
            Star formation timescale in units of the Hubble time. Default value ``0.5``. (This is only relevant for the semi-empirical SFRD model.)

        a : float, optional
            Power law index for the SFRD in the empirical model. Default value ``0.257``. (This is only relevant for the empirical SFRD model.)
        
    grid_on: bool
        Whether to generate a grid of parameter combinations. Default is False, i.e., parameters are varied one at a time. In this case all varied parameters should have the same number of values. If True, then all possible combinations of the parameters will be generated.
    
    Methods
    ~~~~~~~
    '''
    def __init__(self,cosmo=None,astro= None, sfrd=None, grid_on=False, path='echo21_outputs/'):

        if cosmo is None:
            cosmo = {'Ho': 67.4, 'Om_m': 0.315, 'Om_b': 0.049, 'sig8': 0.811, 'ns': 0.965,'Tcmbo': 2.725, 'Yp': 0.245}
        if astro is None:
            astro = {'fLy': 1, 'sLy': 2.64, 'fX': 1, 'wX': 1.5, 'fesc': 0.01}
        if sfrd is None:
            sfrd = {'type': 'phy', 'hmf': 'press74', 'mdef': 'fof', 'Tmin_vir': 1e4}

        self.grid_on = grid_on

        self.comm = pkl5.Intracomm(MPI.COMM_WORLD)
        self.cpu_ind = self.comm.Get_rank()
        self.n_cpu = self.comm.Get_size()

        self.dm_model = 'IDM' if {'mx_gev', 'sigma45'} & cosmo.keys() else 'CDM'

        self.cosmo = _ensure_array_dict(cosmo)
        self.astro = _ensure_array_dict(astro)
        # sfrd contains strings → ignore them
        self.sfrd = _ensure_array_dict(sfrd, ignore_keys=['type','hmf','mdef'])

        cosmo_var, cosmo_fixed = _split_params(self.cosmo)
        astro_var, astro_fixed = _split_params(self.astro)
        sfrd_var, sfrd_fixed   = _split_params(self.sfrd)

        self.var_params = {**cosmo_var, **astro_var, **sfrd_var}
        self.fixed_params = {**cosmo_fixed, **astro_fixed, **sfrd_fixed}

        #param_names is a list of the names of the varying parameters.
        self.param_names = list(self.var_params.keys())
        #param_arrays is a list of the corresponding arrays of values for those parameters.
        self.param_arrays = [self.var_params[k] for k in self.param_names]

        #---------------------------------------------------------------------------------
        if grid_on:
            self.shape = tuple(len(a) for a in self.param_arrays)
            self.N_models = np.prod(self.shape)
            self.get_index = _grid_on_index
        else:
            #Before proceeding check if all the arrays of varying parameters have the same length. If not, then terminate and ask user to correct the input.
            if len(set(map(len, self.param_arrays))) > 1:
                raise ValueError('When grid_on=False, all varying parameters should have the same number of values.')
            self.N_models = len(self.param_arrays[0]) if self.param_arrays else 1

            self.get_index = _grid_off_index

        cosmo_varying = any(np.size(v) > 1 for v in (cosmo_var).values())
        astro_varying = any(np.size(v) > 1 for v in {**astro_var, **sfrd_var}.values())

        #---------------------------------------------------------------------------------
        if not cosmo_varying and not astro_varying:
            self.run_type='single'
            self.message = 'both cosmological and astrophysical parameters are fixed.\n'
            self.initial_conditions = (None , None, None, None)
            self.simulator = dark_ages_to_today

        elif not cosmo_varying and astro_varying:
            obj_dark_ages = funcs(self.fixed_params, dm_model=self.dm_model)
            ic_da = obj_dark_ages.initial_conditions()

            sol_da = obj_dark_ages.igm_solver(Z_da, *ic_da, eqns_func=obj_dark_ages.igm_eqns_da)

            #in dark ages we solver for the transformed gas temperature. So we need to convert to physical temperature
            sol_da[1] = obj_dark_ages._logratio_to_temp(Z_da, sol_da[1])

            #initial conditions for cosmic dawn solver
            self.initial_conditions = tuple(x[-1] for x in sol_da)

            self.run_type='astro'
            self.message = 'cosmological parameters are fixed. Astrophysical parameters are varied.'
            
            self.simulator = cosmic_dawn_beyond
        
        else:
            self.run_type='else'
            self.message = 'cosmological parameters are varied.\n'       
            self.initial_conditions = (None , None, None, None)
            self.simulator = dark_ages_to_today
        #---------------------------------------------------------------------------------

        if self.run_type!='single' and self.n_cpu==1:
            raise ValueError('Please use at least 2 CPUs.')
        #---------------------------------------------------------------------------------

        #Create an output folder where all results will be saved.
        self.path=path
        _create_output_dir(self)
        
        #--------------------------------------------------------------------------------
        #Finally, the redshifts and magnitude should also be pipeline attributes.
        self.one_plus_z = Z_default
        self.one_plus_z_cd = Z_cd
        self.MUV = MUV_default
            
        return None
    
    def run_simulation(self):
        '''
        This is the main function which runs the ECHO21 simulation and saves the outputs. 
        ''' 

        if self.run_type=='single':
        #Cosmological and astrophysical parameters are fixed.
            if self.cpu_ind==0:

                _print_banner()
                print(f'Dark matter type: {self.dm_model}')
                print('\nSimulation type: ',self.message)
                print('Generating',self.N_models,'model ...')

                st = time.perf_counter()
                result = self.simulator(self.fixed_params, *self.initial_conditions, dm_model=self.dm_model)

                #create empty pandas dataframe for consistency with the other cases. 
                self.params_df = pd.DataFrame([])
                if self.dm_model == 'CDM':
                    self.xe, self.Q_Hii, self.xHI, self.Tk, self.Ts, self.T21, self.tau, self.UVLF = result
                else:
                    self.xe, self.Q_Hii, self.xHI, self.Tk, self.Ts, self.T21, self.tau, self.UVLF, self.Tx, self.v_bx = result
                _save_results(self)

                print('\033[32m\nOutputs saved into folder:',self.path,'\033[00m')
                
                et = time.perf_counter()
                # get the execution time
                elapsed_time = et - st
                print('\nExecution time: %.2f seconds' %elapsed_time)

                #========================================================
                #Writing to a summary file
                myfile = _write_summary(self, elapsed_time=elapsed_time)

                _save_pipeline(self)
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
                return None

        #=========================================================================
        else:
            partial_results = []
            partial_params = []
            n_failed = 0
            failed_params = []
            if self.cpu_ind==0:
                #Master CPU
                _print_banner()
                print(f'Dark matter type: {self.dm_model}')
                print('\nSimulation type: ',self.message)
                print('Grid on = ',self.grid_on)
                print('\nGenerating',self.N_models,'models ...')
                st = time.perf_counter()
                done = 0
                pbar = tqdm(total=self.N_models, desc="Processing models", ncols=100)

                while done < self.N_models:
                    status = MPI.Status()
                    progress = self.comm.recv(source=MPI.ANY_SOURCE, tag=77, status=status)
                    done += progress
                    pbar.update(progress)
                pbar.close()
            else:
                #Worker CPU
                for idx in range(self.cpu_ind-1, self.N_models, self.n_cpu-1):
                    all_params_dict, varying_params_only = self.get_index(self, idx)

                    try:
                        result = self.simulator(all_params_dict, *self.initial_conditions, dm_model=self.dm_model)
                    except Exception:
                        n_failed += 1
                        failed_params.append(varying_params_only)
                        self.comm.send(1, dest=0, tag=77)
                        continue

                    partial_params.append(varying_params_only)
                    partial_results.append(result)
                    
                    self.comm.send(1, dest=0, tag=77)

        
            # A single collective is used here on purpose. Issuing several lowercase
            # collectives in a row over a pkl5 communicator -- one of them gathering
            # ragged/empty 'failed_params' lists -- can wedge. Bundling everything into
            # one dict per rank keeps the payload type uniform across ranks (always a
            # dict) and gives a single, robust synchronisation point.
            payload = {
                'results': partial_results,
                'params': partial_params,
                'failed_params': failed_params,
                'n_failed': n_failed,
            }
            gathered = self.comm.gather(payload, root=0)

            if self.cpu_ind == 0:
                # Flatten across ranks (rank order preserved by gather)
                gathered_results       = [r for chunk in gathered for r in chunk['results']]
                gathered_params        = [p for chunk in gathered for p in chunk['params']]
                gathered_failed_params = [fp for chunk in gathered for fp in chunk['failed_params']]
                total_failed      = sum(chunk['n_failed'] for chunk in gathered)
                n_succeeded = len(gathered_results)

                if n_succeeded == 0:
                    warnings.warn('All models failed; nothing to save.')
                else:
                    self.params_df = pd.DataFrame(gathered_params)
                    #If there are more outputs in future, then the unpacking below needs to be changed accordingly.
                    self.xe    = np.vstack([r[0] for r in gathered_results])
                    self.Q_Hii = np.vstack([r[1] for r in gathered_results])
                    self.xHI   = np.vstack([r[2] for r in gathered_results])
                    self.Tk    = np.vstack([r[3] for r in gathered_results])
                    self.Ts    = np.vstack([r[4] for r in gathered_results])
                    self.T21   = np.vstack([r[5] for r in gathered_results])
                    self.tau   = np.concatenate([r[6] for r in gathered_results])
                    self.UVLF  = np.array([r[7] for r in gathered_results])
                    if self.dm_model == 'IDM':
                        self.Tx    = np.vstack([r[8] for r in gathered_results])
                        self.v_bx  = np.vstack([r[9] for r in gathered_results])

                    _save_results(self, total_failed=total_failed, gathered_failed_params=gathered_failed_params, n_succeeded=n_succeeded)                        

                    print('\033[32m\nOutputs saved into folder:',self.path,'\033[00m')
                    
                    et = time.perf_counter()
                    # get the execution time
                    elapsed_time = et - st
                    print('\nProcessing time: %.2f seconds' %elapsed_time)

                #========================================================
                #Writing to a summary file

                myfile = _write_summary(self, elapsed_time=elapsed_time)
                myfile.write('\n{} models generated ({} succeeded, {} failed)'.format(self.N_models, n_succeeded, total_failed))
                myfile.write('\nNumber of CPU(s) = {}'.format(self.n_cpu))
                myfile.write('\n')
                myfile.close()


                _save_pipeline(self)
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
            return None
    
    #End of function run_simulation               
#End of class pipeline
#=======================================================================================================