"""
echopipeline
============
This module contains the class pipeline.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import pandas as pd
import numpy as np
from mpi4py import MPI
from mpi4py.util import pkl5
import os, sys, time
from scipy.interpolate import CubicSpline
from time import localtime, strftime
from tqdm import tqdm

from .const import Zstar, Z_start, Z_end, Z_default, Z_da, Z_cd, flipped_Z_default
from .echofuncs import funcs
from .single_set_solver import *
from .misc import *

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
    
    Z_eval: float
        Array of :math:`1+z` where you want to compute the quantities.
        
    Methods
    ~~~~~~~
    '''
    def __init__(self,cosmo=None,astro= None, sfrd=None,Z_eval=None,path='echo21_outputs/'):

        if cosmo is None:
            cosmo = {'Ho': 67.4, 'Om_m': 0.315, 'Om_b': 0.049, 'sig8': 0.811, 'ns': 0.965,'Tcmbo': 2.725, 'Yp': 0.245}
        if astro is None:
            astro = {'fLy': 1, 'sLy': 2.64, 'fX': 1, 'wX': 1.5, 'fesc': 0.01}
        if sfrd is None:
            sfrd = {'type': 'phy', 'hmf': 'press74', 'mdef': 'fof', 'Tmin_vir': 1e4}

        self.comm = pkl5.Intracomm(MPI.COMM_WORLD)
        self.cpu_ind = self.comm.Get_rank()
        self.n_cpu = self.comm.Get_size()
        
        self.Z_eval = Z_eval

        self.cosmo = ensure_array_dict(cosmo)
        self.astro = ensure_array_dict(astro)
        # sfrd contains strings → ignore them
        self.sfrd = ensure_array_dict(sfrd, ignore_keys=['type','hmf','mdef'])

        cosmo_var, cosmo_fixed = split_params(self.cosmo)
        astro_var, astro_fixed = split_params(self.astro)
        sfrd_var, sfrd_fixed   = split_params(self.sfrd)

        self.var_params = {**cosmo_var, **astro_var, **sfrd_var}
        self.fixed_params = {**cosmo_fixed, **astro_fixed, **sfrd_fixed}

        self.param_names = list(self.var_params.keys())
        self.param_arrays = [self.var_params[k] for k in self.param_names]

        self.shape = tuple(len(a) for a in self.param_arrays)
        self.N_models = np.prod(self.shape)

        cosmo_varying = any(np.size(v) > 1 for v in (cosmo_var).values())
        astro_varying = any(np.size(v) > 1 for v in {**astro_var, **sfrd_var}.values())

        #---------------------------------------------------------------------------------
        if not cosmo_varying and not astro_varying:
            self.run_type='single'
            self.message = 'both cosmological and astrophysical parameters are fixed.\n'
            self.Z_solver = Z_default
            Z_init = Z_start

        elif not cosmo_varying and astro_varying:
            obj_dark_ages = funcs(self.fixed_params)
            Tk_init = obj_dark_ages.basic_cosmo_Tcmb(Z_start)
            xe_init = obj_dark_ages.recomb_Saha_xe(Z_start,Tk_init)
            sol_da = obj_dark_ages.igm_solver(Z_solver=Z_da, xe_init=xe_init, Tk_init=Tk_init)
            xe_da = sol_da[0]
            Tk_da = sol_da[1]
            
            self.run_type='astro'
            self.message = 'cosmological parameters are fixed. Astrophysical parameters are varied.'
            self.Z_solver = Z_cd
            self.xe_init, self.Tk_init = xe_da[-1] , Tk_da[-1]
            self.simulator = cosmic_dawn_beyond
            Z_init = Zstar
        
        else:
            self.run_type='else'
            self.message = 'cosmological parameters are varied.\n'
            self.Z_solver = Z_default           
            self.xe_init, self.Tk_init = None , None
            self.simulator = dark_ages_to_today
            Z_init = Z_start
        #---------------------------------------------------------------------------------

        if self.Z_eval is not None:
            if (self.Z_eval[0]>Z_init or self.Z_eval[-1]<Z_end):
                print('\033[31mYour requested redshift values should satisfy ',Z_init,'>1+z>',Z_end)
                print('Terminating ...\033[00m')
                sys.exit()
            
            if type(self.Z_eval)==np.ndarray or type(self.Z_eval)==list:
                self.Z_eval=np.array(self.Z_eval)
                if self.Z_eval[1]>self.Z_eval[0]:
                    # Arranging redshifts from ascending to descending
                    self.Z_eval = self.Z_eval[::-1]
        #---------------------------------------------------------------------------------

        if self.run_type!='single' and self.n_cpu==1:
            print('\033[31mPlease use at least 2 CPUs. Terminating ... \033[00m')
            sys.exit()
        #---------------------------------------------------------------------------------

        #Create an output folder where all results will be saved.
        self.path=path
        if self.cpu_ind==0:
            if os.path.isdir(self.path)==False:
                print('\nThe requested directory does not exist. Creating ',self.path)
                os.mkdir(self.path)
            
            self.timestamp = strftime("%Y%m%d-%H%M%S", localtime())
            self.path = self.path + 'output_'+self.timestamp+'/'
            os.mkdir(self.path)

            self.formatted_timestamp = self.timestamp[9:11]+':'+self.timestamp[11:13]+':'+self.timestamp[13:15]+' '+self.timestamp[6:8]+'/'+self.timestamp[4:6]+'/'+ self.timestamp[:4]

            save_pipeline(self,'pipe')
        return None
    
    def run_simulation(self):
        '''
        This function solves the thermal and ionization history for default values of redshifts and then interpolates the quantities at your choice of redshifts. Then it solves reionization. Finally, it computes the spin temperature and hence the global 21-cm signal. A text file is generated which will contain the basic information about the simulation. 
        ''' 

        if self.run_type=='single':
        #Cosmological and astrophysical parameters are fixed.
            if self.cpu_ind==0:
                
                st = time.perf_counter()
                
                print_banner()
                print('Dark matter type: cold')
                print('\nSimulation type: ',self.message)
                
                myobj = funcs(self.fixed_params)
                Tk_init = myobj.basic_cosmo_Tcmb(Z_start)
                xe_init = myobj.recomb_Saha_xe(Z_start,Tk_init)

                print('Obtaining the thermal and ionisation history ...')
                sol = myobj.igm_solver(Z_solver=Z_default, xe_init=xe_init, Tk_init = Tk_init)
                
                xe = sol[0]
                Tk = sol[1]

                Q_Hii = myobj.QHii
                Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

                #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
                Tk[0:1806] = smoother(Z_default[0:1806],Tk[0:1806])

                if self.Z_eval is not None:
                    xe = CubicSpline(flipped_Z_default, np.flip(xe))(self.Z_eval)
                    Q_Hii = np.interp(self.Z_eval, flipped_Z_default, np.flip(Q_Hii))
                    Tk = CubicSpline(flipped_Z_default, np.flip(Tk))(self.Z_eval)
                    
                    print('Obtaining spin temperature ...')
                    Ts = myobj.hyfi_spin_temp(Z=self.Z_eval,xe=xe,Tk=Tk)
                    
                    print('Computing the 21-cm signal ...')
                    T21 = myobj.hyfi_twentyone_cm(Z=self.Z_eval,xe=xe,Q=Q_Hii,Ts=Ts)
                    x = self.Z_eval
                else:
                    print('Obtaining spin temperature ...')
                    Ts = myobj.hyfi_spin_temp(Z=Z_default,xe=xe,Tk=Tk)

                    print('Computing the 21-cm signal ...')
                    T21 = myobj.hyfi_twentyone_cm(Z=Z_default,xe=xe,Q=Q_Hii,Ts=Ts)

                    x = Z_default
                
                print('Done.')

                xe_save_name = self.path+'xe'
                Q_save_name = self.path+'Q'
                Tk_save_name = self.path+'Tk'
                Ts_save_name = self.path+'Ts'
                Tcmb_save_name = self.path+'Tcmb'
                T21_save_name = self.path+'T21'
                z_save_name = self.path+'one_plus_z'

                np.save(xe_save_name,xe)
                np.save(Q_save_name,Q_Hii)
                np.save(Tk_save_name,Tk)
                np.save(Ts_save_name,Ts)
                np.save(Tcmb_save_name,myobj.basic_cosmo_Tcmb(x))
                np.save(T21_save_name,T21)
                np.save(z_save_name,x)
               
                print('\033[32mYour outputs have been saved into folder:',self.path,'\033[00m')
                
                et = time.perf_counter()
                # get the execution time
                elapsed_time = et - st
                print('\nExecution time: %.2f seconds' %elapsed_time)

                #========================================================
                #Writing to a summary file
                myfile = write_summary(self, elapsed_time=elapsed_time)

                try:
                    max_T21 = np.min(T21)
                    max_ind = np.where(T21==max_T21)
                    [max_z] = x[max_ind]
                    myfile.write('\n\nStrongest 21-cm signal is {:.2f} mK, observed at z = {:.2f}'.format(max_T21,max_z-1))
                except:
                    pass

                tau_e = None
                try:
                    tau_e = myobj.reion_tau(50)     #To calculate tau even if reionisation is not complete
                    myfile.write("\nTotal Thomson-scattering optical depth = {:.4f}".format(tau_e))
                except:
                    pass

                myfile.write('\n')
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
                return None

        #=========================================================================
        else:
            partial_results = []
            partial_params = []
            if self.cpu_ind==0:
                #Master CPU
                print_banner()
                print('Dark matter type: cold')
                print('\nSimulation type: ',self.message)
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
                    all_params_dict, varying_params_only = params_from_index(self, idx)

                    result = self.simulator(all_params_dict, xe_init= self.xe_init, Tk_init= self.Tk_init, Z_eval = self.Z_eval)
                    
                    partial_params.append(varying_params_only)
                    partial_results.append(result)
                    
                    self.comm.send(1, dest=0, tag=77)

        
        self.comm.Barrier()
        gathered_results = self.comm.gather(partial_results, root=0)
        gathered_params = self.comm.gather(partial_params, root=0)

        if self.cpu_ind == 0:

            # Flatten results
            all_params  = [p for chunk in gathered_params for p in chunk]
            all_results = [r for chunk in gathered_results for r in chunk]

            param_df = pd.DataFrame(all_params)
            #If there are more outputs in future, then the unpacking below needs to be changed accordingly.
            T21 = np.vstack([r[0] for r in all_results])
            xHI = np.vstack([r[1] for r in all_results])
            tau = np.array([r[2] for r in all_results])

            save_path = self.path + 'echo_output.h5'
            with pd.HDFStore(save_path, mode="w") as store:

                # first layer
                store.put("params", param_df)

                # second layer
                if self.Z_eval is not None:
                    store.put("Z", pd.Series(self.Z_eval))
                else:
                    store.put("Z", pd.Series(self.Z_solver))

                # subsequent layers
                store.put("T21", pd.DataFrame(T21))
                store.put("xHI", pd.DataFrame(xHI))
                store.put("tau", pd.Series(tau))

            print('\033[32m\nOutputs saved into folder:',self.path,'\033[00m')
            
            et = time.perf_counter()
            # get the execution time
            elapsed_time = et - st
            print('\nProcessing time: %.2f seconds' %elapsed_time)

            #========================================================
            #Writing to a summary file

            myfile = write_summary(self, elapsed_time=elapsed_time)
            myfile.write('\n{} models generated'.format(self.N_models))
            myfile.write('\nNumber of CPU(s) = {}'.format(self.n_cpu))
            myfile.write('\n')
            myfile.close()
            #========================================================

            print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
        return None
    
    #End of function run_simulation               
#End of class pipeline
#=======================================================================================================