import numpy as np
from mpi4py import MPI
from mpi4py.util import pkl5
from itertools import product
import sys
import time
import os
import pickle
from scipy.interpolate import CubicSpline
from time import localtime, strftime
from tqdm import tqdm

from .const import Zstar, Z_start, Z_end, Z_default, Z_da, Z_cd, flipped_Z_default, phy_sfrd_default_model, emp_sfrd_default_model, semi_emp_sfrd_default_model
from .echofuncs import funcs
from .misc import *

#--------------------------------------------------------------------------------------------

#The following 2 functions will be useful if you want to save and load `pipeline` object.
def save_pipeline(obj, filename):
    '''Saves the class object :class:`pipeline`.
    
    Save the class object :class:`pipeline` for later use. It will save the object in the path where you have all the other outputs from this package.
    
    Parameters
    ~~~~~~~~~~

    obj : class
        This should be the class object you want to save.
        
    filename : str
        Give a file name only to your object, not the full path. obj will be saved in the ``obj.path`` directory.
    
    '''
    try:
        comm = MPI.COMM_WORLD
        cpu_ind = comm.Get_rank()
        Ncpu = comm.Get_size()
    except:
        cpu_ind=0
    if cpu_ind==0:
        if filename[-4:]!='.pkl': filename=filename+'.pkl'
        fullpath = obj.path+filename
        with open(fullpath, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    return None
    
def load_pipeline(filename):
    '''To load the class object :class:`pipeline`.
    
    Parameters
    ~~~~~~~~~~

    filename : str
        This should be the name of the file you gave in :func:`save_pipeline()` for saving class object :class:`pipeline`. Important: provide the full path for ``filename`` with the extension ``.pkl``.
        
    Returns
    ~~~~~~~

    class object    
    '''
    try:
        comm = MPI.COMM_WORLD
        cpu_ind = comm.Get_rank()
        Ncpu = comm.Get_size()
    except:
        cpu_ind=0
    if cpu_ind==0:
        with open(filename, 'rb') as inp:
            echo21obj = pickle.load(inp)
        print('Loaded the echo21 pipeline class object.\n')
    return echo21obj
#--------------------------------------------------------------------------------------------

class pipeline():
    '''
    This class runs the cosmic history solver and produces the global signal and the corresponding redshifts. There are 3 inputs required for a complete specification -- cosmological parameters, astrophysical parameter, and star formation related parameters. They are supplied through arguments, ``cosmo``, ``astro``, and ``sfrd_dic``, respectively. The notation for the parameters is as follows. All of these need to be dictionaries. For example:

    cosmo = {'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'sig8':0.811,'ns':0.965,'Tcmbo':2.725,'Yp':0.245},
    astro = {'fLy':1,'sLy' : 2.64,'fX':1,'wX':1.5, 'fesc':0.0106},
    sfrd_dic = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':1e4}
    
    Parameters
    ~~~~~~~~~~

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
    
    fLy : float, optional
        :math:`f_{\\mathrm{Ly}}`, a dimensionless parameter which controls the emissivity of the Lyman series photons. Default value ``1.0``.
    
    sLy : float, optional
        :math:`s`, spectral index of Lyman series SED, when expressed as :math:`\\epsilon\\propto E^{-s}`. :math:`\\epsilon` is energy emitted per unit energy range and per unit volume. Default value ``2.64``.

    fX : float, optional
        :math:`f_{\\mathrm{X}}`, a dimensionless parameter which controls the emissivity of the X-ray photons. Default value ``1``.
    
    wX : float, optional
        :math:`w`, spectral index of X-ray SED, when expressed as :math:`\\epsilon\\propto E^{-w}`. :math:`\\epsilon` is energy emitted per unit energy range and per unit volume. Default value ``1.5``.

    fesc : float, optional
        :math:`f_{\\mathrm{esc}}`, a dimensionless parameter which controls the escape fraction of the ionizing photons. Default value ``0.01``.

    sfrd_dic : dictionary, optional
        
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


    Methods
    ~~~~~~~
    '''
    def __init__(self,cosmo=None,astro= None, sfrd_dic=None,Z_eval=None,path='echo21_outputs/'):

        if cosmo is None:
            cosmo = {
                'Ho': 67.4, 'Om_m': 0.315, 'Om_b': 0.049, 'sig8': 0.811, 'ns': 0.965,
                'Tcmbo': 2.725, 'Yp': 0.245}
        if astro is None:
            astro = {'fLy': 1, 'sLy': 2.64, 'fX': 1, 'wX': 1.5, 'fesc': 0.0106}
        if sfrd_dic is None:
            sfrd_dic = {'type': 'phy', 'hmf': 'press74', 'mdef': 'fof', 'Tmin_vir': 1e4}

        
        self.comm = pkl5.Intracomm(MPI.COMM_WORLD)
        self.cpu_ind = self.comm.Get_rank()
        self.n_cpu = self.comm.Get_size()

        self.cosmo=cosmo
        self.astro=astro

        
        ####
        #Here I decide whether astro, cosmo, or both parameters are varied.
        self.model = 0
        for keys in self.astro.keys():
            if np.size(self.astro[keys])>1:
                self.model = 1
                break
        
        self.sfrd_type = sfrd_dic['type']
        if self.sfrd_type == 'phy':
            sfrd_dic={**phy_sfrd_default_model,**sfrd_dic}
            self.hmf = sfrd_dic['hmf']
            self.mdef = sfrd_dic['mdef']
            self.Tmin_vir = sfrd_dic['Tmin_vir']
            if self.model==0:
                if np.size(self.Tmin_vir)>1:
                    self.model=1
        elif self.sfrd_type == 'semi-emp':
            sfrd_dic={**semi_emp_sfrd_default_model,**sfrd_dic}
            self.hmf = sfrd_dic['hmf']
            self.mdef = sfrd_dic['mdef']
            self.Tmin_vir = sfrd_dic['Tmin_vir']
            self.t_star = sfrd_dic['t_star']
            if self.model==0:
                if np.size(self.Tmin_vir)>1 or np.size(self.t_star)>1:
                    self.model=1
        elif self.sfrd_type == 'emp':
            sfrd_dic={**emp_sfrd_default_model,**sfrd_dic}
            self.a_sfrd = sfrd_dic['a']
            if self.model==0:
                if np.size(self.a_sfrd)>1:
                    self.model=1

        for keys in self.cosmo.keys():
            if np.size(self.cosmo[keys])>1:
                self.model = self.model+2
                break
        ####

        #echo21 uses a master-worker distribution. So atleast 2 CPUs are required. I check that below.
        if self.model>0 and self.n_cpu==1:
            print('\033[31mPlease use at least 2 CPUs. Terminating ... \033[00m')
            sys.exit()


        #Converting all parameters to array or float according to their multiplicity.
        if self.model==0:
            self.astro=to_float(self.astro)
            self.cosmo=to_float(self.cosmo)
            if self.sfrd_type == 'phy': self.Tmin_vir = to_float(self.Tmin_vir)
            elif self.sfrd_type == 'semi-emp':
                self.Tmin_vir = to_float(self.Tmin_vir)
                self.t_star = to_float(self.t_star)
            else: self.a_sfrd = to_float(self.a_sfrd)
        elif self.model==1:
            self.astro=to_array(self.astro)
            self.cosmo=to_float(self.cosmo)
            if self.sfrd_type == 'phy': self.Tmin_vir = to_array(self.Tmin_vir)
            elif self.sfrd_type == 'semi-emp':
                self.Tmin_vir = to_array(self.Tmin_vir)
                self.t_star = to_array(self.t_star)
            else: self.a_sfrd = to_array(self.a_sfrd)
        elif self.model==2:
            self.astro=to_float(self.astro)
            self.cosmo=to_array(self.cosmo)
            if self.sfrd_type == 'phy': self.Tmin_vir = to_float(self.Tmin_vir)
            elif self.sfrd_type == 'semi-emp':
                self.Tmin_vir = to_float(self.Tmin_vir)
                self.t_star = to_float(self.t_star)
            else: self.a_sfrd = to_float(self.a_sfrd)
        elif self.model==3:
            self.astro=to_array(self.astro)
            self.cosmo=to_array(self.cosmo)
            if self.sfrd_type == 'phy': self.Tmin_vir = to_array(self.Tmin_vir)
            elif self.sfrd_type == 'semi-emp':
                self.Tmin_vir = to_array(self.Tmin_vir)
                self.t_star = to_array(self.t_star)
            else: self.a_sfrd = to_array(self.a_sfrd)
        else:
            print('Impossible!')
            sys.exit()
        
        self.Z_eval = Z_eval

        if type(self.Z_eval)==np.ndarray or type(self.Z_eval)==list:
            self.Z_eval=np.array(self.Z_eval)
            if self.Z_eval[1]>self.Z_eval[0]:
                # Arranging redshifts from ascending to descending
                self.Z_eval = self.Z_eval[::-1]

        self.Ho = cosmo['Ho']
        self.Om_m = cosmo['Om_m']
        self.Om_b = cosmo['Om_b']
        self.sig8 = cosmo['sig8']
        self.ns = cosmo['ns']
        self.Tcmbo = cosmo['Tcmbo']
        self.Yp = cosmo['Yp']
        
        self.fLy = astro['fLy']
        self.sLy = astro['sLy']
        self.fX = astro['fX']
        self.wX = astro['wX']
        self.fesc = astro['fesc']


        #Create an output folder where all results will be saved.
        self.path=path
        if self.cpu_ind==0:
            if os.path.isdir(self.path)==False:
                print('The requested directory does not exist. Creating ',self.path)
                os.mkdir(self.path)
            
            self.timestamp = strftime("%Y%m%d-%H%M%S", localtime())
            self.path = self.path + 'output_'+self.timestamp+'/'
            os.mkdir(self.path)

            self.formatted_timestamp = self.timestamp[9:11]+':'+self.timestamp[11:13]+':'+self.timestamp[13:15]+' '+self.timestamp[6:8]+'/'+self.timestamp[4:6]+'/'+ self.timestamp[:4]

            save_pipeline(self,'pipe')
        return None

    def _write_summary(self, elapsed_time):
        '''
        Given the elapsed time of the code execution write the main summary of the run.
        '''
        sumfile = self.path+"glob_sig_"+self.timestamp+".txt"
        myfile = open(sumfile, "w")
        myfile.write('''\n███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  ██╗
██╔════╝██╔════╝██║  ██║██╔═══██╗╚════██╗███║
█████╗  ██║     ███████║██║   ██║ █████╔╝╚██║
██╔══╝  ██║     ██╔══██║██║   ██║██╔═══╝  ██║
███████╗╚██████╗██║  ██║╚██████╔╝███████╗ ██║
╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═╝\n''')
        myfile.write('Shikhar Mittal, 2025\n')
        myfile.write('\nThis is output_'+self.timestamp)
        myfile.write('\n------------------------------\n')
        myfile.write('\nTime stamp: '+self.formatted_timestamp)
        myfile.write('\n\nExecution time: %.2f seconds' %elapsed_time) 
        myfile.write('\n\n')
        myfile.write('Dark matter type: cold')
        myfile.write('\n\nParameters given:\n')
        myfile.write('-----------------')
        myfile.write('\nHo = {}'.format(self.Ho))
        myfile.write('\nOm_m = {}'.format(self.Om_m))
        myfile.write('\nOm_b = {}'.format(self.Om_b))
        myfile.write('\nsig8 = {}'.format(self.sig8))
        myfile.write('\nns = {}'.format(self.ns))
        myfile.write('\nTcmbo = {}'.format(self.Tcmbo))
        myfile.write('\nYp = {}'.format(self.Yp))

        myfile.write('\n\nfLy = {}'.format(self.fLy))
        myfile.write('\nsLy = {}'.format(self.sLy))
        myfile.write('\nfX = {}'.format(self.fX))
        myfile.write('\nwX = {}'.format(self.wX))
        myfile.write('\nfesc = {}'.format(self.fesc))
        myfile.write('\n\nSFRD')
        myfile.write('\n  Type = '+self.sfrd_type)
        if self.sfrd_type == 'phy':
            myfile.write('\n  HMF = '+self.hmf)
            myfile.write('\n  mdef = '+self.mdef)
            myfile.write('\n  Tmin_vir = {}'.format(self.Tmin_vir))
        elif self.sfrd_type == 'semi-emp':
            myfile.write('\n  HMF = '+self.hmf)
            myfile.write('\n  mdef = '+self.mdef)
            myfile.write('\n  Tmin_vir = {}'.format(self.Tmin_vir))
            myfile.write('\n  t_star = {}'.format(self.t_star))
        else:
            myfile.write('\n  a = {}'.format(self.a_sfrd))
        
        myfile.write('\n')
        return myfile

    def print_input(self):
        '''Prints the input parameters you gave.'''

        print('Dark matter type: cold')

        print('\n\033[93mParameters given:\n')
        print('-----------------')
        print('\nHo = {}'.format(self.Ho))
        print('Om_m = {}'.format(self.Om_m))
        print('Om_b = {}'.format(self.Om_b))
        print('sig8 = {}'.format(self.sig8))
        print('ns = {}'.format(self.ns))
        print('Tcmbo = {}'.format(self.Tcmbo))
        print('Yp = {}'.format(self.Yp))

        print('\n\nfLy = {}'.format(self.fLy))
        print('sLy = {}'.format(self.sLy))
        print('fX = {}'.format(self.fX))
        print('wX = {}'.format(self.wX))
        print('fesc = {}'.format(self.fesc))
        print('\n\nSFRD')
        print('  Type = '+self.sfrd_type)
        if self.sfrd_type == 'phy':
            print('  HMF = '+self.hmf)
            print('  mdef = '+self.mdef)
            print('  Tmin_vir = {}\033[00m\n'.format(self.Tmin_vir))
        elif self.sfrd_type == 'semi-emp':
            print('  HMF = '+self.hmf)
            print('  mdef = '+self.mdef)
            print('  Tmin_vir = {}\033[00m\n'.format(self.Tmin_vir))
            print('  t_star = {}\033[00m\n'.format(self.t_star))
        else:
            print('  a = {}\033[00m\n'.format(self.a_sfrd))

        return None
    
    def glob_sig(self):
        '''
        This function solves the thermal and ionization history for default values of redshifts and then interpolates the quantities at your choice of redshifts. Then it solves reionization. Finally, it computes the spin temperature and hence the global 21-cm signal. A text file is generated which will contain the basic information about the simulation. 
        ''' 

        if self.model==0:
        #Cosmological and astrophysical parameters are fixed.
            if self.cpu_ind==0:
                print_banner()
                print('Dark matter type: cold')
                print('\nBoth cosmological and astrophysical parameters are fixed.\n')
                
                st = time.process_time()
                
                if self.sfrd_type == 'phy':
                    myobj = funcs(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,sig8=self.sig8,ns=self.ns,Tcmbo=self.Tcmbo,Yp=self.Yp,fLy=self.fLy,sLy=self.sLy,fX=self.fX,wX=self.wX,fesc=self.fesc,type = self.sfrd_type,hmf=self.hmf,mdef=self.mdef,Tmin_vir=self.Tmin_vir)
                elif self.sfrd_type == 'semi-emp':
                    myobj = funcs(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,sig8=self.sig8,ns=self.ns,Tcmbo=self.Tcmbo,Yp=self.Yp, fLy=self.fLy,sLy=self.sLy,fX=self.fX,wX=self.wX,fesc=self.fesc,type = self.sfrd_type,hmf=self.hmf,mdef=self.mdef,Tmin_vir=self.Tmin_vir, t_star=self.t_star)
                else:
                    myobj = funcs(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,sig8=self.sig8,ns=self.ns,Tcmbo=self.Tcmbo,Yp=self.Yp,fLy=self.fLy,sLy=self.sLy,fX=self.fX,wX=self.wX,fesc=self.fesc,type = self.sfrd_type,a=self.a_sfrd)
                
                Z_temp = Z_default

                if self.Z_eval is not None:
                    if (self.Z_eval[0]>1501 or self.Z_eval[-1]<Z_end):
                        print('\033[31mYour requested redshift values should satisfy ',1501,'>1+z>',Z_end)
                        print('Terminating ...\033[00m')
                        sys.exit()
                    else:
                        Z_temp = self.Z_eval
                
                print('Obtaining the thermal and ionisation history ...')
                sol = myobj.igm_solver(Z_eval=Z_default)
                
                xe = sol[0]
                Tk = sol[1]

                Q_Hii = myobj.QHii
                Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

                #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
                Tk[0:1806] = smoother(Z_default[0:1806],Tk[0:1806])

                if self.Z_eval is not None:
                    splxe = CubicSpline(flipped_Z_default, np.flip(xe))
                    xe = splxe(self.Z_eval)
                    Q_Hii = np.interp(self.Z_eval, flipped_Z_default, np.flip(Q_Hii))
                    splTk = CubicSpline(flipped_Z_default, np.flip(Tk))
                    Tk = splTk(self.Z_eval)

                print('Obtaining spin temperature ...')
                Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)

                print('Computing the 21-cm signal ...')
                T21_mod1 = myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)
                
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
                np.save(Tcmb_save_name,myobj.basic_cosmo_Tcmb(Z_temp))
                np.save(T21_save_name,T21_mod1)
                np.save(z_save_name,Z_temp)
               
                print('\033[32mYour outputs have been saved into folder:',self.path,'\033[00m')
                
                et = time.process_time()
                # get the execution time
                elapsed_time = et - st
                print('\nExecution time: %.2f seconds' %elapsed_time)

                #========================================================
                #Writing to a summary file
                try:
                    max_T21 = np.min(T21_mod1)
                    max_ind = np.where(T21_mod1==max_T21)
                    [max_z] = Z_temp[max_ind]
                except:
                    pass

                tau_e = None
                try:
                    tau_e = myobj.reion_tau(50)     #To calculate tau even if reionisation is not complete 
                except:
                    pass


                z50 = None
                try:
                    idx = np.where(np.abs(Q_Hii-0.5)<=0.01)[0][0]
                    z50 = Z_default[idx]-1
                    z100 = None
                    try:
                        idx = np.where(Q_Hii>=0.98)[0][0]
                        z100 = Z_default[idx]-1
                        #tau_e = myobj.reion_tau(50)
                    except:
                        print('\n{:.1f} % universe reionised by {:.1f}'.format(100*Q_Hii[-1], Z_temp[-1]-1))
                except:
                    print('\n{:.1f} % universe reionised by {:.1f}'.format(100*Q_Hii[-1], Z_temp[-1]-1))

                myfile = self._write_summary(elapsed_time=elapsed_time)
                
                if z50!=None:
                    myfile.write('\n50% reionisation complete at z = {:.2f}'.format(z50))
                    if z100!=None:
                        myfile.write("\nReionisation complete at z = {:.2f}".format(z100))
                        #myfile.write("\nTotal Thomson-scattering optical depth = {:.4f}".format(tau_e))

                if tau_e != None:
                    myfile.write("\nTotal Thomson-scattering optical depth = {:.4f}".format(tau_e))      #Adding to print tau in summary file, if it is computed successfully.


                try: myfile.write('\n\nStrongest 21-cm signal is {:.2f} mK, observed at z = {:.2f}'.format(max_T21,max_z-1))
                except: pass
                myfile.write('\n')
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
                return None

#=========================================================================
#=========================================================================
        elif self.model==1:
        #Cosmological parameters are fixed so dark ages is solved only once.
            if self.cpu_ind==0:
                print_banner()
                print('Dark matter type: cold')
                print('\nCosmological parameters are fixed. Astrophysical parameters are varied.')
                print('\nGenerating once the thermal and ionization history for dark ages ...')
            
            myobj_da = funcs(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,sig8=self.sig8,ns=self.ns,Tcmbo=self.Tcmbo,Yp=self.Yp)

            sol_da = myobj_da.igm_solver(Z_eval=Z_da)
            xe_da = sol_da[0]
            Tk_da = sol_da[1]

            Z_temp = Z_cd
            if self.Z_eval is not None:
                if (self.Z_eval[0]>Zstar or self.Z_eval[-1]<Z_end):
                    print('\033[31mYour requested redshift values should satisfy ',Zstar,'>1+z>',Z_end)
                    print('Terminating ...\033[00m')
                    sys.exit()
                else:
                    Z_temp = self.Z_eval

            n_values = len(Z_temp)
            if self.sfrd_type=='phy':
                param_grid = list(product(self.fLy,self.sLy,self.fX,self.wX,self.fesc,self.Tmin_vir))
            elif self.sfrd_type=='emp':
                param_grid = list(product(self.fLy,self.sLy,self.fX,self.wX,self.fesc,self.a_sfrd))
            elif self.sfrd_type=='semi-emp':
                param_grid = list(product(self.fLy,self.sLy,self.fX,self.wX,self.fesc,self.Tmin_vir,self.t_star))            

            T21_partial = []
            xHI_partial = []    #Added to save xHI for each model as well as T21
            tau_partial = []

            if self.cpu_ind==0:
                #Master CPU
                n_mod = len(param_grid)
                print('Done.\n\nGenerating',n_mod,'models for cosmic dawn ...\n')
                st = time.process_time()
                done = 0
                pbar = tqdm(total=n_mod, desc="Processing models", ncols=100)

                while done < n_mod:
                    status = MPI.Status()
                    progress = self.comm.recv(source=MPI.ANY_SOURCE, tag=77, status=status)
                    done += progress
                    pbar.update(progress)
                pbar.close()
            else:
                #Worker CPU
                partial_param = param_grid[self.cpu_ind-1::self.n_cpu-1]
                if self.sfrd_type=='phy':
                    for (fly, sly, fx, wx, fesc, tmin_vir) in partial_param:
                        result = cdm_phy_cd(self.Ho,self.Om_m,self.Om_b,self.sig8,self.ns,self.Tcmbo,self.Yp, fly,sly,fx,wx,fesc,tmin_vir,self.hmf,self.mdef,xe_da[-1] , Tk_da[-1], self.Z_eval, Z_temp)
                        T21_partial.append((fly, sly, fx, wx, fesc, tmin_vir, result[0]) )
                        xHI_partial.append((fly, sly, fx, wx, fesc, tmin_vir, result[1]) )     #changes made to misc so that xHI is also output by cdm_phy_cd  
                        tau_partial.append((fly, sly, fx, wx, fesc, tmin_vir, result[2]) )     #changes made to misc so that tau is also output by cdm_phy_cd
                        self.comm.send(1, dest=0, tag=77)
                elif self.sfrd_type=='semi-emp':
                    for (fly, sly, fx, wx, fesc, tmin_vir,t_star) in partial_param:
                        T21_partial.append((fly, sly, fx, wx, fesc, tmin_vir,t_star, cdm_semi_cd(self.Ho,self.Om_m,self.Om_b,self.sig8,self.ns,self.Tcmbo,self.Yp, fly,sly,fx,wx,fesc,tmin_vir,t_star,self.hmf,self.mdef, xe_da[-1],Tk_da[-1],self.Z_eval,Z_temp)) )
                        self.comm.send(1, dest=0, tag=77)
                elif self.sfrd_type=='emp':
                    for (fly, sly, fx, wx, fesc, asfrd) in partial_param:
                        T21_partial.append((fly, sly, fx, wx, fesc, asfrd, cdm_emp_cd(self.Ho,self.Om_m,self.Om_b,self.sig8,self.ns,self.Tcmbo,self.Yp, fly, sly,fx,wx,fesc,asfrd, xe_da[-1],Tk_da[-1],self.Z_eval, Z_temp)) )
                        self.comm.send(1, dest=0, tag=77)
                            
            self.comm.Barrier()
            gathered_T21 = self.comm.gather(T21_partial, root=0)           
            gathered_xHI = self.comm.gather(xHI_partial, root=0) 
            gathered_tau = self.comm.gather(tau_partial, root=0)
            if self.cpu_ind == 0:

                # Flatten results
                all_T21 = [item for chunk in gathered_T21 for item in chunk]
                all_xHI = [item for chunk in gathered_xHI for item in chunk]
                all_tau = [item for chunk in gathered_tau for item in chunk]

                if self.sfrd_type=='phy':
                    T21_cd = np.zeros((np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir),n_values))
                    xHI_cd = np.zeros((np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir),n_values))
                    tau = np.zeros((np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir))) #tau does not depend on redshift, so it has one less dimension than T21 and xHI

                    # Create mapping from values to indices
                    fLy_index = {val: i for i, val in enumerate(self.fLy)}
                    sLy_index = {val: j for j, val in enumerate(self.sLy)}
                    fX_index = {val: k for k, val in enumerate(self.fX)}
                    wX_index = {val: l for l, val in enumerate(self.wX)}
                    fesc_index = {val: m for m, val in enumerate(self.fesc)}
                    Tmin_index = {val: n for n, val in enumerate(self.Tmin_vir)}

                    # Fill T21 array
                    for fly_val, sly_val, fx_val, w_val, fesc_val, tmin_val, val in all_T21:
                        i, j, k, l, m, n = fLy_index[fly_val], sLy_index[sly_val], fX_index[fx_val], wX_index[w_val], fesc_index[fesc_val], Tmin_index[tmin_val]
                        T21_cd[i, j, k, l, m, n, :] = val
                
                    # Fill xHI array, just mirroring logic for T21
                    for fly_val, sly_val, fx_val, w_val, fesc_val, tmin_val, val in all_xHI:
                        i, j, k, l, m, n = fLy_index[fly_val], sLy_index[sly_val], fX_index[fx_val], wX_index[w_val], fesc_index[fesc_val], Tmin_index[tmin_val]
                        xHI_cd[i, j, k, l, m, n, :] = val
                    # Fill tau array, just mirroring logic for T21
                    for fly_val, sly_val, fx_val, w_val, fesc_val, tmin_val, val in all_tau:
                        i, j, k, l, m, n = fLy_index[fly_val], sLy_index[sly_val], fX_index[fx_val], wX_index[w_val], fesc_index[fesc_val], Tmin_index[tmin_val]
                        tau[i, j, k, l, m, n] = val

                elif self.sfrd_type=='semi-emp':
                    T21_cd = np.zeros((np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir),np.size(self.t_star),n_values))

                    # Create mapping from values to indices
                    fLy_index = {val: i for i, val in enumerate(self.fLy)}
                    sLy_index = {val: j for j, val in enumerate(self.sLy)}
                    fX_index = {val: k for k, val in enumerate(self.fX)}
                    wX_index = {val: l for l, val in enumerate(self.wX)}
                    fesc_index = {val: m for m, val in enumerate(self.fesc)}
                    Tmin_index = {val: n for n, val in enumerate(self.Tmin_vir)}
                    t_star_index = {val: o for o, val in enumerate(self.t_star)}

                    # Fill T21 array
                    for fly_val, sly_val, fx_val, w_val, fesc_val, tmin_val, t_star_val, val in all_T21:
                        i, j, k, l, m, n, o = fLy_index[fly_val], sLy_index[sly_val], fX_index[fx_val], wX_index[w_val], fesc_index[fesc_val], Tmin_index[tmin_val], t_star_index[t_star_val]
                        T21_cd[i, j, k, l, m, n, o, :] = val

                elif self.sfrd_type=='emp':
                    T21_cd = np.zeros((np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.a_sfrd),n_values))

                    # Create mapping from values to indices
                    fLy_index = {val: i for i, val in enumerate(self.fLy)}
                    sLy_index = {val: j for j, val in enumerate(self.sLy)}
                    fX_index = {val: k for k, val in enumerate(self.fX)}
                    wX_index = {val: l for l, val in enumerate(self.wX)}
                    fesc_index = {val: m for m, val in enumerate(self.fesc)}
                    a_index = {val: n for n, val in enumerate(self.a_sfrd)}

                    # Fill T21 array
                    for fly_val, sly_val, fx_val, w_val, fesc_val, a_val, val in all_T21:
                        i, j, k, l, m, n = fLy_index[fly_val], sLy_index[sly_val], fX_index[fx_val], wX_index[w_val], fesc_index[fesc_val], a_index[a_val]
                        T21_cd[i, j, k, l, m, n, :] = val


                T21_save_name = self.path+'T21'
                xHI_save_name = self.path+'xHI'
                tau_save_name = self.path+'tau'
                z_save_name = self.path+'one_plus_z'
                
                np.save(T21_save_name,T21_cd)
                np.save(xHI_save_name,xHI_cd)
                np.save(tau_save_name,tau)
                np.save(z_save_name,Z_temp)
                print('\033[32m\nOutput saved into folder:',self.path,'\033[00m')
                
                et = time.process_time()
                # get the execution time
                elapsed_time = et - st
                print('\nProcessing time: %.2f seconds' %elapsed_time)

                #========================================================
                #Writing to a summary file

                myfile = self._write_summary(elapsed_time=elapsed_time)
                myfile.write('\n{} models generated'.format(n_mod))
                myfile.write('\nNumber of CPU(s) = {}'.format(self.n_cpu))
                myfile.write('\n')
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')

#=========================================================================
#=========================================================================
        elif self.model==2:

            if self.cpu_ind==0:
                print_banner()
                print('Dark matter type: cold')
                print('\nOnly cosmological parameters are varied.')
            
            Z_temp = Z_default
            if self.Z_eval is not None:
                if (self.Z_eval[0]>1501 or self.Z_eval[-1]<Z_end):
                    print('\033[31mYour requested redshift values should satisfy ',1501,'>1+z>',Z_end)
                    print('Terminating ...\033[00m')
                    sys.exit()
                else:
                    Z_temp = self.Z_eval

            n_values = len(Z_temp)
            param_grid = list(product(self.Ho, self.Om_m, self.Om_b, self.sig8, self.ns, self.Tcmbo, self.Yp))               

            T21_partial = []
            xHI_partial = []
            tau_partial = []
            if self.cpu_ind==0:
                #Master CPU
                n_mod = len(param_grid)
                print('\nGenerating',n_mod,'models ...')
                st = time.process_time()
                done = 0
                pbar = tqdm(total=n_mod, desc="Processing models", ncols=100)

                while done < n_mod:
                    status = MPI.Status()
                    progress = self.comm.recv(source=MPI.ANY_SOURCE, tag=77, status=status)
                    done += progress
                    pbar.update(progress)
                pbar.close()
            else:
                #Worker CPU
                partial_param = param_grid[self.cpu_ind-1::self.n_cpu-1]
                if self.sfrd_type=='phy':
                    for (Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp) in partial_param:
                        result = cdm_phy_full(Ho,Om_m,Om_b,sig8,ns,Tcmbo,Yp, self.fLy,self.sLy,self.fX,self.wX,self.fesc,self.Tmin_vir,self.hmf,self.mdef, self.Z_eval, Z_temp)
                        T21_partial.append ((Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, result[0]) )
                        xHI_partial.append((Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, result[1]) )
                        tau_partial.append((Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, result[2]) )     #changes made to misc so that tau is also output by cdm_phy_full
                        self.comm.send(1, dest=0, tag=77)
                elif self.sfrd_type=='semi-emp':
                    for (Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp) in partial_param:
                        T21_partial.append ((Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, cdm_semi_full(Ho,Om_m,Om_b,sig8,ns,Tcmbo,Yp, self.fLy,self.sLy,self.fX,self.wX,self.fesc,self.Tmin_vir,self.t_star,self.hmf,self.mdef, self.Z_eval,Z_temp)) )
                        self.comm.send(1, dest=0, tag=77)
                elif self.sfrd_type=='emp':
                    for (Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp) in partial_param:
                        T21_partial.append((Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, cdm_emp_full(Ho,Om_m,Om_b,sig8,ns,Tcmbo,Yp, self.fLy,self.sLy,self.fX,self.wX,self.fesc,self.a_sfrd, self.Z_eval, Z_temp)) )
                        self.comm.send(1, dest=0, tag=77)

            self.comm.Barrier()
            gathered_T21 = self.comm.gather(T21_partial, root=0)           
            gathered_xHI = self.comm.gather(xHI_partial, root=0)
            gathered_tau = self.comm.gather(tau_partial, root=0)           
            
            if self.cpu_ind == 0:

                # Flatten results
                all_T21 = [item for chunk in gathered_T21 for item in chunk]
                all_tau = [item for chunk in gathered_tau for item in chunk]
                all_xHI = [item for chunk in gathered_xHI for item in chunk]

                T21_mod2 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),n_values))
                xHI_mod2 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),n_values))
                tau_mod2 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp)))   #No redshift dimension for tau, hence lack of n_values
                # Create mapping from values to indices
                Ho_index = {val: i for i, val in enumerate(self.Ho)}
                Omm_index = {val: j for j, val in enumerate(self.Om_m)}
                Omb_index = {val: k for k, val in enumerate(self.Om_b)}
                sig8_index = {val: l for l, val in enumerate(self.sig8)}
                ns_index = {val: m for m, val in enumerate(self.ns)}
                Tcmb_index = {val: n for n, val in enumerate(self.Tcmbo)}
                Yp_index = {val: o for o, val in enumerate(self.Yp)}

                # Fill T21 array
                for Ho_val, Omm_val, Omb_val, sig8_val, ns_val, Tcmb_val, Yp_val, val in all_T21:
                    i, j, k, l, m, n, o = Ho_index[Ho_val], Omm_index[Omm_val], Omb_index[Omb_val], sig8_index[sig8_val], ns_index[ns_val], Tcmb_index[Tcmb_val], Yp_index[Yp_val]

                    T21_mod2[i, j, k, l, m, n, o, :] = val

                # Fill xHI array
                for Ho_val, Omm_val, Omb_val, sig8_val, ns_val, Tcmb_val, Yp_val, val in all_xHI:
                    i, j, k, l, m, n, o = Ho_index[Ho_val], Omm_index[Omm_val], Omb_index[Omb_val], sig8_index[sig8_val], ns_index[ns_val], Tcmb_index[Tcmb_val], Yp_index[Yp_val]

                    xHI_mod2[i, j, k, l, m, n, o, :] = val

                # Fill tau array
                for Ho_val, Omm_val, Omb_val, sig8_val, ns_val, Tcmb_val, Yp_val, val in all_tau:
                    i, j, k, l, m, n, o = Ho_index[Ho_val], Omm_index[Omm_val], Omb_index[Omb_val], sig8_index[sig8_val], ns_index[ns_val], Tcmb_index[Tcmb_val], Yp_index[Yp_val]

                    tau_mod2[i, j, k, l, m, n, o] = val

                z_save_name = self.path+'one_plus_z'
                T21_save_name = self.path+'T21'
                xHI_save_name = self.path+'xHI'
                tau_save_name = self.path+'tau'
                np.save(xHI_save_name,xHI_mod2)
                np.save(T21_save_name,T21_mod2)
                np.save(tau_save_name,tau_mod2)     #Saving tau for only cosmo params varied
                np.save(z_save_name,Z_temp)

                print('\033[32m\nOutput saved into folder:',self.path,'\033[00m')
                
                et = time.process_time()
                # get the execution time
                elapsed_time = et - st
                print('\nProcessing time: %.2f seconds' %elapsed_time)
                #========================================================
                #Writing to a summary file

                myfile = self._write_summary(elapsed_time=elapsed_time)
                myfile.write('\n{} models generated'.format(n_mod))
                myfile.write('\nNumber of CPU(s) = {}'.format(self.n_cpu))
                myfile.write('\n')
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
#=========================================================================
#=========================================================================

        elif self.model==3:

            if self.cpu_ind==0:
                print_banner()
                print('Dark matter type: cold')
                print('\nBoth cosmological and astrophysical parameters are varied.')
            
            Z_temp = Z_default
            if self.Z_eval is not None:
                if (self.Z_eval[0]>1501 or self.Z_eval[-1]<Z_end):
                    print('\033[31mYour requested redshift values should satisfy ',1501,'>1+z>',Z_end)
                    print('Terminating ...\033[00m')
                    sys.exit()
                else:
                    Z_temp = self.Z_eval

            n_values = len(Z_temp)
            if self.sfrd_type=='phy':                
                param_grid = list(product(self.Ho, self.Om_m, self.Om_b, self.sig8, self.ns, self.Tcmbo, self.Yp, self.fLy,self.sLy,self.fX,self.wX,self.fesc,self.Tmin_vir))
            elif self.sfrd_type=='semi-emp':
                param_grid = list(product(self.Ho, self.Om_m, self.Om_b, self.sig8, self.ns, self.Tcmbo, self.Yp, self.fLy,self.sLy,self.fX,self.wX,self.fesc,self.Tmin_vir, self.t_star))
            else:
                param_grid = list(product(self.Ho, self.Om_m, self.Om_b, self.sig8, self.ns, self.Tcmbo, self.Yp, self.fLy,self.sLy,self.fX,self.wX,self.fesc,self.a_sfrd))

            T21_partial = []
            xHI_partial = []
            tau_partial = []
            if self.cpu_ind==0:
                #Master CPU
                n_mod = len(param_grid)
                print('\nGenerating',n_mod,'models ...')
                st = time.process_time()
                done = 0
                pbar = tqdm(total=n_mod, desc="Processing models", ncols=100)

                while done < n_mod:
                    status = MPI.Status()
                    progress = self.comm.recv(source=MPI.ANY_SOURCE, tag=77, status=status)
                    done += progress
                    pbar.update(progress)
                pbar.close()
            else:
                #Worker CPU
                partial_param = param_grid[self.cpu_ind-1::self.n_cpu-1]
                if self.sfrd_type=='phy':
                    for (Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fly, sly, fx, wx, fesc, tmin_vir) in partial_param:
                        result = cdm_phy_full(Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fly, sly, fx, wx, fesc, tmin_vir,self.hmf,self.mdef, self.Z_eval, Z_temp)
                        T21_partial.append((Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fly, sly, fx, wx, fesc, tmin_vir,result[0]))
                        xHI_partial.append((Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fly, sly, fx, wx, fesc, tmin_vir, result[1]))
                        tau_partial.append((Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fly, sly, fx, wx, fesc, tmin_vir, result[2]))     #changes made to misc so that tau is also output by cdm_phy_full
                        self.comm.send(1, dest=0, tag=77)
                elif self.sfrd_type=='semi-emp':
                    for (Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fly, sly, fx, wx, fesc, tmin_vir, t_star) in partial_param:
                        T21_partial.append((Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fly, sly, fx, wx, fesc, tmin_vir, t_star, cdm_semi_full(Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp,  fly, sly, fx, wx, fesc, tmin_vir, t_star,self.hmf,self.mdef, self.Z_eval,Z_temp)) )
                        self.comm.send(1, dest=0, tag=77)
                elif self.sfrd_type=='emp':
                    for (Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fly, sly, fx, wx, fesc, a_sfrd) in partial_param:
                        T21_partial.append((Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fly, sly, fx, wx, fesc, a_sfrd, cdm_emp_full( Ho, Om_m, Om_b, sig8, ns, Tcmbo, Yp, fly, sly, fx, wx, fesc, a_sfrd, self.Z_eval, Z_temp)) )
                        self.comm.send(1, dest=0, tag=77)

            self.comm.Barrier()
            gathered_T21 = self.comm.gather(T21_partial, root=0)           
            gathered_xHI = self.comm.gather(xHI_partial, root=0)           
            gathered_tau = self.comm.gather(tau_partial, root=0)           

            if self.cpu_ind == 0:
                # Flatten results
                all_T21 = [item for chunk in gathered_T21 for item in chunk]
                all_xHI = [item for chunk in gathered_xHI for item in chunk]
                all_tau = [item for chunk in gathered_tau for item in chunk]

                #CDM
                if self.sfrd_type=='phy':
                    T21_mod3 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir),n_values))
                    xHI_mod3 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir),n_values))
                    tau_mod3 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir)))   #No redshift dimension for tau, hence lack of n_values
                    # Create mapping from values to indices
                    Ho_index = {val: i for i, val in enumerate(self.Ho)}
                    Omm_index = {val: j for j, val in enumerate(self.Om_m)}
                    Omb_index = {val: k for k, val in enumerate(self.Om_b)}
                    sig8_index = {val: l for l, val in enumerate(self.sig8)}
                    ns_index = {val: m for m, val in enumerate(self.ns)}
                    Tcmb_index = {val: n for n, val in enumerate(self.Tcmbo)}
                    Yp_index = {val: o for o, val in enumerate(self.Yp)}
                    
                    fLy_index = {val: r for r, val in enumerate(self.fLy)}
                    sLy_index = {val: r for r, val in enumerate(self.sLy)}
                    fX_index = {val: r for r, val in enumerate(self.fX)}
                    wX_index = {val: r for r, val in enumerate(self.wX)}
                    fesc_index = {val: r for r, val in enumerate(self.fesc)}
                    Tmin_index = {val: r for r, val in enumerate(self.Tmin_vir)}

                    # Fill T21 array
                    for Ho_val, Omm_val, Omb_val, sig8_val, ns_val, Tcmb_val, Yp_val, fly_val, sly_val, fx_val, w_val, fesc_val, tmin_val, val in all_T21:
                        i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13 = Ho_index[Ho_val], Omm_index[Omm_val], Omb_index[Omb_val], sig8_index[sig8_val], ns_index[ns_val], Tcmb_index[Tcmb_val], Yp_index[Yp_val], fLy_index[fly_val], sLy_index[sly_val], fX_index[fx_val], wX_index[w_val], fesc_index[fesc_val], Tmin_index[tmin_val]
                        
                        T21_mod3[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, :] = val

                    # Fill xHI array
                    for Ho_val, Omm_val, Omb_val, sig8_val, ns_val, Tcmb_val, Yp_val, fly_val, sly_val, fx_val, w_val, fesc_val, tmin_val, val in all_xHI:
                        i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13 = Ho_index[Ho_val], Omm_index[Omm_val], Omb_index[Omb_val], sig8_index[sig8_val], ns_index[ns_val], Tcmb_index[Tcmb_val], Yp_index[Yp_val], fLy_index[fly_val], sLy_index[sly_val], fX_index[fx_val], wX_index[w_val], fesc_index[fesc_val], Tmin_index[tmin_val]
                        
                        xHI_mod3[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, :] = val

                    # Fill tau array
                    for Ho_val, Omm_val, Omb_val, sig8_val, ns_val, Tcmb_val, Yp_val, fly_val, sly_val, fx_val, w_val, fesc_val, tmin_val, val in all_tau:
                        i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13 = Ho_index[Ho_val], Omm_index[Omm_val], Omb_index[Omb_val], sig8_index[sig8_val], ns_index[ns_val], Tcmb_index[Tcmb_val], Yp_index[Yp_val], fLy_index[fly_val], sLy_index[sly_val], fX_index[fx_val], wX_index[w_val], fesc_index[fesc_val], Tmin_index[tmin_val]
                        
                        tau_mod3[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13] = val

                elif self.sfrd_type=='semi-emp':
                    T21_mod3 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir),np.size(self.t_star),n_values))

                    # Create mapping from values to indices
                    Ho_index = {val: i for i, val in enumerate(self.Ho)}
                    Omm_index = {val: j for j, val in enumerate(self.Om_m)}
                    Omb_index = {val: k for k, val in enumerate(self.Om_b)}
                    sig8_index = {val: l for l, val in enumerate(self.sig8)}
                    ns_index = {val: m for m, val in enumerate(self.ns)}
                    Tcmb_index = {val: n for n, val in enumerate(self.Tcmbo)}
                    Yp_index = {val: o for o, val in enumerate(self.Yp)}
                    
                    fLy_index = {val: r for r, val in enumerate(self.fLy)}
                    sLy_index = {val: r for r, val in enumerate(self.sLy)}
                    fX_index = {val: r for r, val in enumerate(self.fX)}
                    wX_index = {val: r for r, val in enumerate(self.wX)}
                    fesc_index = {val: r for r, val in enumerate(self.fesc)}
                    Tmin_index = {val: r for r, val in enumerate(self.Tmin_vir)}
                    t_star_index = {val: r for r, val in enumerate(self.t_star)}

                    # Fill T21 array
                    for Ho_val, Omm_val, Omb_val, sig8_val, ns_val, Tcmb_val, Yp_val, fly_val, sly_val, fx_val, w_val, fesc_val, tmin_val, t_star_val, val in all_T21:
                        i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14 = Ho_index[Ho_val], Omm_index[Omm_val], Omb_index[Omb_val], sig8_index[sig8_val], ns_index[ns_val], Tcmb_index[Tcmb_val], Yp_index[Yp_val], fLy_index[fly_val], sLy_index[sly_val], fX_index[fx_val], wX_index[w_val], fesc_index[fesc_val], Tmin_index[tmin_val], t_star_index[t_star_val]
                        
                        T21_mod3[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, :] = val
                else:
                    #CDM, empirical
                    T21_mod3 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.a_sfrd),n_values))

                    # Create mapping from values to indices
                    Ho_index = {val: i for i, val in enumerate(self.Ho)}
                    Omm_index = {val: j for j, val in enumerate(self.Om_m)}
                    Omb_index = {val: k for k, val in enumerate(self.Om_b)}
                    sig8_index = {val: l for l, val in enumerate(self.sig8)}
                    ns_index = {val: m for m, val in enumerate(self.ns)}
                    Tcmb_index = {val: n for n, val in enumerate(self.Tcmbo)}
                    Yp_index = {val: o for o, val in enumerate(self.Yp)}

                    fLy_index = {val: r for r, val in enumerate(self.fLy)}
                    sLy_index = {val: r for r, val in enumerate(self.sLy)}
                    fX_index = {val: r for r, val in enumerate(self.fX)}
                    wX_index = {val: r for r, val in enumerate(self.wX)}
                    fesc_index = {val: r for r, val in enumerate(self.fesc)}
                    a_index = {val: r for r, val in enumerate(self.a_vir)}

                    # Fill T21 array
                    for Ho_val, Omm_val, Omb_val, sig8_val, ns_val, Tcmb_val, Yp_val, fly_val, sly_val, fx_val, w_val, fesc_val, a_val, val in all_T21:
                        i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13 = Ho_index[Ho_val], Omm_index[Omm_val], Omb_index[Omb_val], sig8_index[sig8_val], ns_index[ns_val], Tcmb_index[Tcmb_val], Yp_index[Yp_val], fLy_index[fly_val], sLy_index[sly_val], fX_index[fx_val], wX_index[w_val], fesc_index[fesc_val], a_index[a_val]
                        
                        T21_mod3[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, :] = val
            
                z_save_name = self.path+'one_plus_z'
                T21_save_name = self.path+'T21'
                xHI_save_name = self.path+'xHI'
                tau_save_name = self.path+'tau'

                np.save(z_save_name,Z_temp)
                np.save(T21_save_name,T21_mod3)
                np.save(xHI_save_name,xHI_mod3)
                np.save(tau_save_name,tau_mod3)     #Saving tau for cosmo and astro params varied

                print('\033[32m\nOutput saved into folder:',self.path,'\033[00m')
                
                et = time.process_time()
                # get the execution time
                elapsed_time = et - st
                print('\nProcessing time: %.2f seconds' %elapsed_time)
                #========================================================
                #Writing to a summary file

                myfile = self._write_summary(elapsed_time=elapsed_time)
                myfile.write('\n{} models generated'.format(n_mod))
                myfile.write('\nNumber of CPU(s) = {}'.format(self.n_cpu))
                myfile.write('\n')
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
        return None
    #End of function glob_sig               
#End of class pipeline
#========================================================================================================