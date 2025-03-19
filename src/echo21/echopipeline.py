import numpy as np
from mpi4py import MPI
import sys
import time
import os
from scipy.interpolate import CubicSpline
from time import localtime, strftime
from pybaselines import Baseline

from .const import Zstar, Z_start, Z_end, Z_default, Z_cd, phy_sfrd_default_model, emp_sfrd_default_model
from .echofuncs import funcs


def _print_banner():
    banner = """\n\033[94m
    ███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  ██╗
    ██╔════╝██╔════╝██║  ██║██╔═══██╗╚════██╗███║
    █████╗  ██║     ███████║██║   ██║ █████╔╝╚██║
    ██╔══╝  ██║     ██╔══██║██║   ██║██╔═══╝  ██║
    ███████╗╚██████╗██║  ██║╚██████╔╝███████╗ ██║
    ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═╝
    Copyright 2025, Shikhar Mittal.                                     
    \033[00m\n"""
    print(banner)
    return None

def _to_array(params):
        try:
            for keys in params.keys():
                if type(params[keys])==list:
                    params[keys]=np.array(params[keys])
                elif type(params[keys])==float or type(params[keys])==int:
                    params[keys]=np.array([params[keys]])
        except:
            if type(params)==list:
                    params=np.array(params)
            elif type(params)==float or type(params)==int:
                params=np.array([params])        
        return params

def _to_float(params):
    try:
        for keys in params.keys():
            if type(params[keys])==list:
                [params[keys]]=params[keys]
            elif type(params[keys])==np.ndarray:
                params[keys]=params[keys][0]
    except:
        if type(params)==list:
                [params]=params
        elif type(params)==np.ndarray:
            params=params[0]
    return params
    
def _no_of_mdls(params):
    prod=1
    for keys in params.keys():
        if type(params[keys])==np.ndarray:
            prod=prod*len(params[keys])
    return prod

def _smoother(x,y):
    baseline_fitter = Baseline(x_data = x)
    y = baseline_fitter.imodpoly(y, poly_order=4)[0]
    return y

class pipeline():
    '''
    This class runs the cosmic history solver and produces the global signal and the corresponding redshifts.
    
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
        
        hmf : str, optional
            HMF model to use. Default value ``press74``. Other commonly used HMFs are
            
            - sheth99 (for Sheth & Tormen 1999)
            
            - tinker08 (for Tinker et al 2008)

        For the full list see `colossus <https://bdiemer.bitbucket.io/colossus/lss_mass_function.html#mass-function-models>`__ page.

        mdef: str, optional
            Definition for halo mass. Default is ``fof``. For most HMFs such as Press-Schechter or Sheth-Tormen friends-of-friends (``fof``) algorithm is used. For Tinker, it is an integer times mean matter density (``<int>m``). See colossus definition `page <https://bdiemer.bitbucket.io/colossus/halo_mass.html>`_
            
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.


    Methods
    ~~~~~~~
    '''
    def __init__(self,cosmo={'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'sig8':0.811,'ns':0.965,'Tcmbo':2.725,'Yp':0.245},astro= {'fLy':1,'sLy':2.64,'fX':1,'wX':1.5,'fesc':0.0106}, sfrd_dic={'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':1e4},Z_eval=None,path='echo21_outputs/'):

        self.comm = MPI.COMM_WORLD
        self.cpu_ind = self.comm.Get_rank()
        self.n_cpu = self.comm.Get_size()

        self.cosmo=cosmo
        self.astro=astro

        
        
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
        
        if self.model==0:
            self.astro=_to_float(self.astro)
            self.cosmo=_to_float(self.cosmo)
            if self.sfrd_type == 'phy': self.Tmin_vir = _to_float(self.Tmin_vir)
            else: self.a_sfrd = _to_float(self.a_sfrd)
        elif self.model==1:
            self.astro=_to_array(self.astro)
            self.cosmo=_to_float(self.cosmo)
            if self.sfrd_type == 'phy': self.Tmin_vir = _to_array(self.Tmin_vir)
            else: self.a_sfrd = _to_array(self.a_sfrd)
        elif self.model==2:
            self.astro=_to_float(self.astro)
            self.cosmo=_to_array(self.cosmo)
            if self.sfrd_type == 'phy': self.Tmin_vir = _to_float(self.Tmin_vir)
            else: self.a_sfrd = _to_float(self.a_sfrd)
        elif self.model==3:
            self.astro=_to_array(self.astro)
            self.cosmo=_to_array(self.cosmo)
            if self.sfrd_type == 'phy': self.Tmin_vir = _to_array(self.Tmin_vir)
            else: self.a_sfrd = _to_array(self.a_sfrd)
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
        


        



        self.path=path
        if self.cpu_ind==0:
            if os.path.isdir(self.path)==False:
                print('The requested directory does not exist. Creating ',self.path)
                os.mkdir(self.path)
            
            self.timestamp = strftime("%Y%m%d-%H%M%S", localtime())
            self.path = self.path + 'output_'+self.timestamp+'/'
            os.mkdir(self.path)

            self.formatted_timestamp = self.timestamp[9:11]+':'+self.timestamp[11:13]+':'+self.timestamp[13:15]+' '+self.timestamp[6:8]+'/'+self.timestamp[4:6]+'/'+ self.timestamp[:4]
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
        myfile.write('\n')
        myfile.write('\nParameters given:\n')
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
        try:
            myfile.write('\n  HMF = '+self.hmf)
            myfile.write('\n  mdef = '+self.mdef)
            myfile.write('\n  Tmin_vir = {}'.format(self.Tmin_vir))
        except:
            myfile.write('\n  a = {}'.format(self.a_sfrd))
        
        myfile.write('\n')
        return myfile

    def glob_sig(self):
        '''
        This function solves the thermal and ionization history for default values of redshifts and then interpolates the quantities at your choice of redshifts.  Then it solves reionization. Finally, it computes the spin temperature and hence the global 21-cm signal. A text file is generated which will contain the basic information about the simulation. 
        ''' 
        #completed = 0
        if self.model==0:
        #Cosmological and astrophysical parameters are fixed.
            if self.cpu_ind==0:
                _print_banner()
                print('Both cosmological and astrophysical parameters are fixed.\n')
                
                st = time.process_time()
                
                if self.sfrd_type == 'phy':
                    myobj = funcs(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,sig8=self.sig8,ns=self.ns,Tcmbo=self.Tcmbo,Yp=self.Yp,fLy=self.fLy,sLy=self.sLy,fX=self.fX,wX=self.wX,fesc=self.fesc,type = self.sfrd_type,hmf=self.hmf,mdef=self.mdef,Tmin_vir=self.Tmin_vir)
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

                Q_Hii = myobj.reion_solver()
                Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

                #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
                Tk[0:1806] = _smoother(Z_default[0:1806],Tk[0:1806])

                Q_Hii_default = Q_Hii  #We need this for computing CMB optical depth

                if self.Z_eval is not None:
                    splxe = CubicSpline(np.flip(Z_default), np.flip(xe))
                    xe = splxe(self.Z_eval)
                    Q_Hii = np.interp(self.Z_eval, np.flip(Z_default), np.flip(Q_Hii))
                    splTk = CubicSpline(np.flip(Z_default), np.flip(Tk))
                    Tk = splTk(self.Z_eval)

                print('Obtaining spin temperature ...')
                Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)

                print('Computing the 21-cm signal ...')
                T21_mod1 = myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)
                
                print('Done.')

                xe_save_name = self.path+'xe'
                Q_save_name = self.path+'Q'
                Q_default_save_name = self.path+'Q_default'
                Tk_save_name = self.path+'Tk'
                Ts_save_name = self.path+'Ts'
                Tcmb_save_name = self.path+'Tcmb'
                T21_save_name = self.path+'T21'
                z_save_name = self.path+'one_plus_z'
                
                np.save(xe_save_name,xe)
                np.save(Q_save_name,Q_Hii)
                np.save(Q_default_save_name,Q_Hii_default)
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
                max_T21 = np.min(T21_mod1)
                max_ind = np.where(T21_mod1==max_T21)
                [max_z] = Z_temp[max_ind]

                z50 = None
                try:
                    idx = np.argmin(np.abs(Q_Hii_default-0.5))
                    z50 = Z_default[idx]-1
                    z100 = None
                    try:
                        idx = np.where(Q_Hii_default>=0.98)[0][0]
                        z100 = Z_default[idx]-1
                        tau_e = myobj.reion_tau(60,Q_Hii_default)
                    except:
                        print('\n{:.1f} % universe reionised'.format(100*Q_Hii_default[-1]))
                except:
                    print('\nNote even 50% reionisation complete until today!')

                myfile = self._write_summary(elapsed_time=elapsed_time)
                
                if z50!=None:
                    myfile.write('\n50% reionisation complete at z = {:.2f}'.format(z50))
                    if z100!=None:
                        myfile.write("\nReionisation complete at z = {:.2f}".format(z100))
                        myfile.write("\nTotal Thomson-scattering optical depth = {:.4f}".format(tau_e))

                myfile.write('\n\nStrongest 21-cm signal is {:.2f} mK, observed at z = {:.2f}'.format(max_T21,max_z-1))
                myfile.write('\n')
                myfile.close()
                #========================================================

                print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
                return None
            
        elif self.model==1:
        #Cosmological parameters are fixed so dark ages is solved only once.
            if self.cpu_ind==0:
                _print_banner()
                print('Cosmological parameters are fixed. Astrophysical parameters are varied.')
                print('\nGenerating once the thermal and ionisation history for dark ages ...')
            
            myobj_da = funcs(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,sig8=self.sig8,ns=self.ns,Tcmbo=self.Tcmbo,Yp=self.Yp)

            Z_da = np.linspace(Z_start,Zstar,2000)
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
            
            n_mod = _no_of_mdls(self.astro)
            if self.sfrd_type=='phy':
                n_mod = n_mod * np.size(self.Tmin_vir)
                arr = np.arange(n_mod)
                arr = np.reshape(arr,[np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir)])
                T21_cd = np.zeros((np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir),n_values))
            elif self.sfrd_type=='emp':
                n_mod = n_mod * np.size(self.a_sfrd)
                arr = np.arange(n_mod)
                arr = np.reshape(arr,[np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.a_sfrd)])
                arr = np.reshape(arr,[np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.a_sfrd)])
                T21_cd = np.zeros((np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.a_sfrd),n_values))

            
            if self.cpu_ind==0: print('Done.\n\nGenerating',n_mod,'models for cosmic dawn ...\n')

            st = time.process_time()            
            
            
            
            
            if self.sfrd_type == 'phy':
            #This is the case for physically motivated SFRD.
                for i in range(n_mod):
                    if (self.cpu_ind == int(i/int(n_mod/self.n_cpu))%self.n_cpu):
                        ind=np.where(arr==i)

                        myobj_cd = funcs(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,sig8=self.sig8,ns=self.ns,Tcmbo=self.Tcmbo,Yp=self.Yp,fLy=self.fLy[ind[0][0]],sLy=self.sLy[ind[1][0]],fX=self.fX[ind[2][0]],wX=self.wX[ind[3][0]],fesc=self.fesc[ind[4][0]], type='phy', hmf=self.hmf, mdef = self.mdef, Tmin_vir=self.Tmin_vir[ind[5][0]])
                        
                        sol_cd = myobj_cd.igm_solver(Z_eval=Z_cd,xe_init=xe_da[-1],Tk_init=Tk_da[-1])
                        
                        xe_cd = sol_cd[0]
                        Tk_cd = sol_cd[1]

                        Q_cd = myobj_cd.reion_solver()

                        if self.Z_eval is not None:
                            splxe = CubicSpline(np.flip(Z_cd), np.flip(xe_cd))
                            xe_cd = splxe(self.Z_eval)
                            Q_cd = np.interp(self.Z_eval, np.flip(Z_cd), np.flip(Q_cd))
                            splTk = CubicSpline(np.flip(Z_cd), np.flip(Tk_cd))
                            Tk_cd = splTk(self.Z_eval)

                        Ts_cd= myobj_cd.hyfi_spin_temp(Z=Z_temp,xe=xe_cd,Tk=Tk_cd)
                        T21_cd[ind[0][0],ind[1][0],ind[2][0],ind[3][0],ind[4][0],ind[5][0],:]= myobj_cd.hyfi_twentyone_cm(Z=Z_temp,xe=xe_cd,Q=Q_cd,Ts=Ts_cd)
            else:
            #This is the case for empirically motivated SFRD.
                for i in range(n_mod):
                    if (self.cpu_ind == int(i/int(n_mod/self.n_cpu))%self.n_cpu):
                        ind=np.where(arr==i)

                        myobj_cd = funcs(Ho=self.Ho,Om_m=self.Om_m,Om_b=self.Om_b,sig8=self.sig8,ns=self.ns,Tcmbo=self.Tcmbo,Yp=self.Yp,fLy=self.fLy[ind[0][0]],sLy=self.sLy[ind[1][0]],fX=self.fX[ind[2][0]],wX=self.wX[ind[3][0]],fesc=self.fesc[ind[4][0]], type = 'emp', a=self.a_sfrd[ind[5][0]])
                        
                        sol_cd = myobj_cd.igm_solver(Z_eval=Z_cd,xe_init=xe_da[-1],Tk_init=Tk_da[-1])
                        
                        xe_cd = sol_cd[0]
                        Tk_cd = sol_cd[1]

                        Q_cd = myobj_cd.reion_solver()

                        if self.Z_eval is not None:
                            splxe = CubicSpline(np.flip(Z_cd), np.flip(xe_cd))
                            xe_cd = splxe(self.Z_eval)
                            Q_cd = np.interp(self.Z_eval, np.flip(Z_cd), np.flip(Q_cd))
                            splTk = CubicSpline(np.flip(Z_cd), np.flip(Tk_cd))
                            Tk_cd = splTk(self.Z_eval)

                        Ts_cd= myobj_cd.hyfi_spin_temp(Z=Z_temp,xe=xe_cd,Tk=Tk_cd)
                        T21_cd[ind[0][0],ind[1][0],ind[2][0],ind[3][0],ind[4][0],ind[5][0],:]= myobj_cd.hyfi_twentyone_cm(Z=Z_temp,xe=xe_cd,Q=Q_cd,Ts=Ts_cd)




            self.comm.Barrier()
            if self.cpu_ind!=0:
                self.comm.send(T21_cd, dest=0)
            else:
                print('\nDone.')
                for j in range(1,self.n_cpu):
                    T21_cd = T21_cd + self.comm.recv(source=j)
                
                T21_save_name = self.path+'T21'
                z_save_name = self.path+'one_plus_z'
                
                np.save(T21_save_name,T21_cd)
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

        elif self.model==2:

            if self.cpu_ind==0:
                _print_banner()
                print('Only cosmological parameters are varied.')
            

            Z_temp = Z_default
            if self.Z_eval is not None:
                if (self.Z_eval[0]>1501 or self.Z_eval[-1]<Z_end):
                    print('\033[31mYour requested redshift values should satisfy ',1501,'>1+z>',Z_end)
                    print('Terminating ...\033[00m')
                    sys.exit()
                else:
                    Z_temp = self.Z_eval

            n_values = len(Z_temp)
            
            n_mod = _no_of_mdls(self.cosmo)
            arr = np.arange(n_mod)
            arr = np.reshape(arr,[np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp)])
            T21_mod2 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),n_values))
            
            if self.cpu_ind==0: print('\nGenerating',n_mod,'models ...')
            st = time.process_time()
            
            if self.sfrd_type == 'phy':
            #This is the case for physically motivated SFRD.
                for i in range(n_mod):
                    if (self.cpu_ind == int(i/int(n_mod/self.n_cpu))%self.n_cpu):
                        ind=np.where(arr==i)

                        myobj = funcs(Ho=self.Ho[ind[0][0]],Om_m=self.Om_m[ind[1][0]],Om_b=self.Om_b[ind[2][0]],sig8=self.sig8[ind[3][0]],ns=self.ns[ind[4][0]],Tcmbo=self.Tcmbo[ind[5][0]],Yp=self.Yp[ind[6][0]],fLy=self.fLy,sLy=self.sLy,fX=self.fX,wX = self.wX, fesc=self.fesc, type='phy', Tmin_vir=self.Tmin_vir, mdef = self.mdef, hmf=self.hmf)
                        sol = myobj.igm_solver(Z_eval=Z_default)

                        xe = sol[0]
                        Tk = sol[1]

                        Q_Hii = myobj.reion_solver()
                        Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

                        #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
                        Tk[0:1806] = _smoother(Z_default[0:1806],Tk[0:1806])

                        if self.Z_eval is not None:
                            splxe = CubicSpline(np.flip(Z_default), np.flip(xe))
                            xe = splxe(self.Z_eval)
                            Q_Hii = np.interp(self.Z_eval, np.flip(Z_default), np.flip(Q_Hii))
                            splTk = CubicSpline(np.flip(Z_default), np.flip(Tk))
                            Tk = splTk(self.Z_eval)

                        Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)
                        T21_mod2[ind[0][0],ind[1][0],ind[2][0],ind[3][0],[ind[4][0]],[ind[5][0]],[ind[6][0]],:] = myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)
            else:
            #This is the case for empirically motivated SFRD.
                for i in range(n_mod):
                    if (self.cpu_ind == int(i/int(n_mod/self.n_cpu))%self.n_cpu):
                        ind=np.where(arr==i)

                        myobj = funcs(Ho=self.Ho[ind[0][0]],Om_m=self.Om_m[ind[1][0]],Om_b=self.Om_b[ind[2][0]],sig8=self.sig8[ind[3][0]],ns=self.ns[ind[4][0]],Tcmbo=self.Tcmbo[ind[5][0]],Yp=self.Yp[ind[6][0]],fLy=self.fLy,sLy=self.sLy,fX=self.fX,wX = self.wX, fesc=self.fesc, type='emp', a=self.a_sfrd)
                        sol = myobj.igm_solver(Z_eval=Z_default)

                        xe = sol[0]
                        Tk = sol[1]

                        Q_Hii = myobj.reion_solver()
                        Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

                        #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
                        Tk[0:1806] = _smoother(Z_default[0:1806],Tk[0:1806])

                        if self.Z_eval is not None:
                            splxe = CubicSpline(np.flip(Z_default), np.flip(xe))
                            xe = splxe(self.Z_eval)
                            Q_Hii = np.interp(self.Z_eval, np.flip(Z_default), np.flip(Q_Hii))
                            splTk = CubicSpline(np.flip(Z_default), np.flip(Tk))
                            Tk = splTk(self.Z_eval)

                        Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)
                        T21_mod2[ind[0][0],ind[1][0],ind[2][0],ind[3][0],[ind[4][0]],[ind[5][0]],[ind[6][0]],:] = myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)
            
            self.comm.Barrier()
            if self.cpu_ind!=0:
                self.comm.send(T21_mod2, dest=0)
            else:
                print('Done.\n')
                for j in range(1,self.n_cpu):
                    T21_mod2 = T21_mod2 + self.comm.recv(source=j)
                
                z_save_name = self.path+'one_plus_z'
                T21_save_name = self.path+'T21'
                
                np.save(T21_save_name,T21_mod2)
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

        elif self.model==3:

            if self.cpu_ind==0:
                _print_banner()
                print('Both cosmological and astrophysical parameters are varied.')
            

            Z_temp = Z_default
            if self.Z_eval is not None:
                if (self.Z_eval[0]>1501 or self.Z_eval[-1]<Z_end):
                    print('\033[31mYour requested redshift values should satisfy ',1501,'>1+z>',Z_end)
                    print('Terminating ...\033[00m')
                    sys.exit()
                else:
                    Z_temp = self.Z_eval

            n_values = len(Z_temp)
            
            n_mod = _no_of_mdls(self.astro)*_no_of_mdls(self.cosmo)

            if self.sfrd_type=='phy':
                n_mod = n_mod * np.size(self.Tmin_vir)
                arr = np.arange(n_mod)
                arr = np.reshape(arr,[np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir)])
                T21_mod3 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.Tmin_vir),n_values))
            else:
                arr = np.arange(n_mod)
                arr = np.reshape(arr,[np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.a_sfrd)])
                T21_mod3 = np.zeros((np.size(self.Ho),np.size(self.Om_m),np.size(self.Om_b),np.size(self.sig8),np.size(self.ns),np.size(self.Tcmbo),np.size(self.Yp),np.size(self.fLy),np.size(self.sLy),np.size(self.fX),np.size(self.wX),np.size(self.fesc),np.size(self.a_sfrd),n_values))
            
            if self.cpu_ind==0: print('\nGenerating',n_mod,'models ...')
            st = time.process_time()
            
            if self.sfrd_type=='phy':
                for i in range(n_mod):
                    if (self.cpu_ind == int(i/int(n_mod/self.n_cpu))%self.n_cpu):
                        ind=np.where(arr==i)

                        myobj = funcs(Ho=self.Ho[ind[0][0]],Om_m=self.Om_m[ind[1][0]],Om_b=self.Om_b[ind[2][0]],sig8=self.sig8[ind[3][0]],ns=self.ns[ind[4][0]],Tcmbo=self.Tcmbo[ind[5][0]],Yp=self.Yp[ind[6][0]],fLy=self.fLy[ind[7][0]],sLy=self.sLy[ind[8][0]],fX=self.fX[ind[9][0]],wX=self.wX[ind[10][0]],fesc=self.fesc[ind[11][0]],Tmin_vir=self.Tmin_vir[ind[12][0]], hmf=self.hmf, mdef = self.mdef, type = 'phy')
                        sol = myobj.igm_solver(Z_eval=Z_default)

                        xe = sol[0]
                        Tk = sol[1]

                        Q_Hii = myobj.reion_solver()
                        Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

                        #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
                        Tk[0:1806] = _smoother(Z_default[0:1806],Tk[0:1806])

                        if self.Z_eval is not None:
                            splxe = CubicSpline(np.flip(Z_default), np.flip(xe))
                            xe = splxe(self.Z_eval)
                            Q_Hii = np.interp(self.Z_eval, np.flip(Z_default), np.flip(Q_Hii))
                            splTk = CubicSpline(np.flip(Z_default), np.flip(Tk))
                            Tk = splTk(self.Z_eval)

                        Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)
                        T21_mod3[ind[0][0],ind[1][0],ind[2][0],ind[3][0],[ind[4][0]],[ind[5][0]],[ind[6][0]],[ind[7][0]],[ind[8][0]],[ind[9][0]],[ind[10][0]],[ind[11][0]],[ind[12][0]],:] = myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)
            else:
                for i in range(n_mod):
                    if (self.cpu_ind == int(i/int(n_mod/self.n_cpu))%self.n_cpu):
                        ind=np.where(arr==i)

                        myobj = funcs(Ho=self.Ho[ind[0][0]],Om_m=self.Om_m[ind[1][0]],Om_b=self.Om_b[ind[2][0]],sig8=self.sig8[ind[3][0]],ns=self.ns[ind[4][0]],Tcmbo=self.Tcmbo[ind[5][0]],Yp=self.Yp[ind[6][0]],fLy=self.fLy[ind[7][0]],sLy=self.sLy[ind[8][0]],fX=self.fX[ind[9][0]],wX=self.wX[ind[10][0]],fesc=self.fesc[ind[11][0]],a=self.a_sfrd[ind[12][0]], type = 'emp')
                        sol = myobj.igm_solver(Z_eval=Z_default)

                        xe = sol[0]
                        Tk = sol[1]

                        Q_Hii = myobj.reion_solver()
                        Q_Hii = np.concatenate((np.zeros(2000),Q_Hii))

                        #Because of the stiffness of the ODE at high z, we need to smoothen Tk.
                        Tk[0:1806] = _smoother(Z_default[0:1806],Tk[0:1806])

                        if self.Z_eval is not None:
                            splxe = CubicSpline(np.flip(Z_default), np.flip(xe))
                            xe = splxe(self.Z_eval)
                            Q_Hii = np.interp(self.Z_eval, np.flip(Z_default), np.flip(Q_Hii))
                            splTk = CubicSpline(np.flip(Z_default), np.flip(Tk))
                            Tk = splTk(self.Z_eval)

                        Ts = myobj.hyfi_spin_temp(Z=Z_temp,xe=xe,Tk=Tk)
                        T21_mod3[ind[0][0],ind[1][0],ind[2][0],ind[3][0],[ind[4][0]],[ind[5][0]],[ind[6][0]],[ind[7][0]],[ind[8][0]],[ind[9][0]],[ind[10][0]],[ind[11][0]],[ind[12][0]],:] = myobj.hyfi_twentyone_cm(Z=Z_temp,xe=xe,Q=Q_Hii,Ts=Ts)


            self.comm.Barrier()
            if self.cpu_ind!=0:
                self.comm.send(T21_mod3, dest=0)
            else:
                print('Done.\n')
                for j in range(1,self.n_cpu):
                    T21_mod3 = T21_mod3 + self.comm.recv(source=j)
                
                z_save_name = self.path+'one_plus_z'
                T21_save_name = self.path+'T21'
                
                np.save(z_save_name,Z_temp)
                np.save(T21_save_name,T21_mod3)
                

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