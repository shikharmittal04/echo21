'''
Shikhar Mittal
The is the main code which does two main tasks:
1) First it solves the ODEs to get xe and Tk,
2) Second it will use these to compute the global cosmological 21-cm signal. 
Here 
'''

from mpi4py import MPI
import numpy as np
import sys
import time
import os
from .cosmic_history import run_solver
from .extras import to_array, to_float, no_of_mdls
from .hyperfine import twentyone_cm
from .const import Zstar

comm = MPI.COMM_WORLD
cpu_ind = comm.Get_rank()
n_cpu = comm.Get_size()

def print_banner():
    banner = """\n\033[94m
	███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  ██╗
	██╔════╝██╔════╝██║  ██║██╔═══██╗╚════██╗███║
	█████╗  ██║     ███████║██║   ██║ █████╔╝╚██║
	██╔══╝  ██║     ██╔══██║██║   ██║██╔═══╝  ██║
	███████╗╚██████╗██║  ██║╚██████╔╝███████╗ ██║
	╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═╝                                         
    \033[00m\n"""
    print(banner)
    return None

def glob_sig(cosmo={'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'Tcmbo':2.725,'Yp':0.245},astro= {'falp':1,'fX':0.1,'fstar':0.1,'Tmin_vir':1e4},Z_eval=None,path=''):

	model = 0
	for keys in astro.keys():
		if np.size(astro[keys])>1:
			model = model+1
			break
			
	for keys in cosmo.keys():
		if np.size(cosmo[keys])>1:
			model = model+2
			break
	
	if model==0:
		astro=to_float(astro)
		cosmo=to_float(cosmo)
	elif model==1:
		astro=to_array(astro)
		cosmo=to_float(cosmo)
	elif model==3:
		astro=to_array(astro)
		cosmo=to_array(cosmo)
	else:
		print('Currently not designed to work with varying cosmological parameters only!')
		sys.exit()
			
	Ho = cosmo['Ho']
	Om_m = cosmo['Om_m']
	Om_b = cosmo['Om_b']
	Tcmbo = cosmo['Tcmbo']
	Yp = cosmo['Yp']
	
	falp = astro['falp']
	fX = astro['fX']
	fstar = astro['fstar']
	Tmin_vir = astro['Tmin_vir']
	
	if os.path.isdir(path)==False:
		print('The requested directory does not exist. Creating one ...')
		os.mkdir(path)
                
	if model==0:
	#Cosmological and astrophysical parameters are fixed.
		if cpu_ind==0:
			
			print_banner()
			
			if type(Z_eval)==np.ndarray or type(Z_eval)==list:
				Z_eval=np.array(Z_eval)
				if Z_eval[1]>Z_eval[0]:
					Z_eval = Z_eval[::-1]
			elif Z_eval==None:
				Z_eval = np.linspace(1501,6,1500)
			else:
				print('\033[31mError! Z_eval not recognised!\033[00m')
				sys.exit()
			
			st = time.process_time()
			
			print('Obtaining the thermal and ionisation history ...')
			sol = run_solver(Ho,Om_m,Om_b,Tcmbo,Yp,falp,fX,fstar,Tmin_vir,1501,6,Z_eval)
			
			print('Computing the 21-cm signal ...')
			T21 = twentyone_cm(Z_eval,sol.xe,sol.Tk, Ho,Om_m, Om_b,Tcmbo, Yp, falp,fstar,Tmin_vir)
			
			print('Done.')
			
			T21_save_name = path+'T21'
			z_save_name = path+'z'
			
			np.save(T21_save_name,T21)
			np.save(z_save_name,Z_eval)
			
			print('\033[32mYour T21s have been saved into file:',T21_save_name,'\033[00m')
			
			et = time.process_time()
			# get the execution time
			elapsed_time = et - st
			print('\nExecution time: %.2f seconds' %elapsed_time)
			print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
			return None
		
	elif model==1:
	#Cosmological parameters are fixed so dark ages is solved only once.
		if cpu_ind==0: print_banner()
		
		if(n_cpu==1):
			print("\033[91mBetter to parallelise. Eg. 'mpirun -np 4 python3 %s', where 4 specifies the number of tasks.\033[00m" %(sys.argv[0]))

		if cpu_ind==0: print('Generating once the thermal and ionisation history up to dark ages ...')
		Z_da = np.linspace(1501,Zstar,1400)
		sol_da = run_solver(Ho,Om_m,Om_b,Tcmbo,Yp,falp[0],fX[0],fstar[0],Tmin_vir[0],Z_start=1501,Z_end=Zstar, Z_eval=Z_da)
		xe_da = sol_da.xe
		Tk_da = sol_da.Tk

		if type(Z_eval)==np.ndarray or type(Z_eval)==list:
			Z_eval=np.array(Z_eval)
			if Z_eval[1]>Z_eval[0]:
				Z_eval = Z_eval[::-1]
			if Z_eval[0]>Zstar:
				print('Error: first value should be below or equal to Zstar (= 60)')
				sys.exit()
		elif Z_eval==None:
			Z_eval = np.linspace(Zstar,6,200)

		n_values=len(Z_eval)
		
		n_mod = no_of_mdls(astro)
		arr = np.arange(n_mod)
		arr = np.reshape(arr,[np.size(falp),np.size(fX),np.size(fstar),np.size(Tmin_vir)])
		T21 = np.zeros((np.size(falp),np.size(fX),np.size(fstar),np.size(Tmin_vir),n_values))
		
		if cpu_ind==0: print('Done.\nGenerating',n_mod,'21-cm global signals ...\n')
		st = time.process_time()
		for i in range(n_mod):
			if (cpu_ind == int(i/int(n_mod/n_cpu))%n_cpu):
				ind=np.where(arr==i)
				sol_cd = run_solver(Ho,Om_m,Om_b,Tcmbo,Yp,falp[ind[0][0]],fX[ind[1][0]],fstar[ind[2][0]],
				Tmin_vir[ind[3][0]],Zstar,6,Z_eval,xe_da[-1],Tk_da[-1])
				T21[ind[0][0],ind[1][0],ind[2][0],ind[3][0],:]= twentyone_cm(Z_eval,sol_cd.xe,sol_cd.Tk,Ho,Om_m,Om_b,Tcmbo,Yp,falp[ind[0][0]],fstar[ind[2][0]],Tmin_vir[ind[3][0]])
		
		comm.Barrier()
		if cpu_ind!=0:
			comm.send(T21, dest=0)
		else:
			print('Done.')
			for j in range(1,n_cpu):
				T21 = T21 + comm.recv(source=j)
			
			T21_save_name = path+'T21_'+str(np.size(falp))+str(np.size(fX))+str(np.size(fstar))+str(np.size(Tmin_vir))
			z_save_name = path+'z'
			
			np.save(T21_save_name,T21)
			np.save(z_save_name,Z_eval)
			print('\033[32m\nYour T21s have been saved into file:',T21_save_name,'\033[00m')
			
			et = time.process_time()
			# get the execution time
			elapsed_time = et - st
			print('\nProcessing time: %.2f seconds' %elapsed_time)
			print('\n\033[94m================ End of ECHO21 ================\033[00m\n')

	elif model==3:
		
		if cpu_ind==0: print_banner()
		
		if type(Z_eval)==np.ndarray or type(Z_eval)==list:
			Z_eval=np.array(Z_eval)
			if Z_eval[1]>Z_eval[0]:
				Z_eval = Z_eval[::-1]
			if Z_eval[0]>1501 or Z_eval[-1]<6:
				print('Error: redshift values not within the range')
				sys.exit()
		elif Z_eval==None:
			Z_eval = np.linspace(1501,6,2000)
		
		n_values=len(Z_eval)
			
		if(n_cpu==1):
			print('Error: you want to generate global signals for multiple parameter values.')
			print("Run as, say, 'mpirun -n 4 python3 %s', where 4 specifies the number of CPUs." %(sys.argv[0]))
			sys.exit()		
		
		n_mod = no_of_mdls(astro)*no_of_mdls(cosmo)
		arr = np.arange(n_mod)
		arr = np.reshape(arr,[np.size(Ho),np.size(Om_m),np.size(Om_b),np.size(Tcmbo),np.size(Yp),np.size(falp),np.size(fX),np.size(fstar),np.size(Tmin_vir)])
		T21 = np.zeros((np.size(Ho),np.size(Om_m),np.size(Om_b),np.size(Tcmbo),np.size(Yp),
		np.size(falp),np.size(fX),np.size(fstar),np.size(Tmin_vir),n_values))
		
		if cpu_ind==0: print('Generating',n_mod,'models ...')
		st = time.process_time()
		
		for i in range(n_mod):
			if (cpu_ind == int(i/int(n_mod/n_cpu))%n_cpu):
				ind=np.where(arr==i)
				sol = run_solver(Ho[ind[0][0]],Om_m[ind[1][0]],Om_b[ind[2][0]],Tcmbo[ind[3][0]],Yp[ind[4][0]],
				falp[ind[5][0]],fX[ind[6][0]],fstar[ind[7][0]],Tmin_vir[ind[8][0]],Z_start=1501,Z_end=6,Z_eval=Z_eval)
				T21[ind[0][0],ind[1][0],ind[2][0],ind[3][0],[ind[4][0]],[ind[5][0]],
				[ind[6][0]],[ind[7][0]],[ind[8][0]]:] = twentyone_cm(Z_eval,sol_cd.xe,sol_cd.Tk,Ho[ind[0][0]],Om_m[ind[1][0]],Om_b[ind[2][0]],Tcmbo[ind[3][0]],Yp[ind[4][0]],falp[ind[5][0]],fstar[ind[7][0]],Tmin_vir[ind[8][0]])
		
		comm.Barrier()
		if cpu_ind!=0:
			comm.send(T21, dest=0)
		else:
			print('Done.')
			for j in range(1,n_cpu):
				T21 = T21 + comm.recv(source=j)
			save_name = path+'T21_'+str(np.size(Ho))+str(np.size(Om_m))+str(np.size(Om_b))+str(np.size(Tcmbo))+str(np.size(Yp))+str(np.size(falp))+str(np.size(fX))+str(np.size(fstar))+str(np.size(Tmin_vir))+'.npy'
			np.save(save_name,T21)
			
			print('Your T21s have been saved into file:',save_name)
			
			et = time.process_time()
			# get the execution time
			elapsed_time = et - st
			print('\nProcessing time: %.2f seconds' %elapsed_time)
			print('\n\033[94m================ End of ECHO21 ================\033[00m\n')
				
#End of function glob_sig.

