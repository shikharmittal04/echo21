import scipy.special as scsp
import scipy.integrate as scint
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.lss import mass_function

class echo():
	def __init__():
	
	class basic_cosmo():
		def mu(xe,Yp=0.245):
			'''
			Average baryon mass in amu
			'''
			return 4/(4-3*Yp+4*xe*(1-Yp))

		def xHe(Yp=0.245):
			'''
			Ratio of He number to H number; n_He/n_H
			'''
			return 0.25*Yp/(1-Yp)

		def Tcmb(Z,Tcmbo=2.725):
			'''
			CMB temperature in K
			'''
			return Tcmbo*Z

		def rho_crit(Ho=67.4):
			'''
			Critical density of the Universe today; kg.m^-3 
			'''
			return 3*Ho**2/(8*np.pi*GN*Mpc2km**2)

		def nH(Z,Ho=67.4,Om_b=0.049,Yp=0.245):
			'''
			Hydrogen number density (proper); m^-3
			'''
			return rho_crit(Ho)*Om_b*(1-Yp)*Z**3/mP

		def H(Z,Ho=67.4,Om_m=0.315, Tcmbo=2.725):
			'''
			Hubble factor in SI units; sec^-1
			'''
			Om_lam = 1-Om_m
			Om_r = (1+fnu)*aS*Tcmbo**4/(cE**2*rho_crit(Ho))
			return Ho*(Om_r*Z**4+Om_m*Z**3+Om_lam)**0.5/Mpc2km
			
	class recomb():
		def alpha(Tk):
			'''
			:math:`\\alpha_{\\mathrm{B}}=\\alpha_{\\mathrm{B}}(T)`
			
			The effective case-B recombination coefficient for hydrogen. See Eq. (70) from `Seager et al (2000) <https://iopscience.iop.org/article/10.1086/313388>`__
			
			Arguments
			---------
			
			Tk : float
				Gas temperature in units of Kelvin
			
			Returns
			-------
			
			float
				The effective case-B recombination coefficient for hydrogen :math:`(\\mathrm{m}^3\\mathrm{s}^{-1})`
				
			'''
			t=Tk/10000
			return (1e-19)*Feff*A*t**b/(1+c*t**d)

		def beta(Tk): #Photo-ionization rate
			return alpha(Tk)*(2*np.pi*me*kB*Tk/hP**2)**1.5*np.exp(-B2/(kB*Tk))

		def Krr(Z,Ho,Om_m, Tcmbo): #Redshifting rate
			return lam_alpha**3/(8*np.pi*H(Z,Ho,Om_m, Tcmbo))

		def Peebles_C(Z,xe,Tk, Ho,Om_m,Om_b,Tcmbo,Yp):
			'''
			:math:`C_{\mathrm{P}}`
			
			Arguments
			---------
			
			Z : float
				1 + redshift, dimensionless
			
			xe : float
				Electron fraction, dimensionless
				
			Tk : float
				Gas temperature in units of Kelvin
			
			Returns
			-------
			
			float
				Peebles 'C' factor appearing in Eq. (71) from `Seager et al (2000) <https://iopscience.iop.org/article/10.1086/313388>`__, dimensionless.
			'''
			
			return (1+Krr(Z,Ho,Om_m, Tcmbo)*Lam_H*nH(Z, Ho,Om_b,Yp)*(1-xe))/(1+Krr(Z, Ho,Om_m, Tcmbo)*(Lam_H+beta(Tk))*nH(Z, Ho,Om_b,Yp)*(1-xe))

		def Saha_xe(Z,Tk, Ho,Om_b,Yp):
		
			S=1/nH(Z, Ho,Om_b,Yp)*(2*np.pi*me*kB*Tk/hP**2)**1.5*np.exp(-B1/(kB*Tk))
			return (np.sqrt(S**2+4*S)-S)/2
	
	class hmf():
		def sfr_model(hmf_name='press74',sfe_name='const'):
			global hmf_model
			hmf_model=hmf_name
			global sfe_model
			sfe_model=sfe_name
			return
			
		def dndlnM(M,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Tmin_vir=1e4,cosmo=None,astro=None):
			'''
			M is the halo mass in units of solar mass
			Output is dn/d(ln M)=M.dn/dM in units of cMpc**-3
			This function is usually what people refer to, when they write 'HMF'
			'''
			if cosmo!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
			if astro!=None:
				Tmin_vir = astro['Tmin_vir']
				
			my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
			cosmology.setCosmology('my_cosmo', my_cosmo)
			h100=Ho/100
			M_by_h = M*h100 #M in units of solar mass/h
			return h100**3*mass_function.massFunction(M_by_h, Z-1, q_in='M', q_out='dndlnM', model = hmf_model)

		def dndM(M,Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Tmin_vir=1e4,cosmo=None):
			'''
			M in the halo mass in units of solar mass
			Output is dn/dM in units of cMpc**-3.Msun**-1
			'''
			if cosmo!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
				Tmin_vir = astro['Tmin_vir']
			
			my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
			cosmology.setCosmology('my_cosmo', my_cosmo)
			return 1/M*dndlnM(M,Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir)

		#For details see eq.(50),(52) and (53) from Mittal & Kulkarni (2021), MNRAS
		def m_min(Z,Om_m=0.315,Tmin_vir=1e4):
			'''
			This function gives the minimum halo mass for star formation
			The mass returned is in units of solar mass/h
			Optional parameters are relative matter density & miminum virial temperature (Tmin_vir in K)
			'''
			return 1e8*Om_m**(-0.5)*(10/Z*0.6/1.22*Tmin_vir/1.98e4)**1.5

		def f_coll(Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Tmin_vir=1e4,cosmo=None,astro=None):
			'''
			Collapse fraction: fraction of total matter that collapsed into the haloes
			'''
			if cosmo!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
			if astro!=None:
				Tmin_vir = astro['Tmin_vir']
			
			if hmf_model=='press74':
				my_cosmo = {'flat': True, 'H0': Ho, 'Om0': Om_m, 'Ob0': Om_b, 'sigma8': 0.811, 'ns': 0.965,'relspecies': True,'Tcmb0': Tcmbo}
				cosmology.setCosmology('my_cosmo', my_cosmo)
				return scsp.erfc(peaks.peakHeight(m_min(Z,Om_m,Tmin_vir),Z-1)/np.sqrt(2))
			else:
				h100=Ho/100
				numofZ = np.size(Z)
				
				if numofZ == 1:
					if type(Z)==np.ndarray: Z=Z[0]
					M_space = np.logspace(np.log10(m_min(Z,Om_m,Tmin_vir)/h100),18,1500)	#These masses are in solar mass. 
					hmf_space = dndlnM(M=M_space,Z=Z,Ho=Ho,Om_m=Om_m,Om_b=Om_b,Tcmbo=Tcmbo,Tmin_vir=Tmin_vir)	#Corresponding HMF values are in cMpc^-3 
					rho_halo = Msolar_by_Mpc3_to_kg_by_m3*np.trapz(hmf_space,M_space)	#matter density collapsed as haloes (in kg/m^3, comoving)
				else:	
					rho_halo = np.zeros(numofZ)
					counter=0
					for i in Z:
						M_space = np.logspace(np.log10(m_min(i,Om_m,Tmin_vir)/h100),18,1500)	#These masses are in solar mass. 
						hmf_space = dndlnM(M=M_space,Z=i,Ho=Ho,Om_m=Om_m,Om_b=Om_b,Tcmbo=Tcmbo,Tmin_vir=Tmin_vir)	#Corresponding HMF values are in cMpc^-3 
						rho_halo[counter] = Msolar_by_Mpc3_to_kg_by_m3*np.trapz(hmf_space,M_space)	#matter density collapsed as haloes (in kg/m^3, comoving)
						counter=counter+1
				return rho_halo/(Om_m*rho_crit(Ho))
			


		def dfcoll_dz(Z,Ho,Om_m,Om_b,Tcmbo, Tmin_vir):
			'''
			Derivative of the collapse fraction
			'''
			return (f_coll(Z+1e-3, Ho, Om_m,Om_b,Tcmbo, Tmin_vir)-f_coll(Z, Ho, Om_m,Om_b,Tcmbo, Tmin_vir))*1e3

		 
		def SFRD(Z,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
			'''
			This function returns the comoving star formation rate density in units of kg/s/m^3
			'''
			if cosmo!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
			if astro!=None:
				fstar = astro['fstar']
				Tmin_vir = astro['Tmin_vir']
			
			return -Z*fstar*Om_b*rho_crit(Ho)*dfcoll_dz(Z,Ho,Om_m,Om_b,Tcmbo,Tmin_vir)*H(Z, Ho,Om_m, Tcmbo)

	class cosmic_history():
		class xe_Tk():
			def __init__(self,xe,Tk):
				self.xe=xe
				self.Tk=Tk

		#The following function has the differential equations governing the ionisation and thermal history.
		def _eqns(Z,V,Ho,Om_m,Om_b,Tcmbo,Yp,falp,fX,fstar,Tmin_vir):
			'''
			When solving upto the end of dark ages, only cosmological parameters will be used.
			Beyond Zstar, i.e., beginning of cosmic dawn astrophysical will also be used.
			'''
			xe = V[0]
			Tk = V[1]
			
			#eq1 is (1+z)d(xe)/dz; see Weinberg's Cosmology book or eq.(71) from Seager et al (2000), ApJSS
			eq1 = 1/H(Z,Ho,Om_m, Tcmbo)*Peebles_C(Z,xe,Tk, Ho,Om_m,Om_b,Tcmbo,Yp)*(xe**2*nH(Z,Ho,Om_b,Yp)*alpha(Tk)-beta(Tk)*(1-xe)*np.exp(-Ea/(kB*Tk)))
			
			#eq2 is (1+z)dT/dz; see eq.(2.31) from Mittal et al (2022), JCAP
			
			if Z>Zstar:
				eq2 = 2*Tk-Tk*eq1/(1+xHe(Yp)+xe)-Ecomp(Z,xe,Tk,Ho,Om_m,Tcmbo,Yp)
			else:
				eq2 = 2*Tk-Tk*eq1/(1+xHe(Yp)+xe)-Ecomp(Z,xe,Tk,Ho,Om_m,Tcmbo,Yp)-Ex(Z,xe,Ho,Om_m,Om_b,Tcmbo,fX,fstar,Tmin_vir)-Elya(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp,falp,fstar,Tmin_vir)
			
			return np.array([eq1,eq2])

		def run_solver(Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fX=0.1,fstar=0.1,Tmin_vir=1e4,Z_start=1501,Z_end=6,Z_eval=Z_default, xe_init=None,Tk_init=None,cosmo=None, astro=None):
			
			if cosmo!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
				Yp = cosmo['Yp']
			if astro!=None:
				falp = astro['falp']
				fX = astro['fX']
				fstar = astro['fstar']
				Tmin_vir = astro['Tmin_vir']

			if Z_start==1501:
				Tk_init = Tcmb(Z_start,Tcmbo)
				xe_init = Saha_xe(Z_start,Tk_init, Ho,Om_b,Yp)
			elif xe_init==None and Tk_init==None:
				raise Exception('Initial conditions missing.')
				
			Sol = scint.solve_ivp(lambda a, Var: -_eqns(1/a,Var,Ho,Om_m,Om_b,Tcmbo,Yp,falp,fX,fstar,Tmin_vir)/a, [1/Z_start, 1/Z_end],[xe_init,Tk_init],method='Radau',t_eval=1/Z_eval) 
			
			#Obtaining the solutions ...
			xe=Sol.y[0]
			Tk=Sol.y[1]

			return xe_Tk(xe,Tk)

	class hyperfine():
	
		def kHH(Tk):
			'''
			Volumetric spin flip rate for H-H collision; m^3/s
			This fitting function and the next one is available from Pritchard & Loeb (2012)
			'''
			return 3.1*10**(-17)*Tk**0.357*np.exp(-32/Tk)

		def keH(Tk):
			'''
			Volumetric spin flip rate for e-H collision; m^3/s
			'''
			return np.where(Tk<10**4,10**(-15.607+0.5*np.log10(Tk)*np.exp(-(np.abs(np.log10(Tk)))**4.5/1800) ),10**(-14.102))

		def col_coup(Z,xe,Tk, Ho=67.4,Om_b=0.049,Tcmbo=2.725,Yp=0.245,cosmo=None):
			'''
			Collisional coupling, x_c; dimensionless
			'''
			if cosmo!=None:
				Ho = cosmo['Ho']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
				Yp = cosmo['Yp']

			return Tstar*nH(Z,Ho,Om_b,Yp)*((1-xe)*kHH(Tk)+xe*keH(Tk))/(A10*Tcmb(Z,Tcmbo))

		def lya_coup(Z,xe,Tk, Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
			'''
			Ly-alpha coupling or the Wouthuysen-Field coupling, x_alpha; dimensionless.
			'''
			if cosmo!=None and astro!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
				Yp = cosmo['Yp']
				
				falp = astro['falp']
				fstar = astro['fstar']
				Tmin_vir = astro['Tmin_vir']
			
			S = scatter_corr(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)
			J = lya_spec_inten(Z,xe,Ho,Om_m,Om_b,Tcmbo,falp,fstar,Tmin_vir)	#'undistorted' background Spec. Inte. of Lya photons.
			Jo = 5.54e-8*Z 		#eq.(24) in Mittal & Kulkarni (2021)
			return S*J/Jo

		def spin_temp(Z,xe,Tk, Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
			'''
			Spin temperature in K.
			'''
			if cosmo!=None and astro!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
				Yp = cosmo['Yp']
				
				falp = astro['falp']
				fstar = astro['fstar']
				Tmin_vir = astro['Tmin_vir']

			xa = lya_coup(Z,xe,Tk, Ho,Om_m,Om_b,Tcmbo,Yp, falp,fstar,Tmin_vir)
			xk = col_coup(Z,xe,Tk, Ho,Om_b,Tcmbo,Yp)
			Ts = ( 1  + xa + xk)/(1/Tcmb(Z, Tcmbo) +  (xk+xa)/Tk )	#We assume the colour temperature is same as Tk.
			return Ts

		def Terb(Z,Tcmbo,zeta_erb): #Net background temperature (includes CMB)
			return np.where(Z<Zstar,Tcmbo*Z*(1+0.169*zeta_erb*Z**2.6),Tcmb(Z,Tcmbo))

		def twentyone_cm(Z,xe,Tk, Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
			'''
			The global (sky-averaged) 21-cm signal in mK.
			'''
			if cosmo!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
				Yp = cosmo['Yp']
			if astro!=None:
				falp = astro['falp']
				fstar = astro['fstar']
				Tmin_vir = astro['Tmin_vir']

			xHI=(1-xe)
			h100 = Ho/100
			Ts = spin_temp(Z,xe,Tk, Ho, Om_m, Om_b, Tcmbo, Yp,falp,fstar,Tmin_vir)
			return 27*xHI*((1-Yp)/0.76)*(Om_b*h100**2/0.023)*np.sqrt(0.15*Z/(10*Om_m*h100**2))*(1-Tcmb(Z)/Ts)

	class extras():
	
		def to_array(params):
			for keys in params.keys():
				if type(params[keys])==list:
					params[keys]=np.array(params[keys])
				elif type(params[keys])==float or type(params[keys])==int:
					params[keys]=np.array([params[keys]])
			return params

		def to_float(params):
			for keys in params.keys():
				if type(params[keys])==list:
					[params[keys]]=params[keys]
				elif type(params[keys])==np.ndarray:
					params[keys]=params[keys][0]
			return params
			
		def no_of_mdls(params):
			prod=1
			for keys in params.keys():
				if type(params[keys])==np.ndarray:
					prod=prod*len(params[keys])
			return prod

		def recoil(Tk):
			'''
			The recoil parameter. Eq.(15) in Mittal & Kulkarni (2021).
			'''
			return 0.02542/np.sqrt(Tk)

		def dopp(Tk):
			'''
			Doppler width for Lya-HI interaction. Eq.(14) in Mittal & Kulkarni (2021).
			'''
			return nu_alpha*np.sqrt(2*kB*Tk/(mP*cE**2))

		def a_tau(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp):
			'''
			Returns the product a*tau, since all the relevant formulae require the product only.
			a is the Voigt parameter and tau is the optical depth of Lya photons.
			'''
			tau = 3/(8*np.pi)*A_alpha/H(Z,Ho,Om_m,Tcmbo)*nH(Z,Ho,Om_b,Yp)*(1-xe)*lam_alpha**3
			a = A_alpha/(4*np.pi*dopp(Tk))
			return a*tau
			
		def zeta(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp):
			'''
			A dimensionless number. See below Eq.(12) in Chuzhoy & Shapiro (2006).
			'''
			return 4/3*np.sqrt(a_tau(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)*recoil(Tk)**3/np.pi)

		def xi2(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp):
			'''
			A dimensionless number. Eq.(39) in Mittal & Kulkarni (2021).
			'''
			return (4*a_tau(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)*recoil(Tk)**3/np.pi)**(1/3)

		def scatter_corr(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp):
			'''
			This is the scattering correction, S. I am using the approximate version from Chuzhoy & Shapiro (2006).
			'''
			return np.exp(-1.69*zeta(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)**0.667)
			
		def _eps_alpha_beta(Z,E, Ho,Om_m,Om_b,Tcmbo,fstar,Tmin_vir):
			phi = hP/eC*2902.91*(E/13.6)**-0.86
			return 1/(1.22*mP)*phi*SFRD(Z,Ho,Om_m,Om_b,Tcmbo,fstar,Tmin_vir)

		def _eps_above_beta(Z,E, Ho,Om_m,Om_b,Tcmbo,fstar,Tmin_vir):
			'''
			Comoving emissivity in units of number per unit time per unit frequency per unit volume (s^-1.m^-3.Hz^-1)
			Valid only for photons of energy above Ly beta, i.e., E > 12.089 eV
			'''
			phi = hP/eC*1303.34*(E/13.6)**-7.658 #this is the SED in units of number per baryon per unit frequency (Hz^-1)
			return 1/(1.22*mP)*phi*SFRD(Z,Ho,Om_m,Om_b,Tcmbo,fstar,Tmin_vir)

		def lya_spec_inten(Z,xe,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,falp=1,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
			'''
			Specific intensity of Ly-alpha photons in terms of number per unit time per unit area per unit frequency per unit solid angle
			(m^-2.s^-1.Hz^-1.sr^-1)
			'''
			if cosmo!=None and astro!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
				
				falp = astro['falp']
				fstar = astro['fstar']
				Tmin_vir = astro['Tmin_vir']
				
			loc=0
			flag=False
			integ=0
			if type(Z)==float or type(Z)==int:
				if Z>Zstar:
					return 0
				Zmax = 32/27*Z
				temp = np.linspace(Z,Zmax,10)
				integ = scint.trapz(_eps_alpha_beta(temp,10.2*temp/Z, Ho,Om_m,Om_b,Tcmbo,fstar,Tmin_vir)/H(temp, Ho, Om_m,Tcmbo),temp)
				for ni in np.arange(4,24):
					Zmax = (1-1/(ni+1)**2)/(1-1/ni**2)*Z
					temp = np.linspace(Z,Zmax,5)
					integ = integ+Pn[ni-4]*scint.trapz(_eps_above_beta(temp,13.6*(1-1/ni**2)*temp/Z,Ho,Om_m,Om_b,Tcmbo,fstar,Tmin_vir)/H(temp, Ho, Om_m,Tcmbo),temp)
			
			elif type(Z)==np.ndarray or type(Z)==list:
				if Z[0]>Zstar:
					flag=True
					loc = np.where(Z<Zstar)[0][0]
					Z=Z[loc:]
				
				counter=0
				numofZ = len(Z)
				integ=np.zeros(numofZ)
				for Z_value in Z:
					Zmax = 32/27*Z_value
					temp = np.linspace(Z_value,Zmax,10)
					integ[counter] = scint.trapz(_eps_alpha_beta(temp,10.2*temp/Z_value, Ho,Om_m,Om_b,Tcmbo,fstar,Tmin_vir)/H(temp, Ho, Om_m,Tcmbo),temp)

					for ni in np.arange(4,24):
						Zmax = (1-1/(ni+1)**2)/(1-1/ni**2)*Z_value
						temp = np.linspace(Z_value,Zmax,5)
						integ[counter] = integ[counter]+Pn[ni-4]*scint.trapz(_eps_above_beta(temp,13.6*(1-1/ni**2)*temp/Z_value,Ho,Om_m,Om_b,Tcmbo,fstar,Tmin_vir)/H(temp, Ho, Om_m,Tcmbo),temp)
					
					counter=counter+1
			

			J_temp = falp*cE/(4*np.pi)*Z**2*integ
			if flag == True:
				J_before_CD = np.zeros(loc)
				J_after_CD = J_temp
				return np.concatenate((J_before_CD,J_after_CD))
			else:
				return J_temp
	
	class heating():
		
		def Ecomp(Z,xe,Tk,Ho=67.4,Om_m=0.315,Tcmbo=2.725,Yp=0.245,cosmo=None):
			'''
			See eq.(2.32) from Mittal et al (2022), JCAP.
			(However, there is a minor typo in that equation; numerator has an xe missing.)
			'''
			if cosmo!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Tcmbo = cosmo['Tcmbo']
				Yp = cosmo['Yp']
			
			return (8*sigT*aS)/(3*me*cE)*Tcmb(Z,Tcmbo)**4*xe*(Tcmb(Z,Tcmbo)-Tk)/(H(Z,Ho,Om_m,Tcmbo)*(1+xHe(Yp)+xe))


		def Elya(Z,xe,Tk,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fstar=0.1,Tmin_vir=1e4):
			'''
			Lya heating rate. For details see Mittal & Kulkarni (2021), MNRAS
			'''
			eta = recoil(Tk)
			S = scatter_corr(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)
			atau = a_tau(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp)
			arr = scsp.airy(-xi2(Z,xe,Tk,Ho,Om_m,Om_b,Tcmbo,Yp))
			
			Ic = eta*(2*np.pi**4*atau**2)**(1/3)*(arr[0]**2+arr[2]**2)
			Ii = eta*np.sqrt(atau/2)*scint.quad(lambda y:y**(-1/2)*np.exp(-2*eta*y-np.pi*y**3/(6*atau))*scsp.erfc(np.sqrt(np.pi*y**3/(2*atau))),0,np.inf)[0]-S*(1-S)/(2*eta)
			J = lya_spec_inten(Z,xe,Ho,Om_m,Om_b,Tcmbo,falp,fstar,Tmin_vir)
			nbary = (1+xHe(Yp))*nH(Z,Ho,Om_b,Yp)
			return 8*np.pi/3 * hP/(kB*lam_alpha) * J*dopp(Tk)/nbary * (Ic+Ii)
		   
		'''
		def tau(E,Z,Z1,x_HI):     #X-ray optical depth
				Z2=np.linspace(Z,Z1[1:,],20)
				taux=(Z/E)**3*7341856114*x_HI**(1/3)*scint.trapz(1/(Z2*H(Z2)) ,Z2,axis=0)
				return np.insert(taux,0,0)

		def Jx(E,Z,x_HI):        #Number of X-ray photons/(area-time-solid angle-energy(in eV))
				Steps=int(5*(31-Z))
				Z1=np.linspace(Z,31,Steps+1)
				return 944580047*Z**2*scint.trapz(epsilon_x(E*Z1/Z,Z1)*np.exp(-tau(E,Z,Z1,x_HI))/H(Z1),Z1)

		def epsilon_x(E,Z):
				return 10**log_fx*(w-1)/Eo*(E/Eo)**(-1-w)*SFRD(Z)/(1-(Eo/3e4)**(w-1))

		def sig(E):           #Phototionisation cross section of hydrogen, but took the constant factor into Γx and Ex
				X=E/0.4298
				return (X-1)**2*X**-4.0185/(1+np.sqrt(X/32.88))**2.963

		def Gam_and_Ex(Z,x_e):      #Photoheating
				mu=4/(4-3*Yp+4*x_e*(1-Yp))
				x_HI=1-x_e
				if x_e<1.0:
				        f_heat=1-(1-x_e**0.2663)**1.3163
				else:
				        f_heat=1

				f_ion=0.3908*(1-x_e**0.4092)**1.7592
				Γx=4*np.pi*scint.quad(lambda E: sig(E)*Jx(E,Z,x_HI),Eo,30000)[0]
				Hx=4*np.pi*scint.quad(lambda E:(E-Ei)*sig(E)*Jx(E,Z,x_HI),Eo,30000)[0]
				Gam_x=(Γx+f_ion*Hx/Ei)*168.94/mu
				Ex=f_heat*1306992.8*(1-Yp)/H(Z)*(1-x_e)*Hx
				return np.array([Gam_x,Ex])
		'''

		def Ex(Z,xe,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,fX=0.1,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
			'''
			See eq. (11) from Furlanetto (2006)
			'''
			def fXh(xe):
				return 1-(1-xe**0.2663)**1.3163
			
			if cosmo!=None and astro!=None:
				Ho = cosmo['Ho']
				Om_m = cosmo['Om_m']
				Om_b = cosmo['Om_b']
				Tcmbo = cosmo['Tcmbo']
				
				fX = astro['fX']
				fstar = astro['fstar']
				Tmin_vir = astro['Tmin_vir']
			
			return 5e5*fX*fstar*fXh(xe)*Z*np.abs(dfcoll_dz(Z,Ho,Om_m,Om_b,Tcmbo, Tmin_vir))


