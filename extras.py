from .const import *
import numpy as np
from .hmf import SFRD
from .basic_cosmo import *
import scipy.integrate as scint
import scipy.special as scsp

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


#--------------------------------------------------------------------------------------------
# The following functions are relevant to electromagnetically charged DM particles.
 
def u_t(xe,Tk,Tx, Yp=0.245,mx_gev=1,target='p'):
	'''
	Output: thermal velocity in m/s
	Input: 1. mx_gev should be in GeV
    	   2. T's should be in K
	'''
	mx = mx_gev*GeV2kg
	if (target == 'e'):
		return np.sqrt(kB*Tk/me+kB*Tx/mx)
	if (target == 'p'):
		return np.sqrt(kB*Tk/mu_bx(xe,Yp,mx_gev))

def r_t(xe,Tk,Tx,v_bx, Yp=0.245,mx_gev=1, target='p'):
	'''
    Output: ratio of relative velocity of DM and baryons with the thermal velocity (dimensionless)
    Input: 1. T's should be in K
    	   2. v_bx should be in m/s
    	   3. mx_gev should be in GeV
	'''
	if (target == 'e'):
		return v_bx/u_t(xe,Tk,Tx, Yp,mx_gev,'e')
	if (target == 'p'):
		return v_bx/u_t(xe,Tk,Tx, Yp,mx_gev,'p')

def F(r):
	return scsp.erf(r/np.sqrt(2))- np.sqrt(2/np.pi)*r*np.exp(-r**2/2)



def D(Z,xe,Tk,Tx,v_bx, Ho=67.4,Om_m=0.315,Om_b=0.049,Yp=0.245,fdm=1,mx_gev=1,sigma45=1):
	'''
	This is the drag term. Final result in m.s^-2
	fdm is the fraction of DM that is Coloumb like. Dimensionless
	mx is the mass of DM particle in GeV
	v_bx is the relative baryon and DM velocity in m/s
	T's are in K
	'''
	sigma0 = sigma45*sig_ten45m2   #in m^2
	mx = mx_gev*GeV2kg
	
	rho_b = Z**3*rho_crit(Ho)*Om_b
	rho_x = Z**3*fdm*rho_crit(Ho)*(Om_m-Om_b)
	#nH = rho_b/mp #Doubt #1/m^3
	return cE**4*sigma0*(rho_x+rho_b)/(mx+mu(xe,Yp)*mP) * F(r_t(xe,Tk,Tx,v_bx,Yp,mx_gev,'p'))/v_bx**2
	
def mu_bx(xe,Yp=0.245,mx_gev=1):
	'''
	xe is electron fraction (dimensionless)
	mx_gev should be in GeV
	output is reduced mass in kg
	'''
	mx = mx_gev*GeV2kg
	return mu(xe,Yp)*mP*mx/(mu(xe,Yp)*mP+mx)
	
	
