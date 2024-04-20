'''
Shikhar Mittal
General comments:

All the heating/cooling terms in units of temperature, i.e., -1/H*d(Tk)/dt.
This in turn can be written as 2q/(3n_b.K_B.H(z)), where q is the volumetric heating rate.

'Z' is 1+z, where z is the redshift.
'Tk' is gas kinetic temperature in kelvin.
'xe' is the electron fraction defined as number density of electron relative to total hydrogen number density.

'''
from .const import *
from .basic_cosmo import *
from .hmf import dfcoll_dz
from .extras import *
from .hyperfine import spin_temp, cmb_coup
import numpy as np
import scipy.special as scsp
import scipy.integrate as scint

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


def Ecmb(Z,xe,Tk,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,Yp=0.245,falp=1,fstar=0.1,Tmin_vir=1e4,cosmo=None,astro=None):
	if cosmo!=None:
		Ho = cosmo['Ho']
		Om_m = cosmo['Om_m']
		Om_b = cosmo['Om_b']
		Tcmbo = cosmo['Tcmbo']
	
	if astro!=None:
		falp = astro['falp']
		fstar = astro['fstar']
		Tmin_vir = astro['Tmin_vir']
	
	Ts = spin_temp(Z,xe,Tk, Ho,Om_m,Om_b,Tcmbo,Yp,falp,fstar,Tmin_vir)  
	return (1-xe)/(1+xHe(Yp)+xe)*A10*(0.5/H(Z,Ho,Om_m,Tcmbo))*(Tcmb(Z,Tcmbo)/Ts-1)*Tstar*cmb_coup(Z,xe,Ts,Ho,Om_m,Om_b,Tcmbo,Yp)

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

def Ex(Z,xe,Ho=67.4,Om_m=0.315,Om_b=0.049,Tcmbo=2.725,fX=0.1,fstar=0.1,Tmin_vir=1e4,fdm=1,mx_gev=1,sigma45=1,cosmo=None,astro=None):
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
	
	return 5e5*fX*fstar*fXh(xe)*Z*np.abs(dfcoll_dz(Z,Ho,Om_m,Om_b,Tcmbo, Tmin_vir,fdm,mx_gev,sigma45))

#------------------------------------------------------------------------------------
def Ex2b(Z,xe,Tk,Tx,v_bx,Ho=67.4,Om_m=0.315,Om_b=0.049, Tcmbo=2.725,Yp=0.245, fdm=1,mx_gev=1,sigma45=1):
	'''
	This corresponds to the heat that flows into the baryonic system from the DM.
	fdm is the fraction of DM that is Coloumb like. Dimensionless
	mx_gev is the mass of DM particle in GeV
	v_bx is the relative baryon and DM velocity in m/s
	T's are in K
	Output in K
	'''
	sigma0 = sigma45*sig_ten45m2
	mx = mx_gev*GeV2kg
	
	rho_x = fdm*rho_crit(Ho)*(Om_m-Om_b)*Z**3	# fraction of DM which is coloumb-like (in kg/m^3 proper)
	rho_b = rho_crit(Ho)*Om_b*Z**3 #mass density of baryons only (in kg/m^3 proper)
	
	#nH = rho_b/mp        #Not relevant for Columb like models
	#rho_e = xe*me*nH     #Not relevant for Columb like models
	#rho_p = xe*mp*nH     #Not relevant for Columb like models
	nx = rho_x/mx
	#f_He = 0.08
	#part3 = nx*xe/(1+f_He)             #Not relevant for Columb like models
	#re = r_t(v,Tb,Tx,mx,epsilon, 'e')  #Not relevant for Columb like models
	rp = r_t(xe,Tk,Tx,v_bx, Yp,mx_gev,'p') #Dimensionless
	#ue = u_t(Tb, Tx,mx,epsilon, 'e')   #Not relevant for Columb like models
	up = u_t(xe,Tk,Tx, Yp,mx_gev,'p')
	term1 = cE**4*2*mu(xe,Yp)*mP*rho_x*sigma0*np.exp(-rp**2/2)*(Tx-Tk)/((mx+mu(xe,Yp)*mP)**2*np.sqrt(2*np.pi)*up**3)
	term2 = 1/kB*rho_x/(rho_x+rho_b)*mu_bx(xe,Yp,mx_gev)*v_bx*D(Z,xe,Tk,Tx,v_bx,Ho,Om_m,Om_b,Yp,fdm,mx_gev,sigma45)
	return 2/(3*H(Z, Ho,Om_m,Tcmbo))*(term1+term2)

def Eb2x(Z,xe,Tk,Tx,v_bx,Ho=67.4,Om_m=0.315,Om_b=0.049, Tcmbo=2.725,Yp=0.245,fdm=1, mx_gev=1, sigma45=1):
	'''
	This corresponds to the heat that flows into the DM from the baryonic system.
	fdm is the fraction of DM that is Coloumb like. Dimensionless
	mx_gev is the mass of DM particle in GeV
	v_bx is the relative baryon and DM velocity in m/s
	T's are in K
	Output in K
	'''
	sigma0 = sigma45*sig_ten45m2
	mx = mx_gev*GeV2kg
	
	rho_x = fdm*rho_crit(Ho)*(Om_m-Om_b)*Z**3	# fraction of DM which is coloumb-like (in kg/m^3 proper)
	rho_b = rho_crit(Ho)*Om_b*Z**3 #mass density of baryons only (in kg/m^3 proper)

	#         nH = rho_b/mp #Doubt
	#         rho_e = xe*me*nH
	#         rho_p = xe*mp*nH
	nx = rho_x/mx
	#         f_He = 0.08
	#         part3 = nx*xe/(1+f_He)
	#re = r_t(v,Tb,Tx,mx,epsilon, 'e')
	rp = r_t(xe,Tk,Tx,v_bx, Yp,mx_gev, 'p')
	#ue = u_t(Tb, Tx,mx,epsilon, 'e')
	up = u_t(xe,Tk,Tx, Yp,mx_gev, 'p')
	term1 = cE**4*2*mx*rho_b*sigma0*np.exp(-rp**2/2)*(Tk-Tx)/((mx+mu(xe,Yp)*mP)**2*np.sqrt(2*np.pi)*up**3)
	term2 = 1/kB*rho_b/(rho_x+rho_b)*mu_bx(xe,Yp,mx_gev)*v_bx*D(Z,xe,Tk,Tx,v_bx,Ho,Om_m,Om_b,Yp,fdm,mx_gev,sigma45)
	return 2/(3*H(Z,Ho,Om_m, Tcmbo))*(term1+term2)

