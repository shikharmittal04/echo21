import numpy as np

#========================================================================================================
#Universal constants in SI units
GN=6.67e-11 #Gravitational constant
cE=2.998e8  #Speed of light
kB=1.38e-23 #Boltzmann constant
hP=6.634e-34 #Planck's contant
mP=1.67e-27 #Mass of proton
me=9.1e-31 #Mass of electron
eC=1.6e-19 #Charge of electron
epsilon=8.85e-12 #Permittivity of free space

aS=7.52e-16 #Stephan's radiation constant
sigT=6.65e-29 #Thomson scattering cross-section, m^2

fnu = 0.68 #neutrino contribution to energy density in relativistic species; 3 massless nu's
#-------------------------------------------------------------

#Conversions
 
Mpc2km = 3.0857e19
Msolar = 1.989e30 #Mass of sun in kg
Msolar_by_Mpc3_to_kg_by_m3 = Msolar*(1000*Mpc2km)**-3
year = 365*86400
Msolar_by_Mpc3_year_to_kg_by_m3_sec = Msolar*(1000*Mpc2km)**-3*year**-1
Msolar_by_year_to_kg_by_sec = Msolar*year**-1

#-------------------------------------------------------------
#Hardcoded but later we want to change some of these

a = 1.127
b = 2.5
Mdot0 = 3     #Solar mass per year
L_UV = 8.695e20  #W/Hz/(Msun/yr)

fstar = 0.1
Iion = 10**53.44/Msolar_by_year_to_kg_by_sec #Yield of ionising photons. From Madau & Fragos (2017).

N_alpha_infty = 10000   #Total number of Lyman series photons between Ly-alpha and Ly-limit lines.

tilda_E1, tilda_E0, E1, E0 = 30,0.2,8,0.5 #Energies in keV
CX_fid = 2.61e32/Msolar_by_year_to_kg_by_sec #Lx-SFR relation in units of m^2/s^2.
#--------------------------------------------------------------------------------------------------

Zstar = 60 #redshift of the beginning of star formation

Z_start = 1501
Z_end = 1

Z_cd = np.concatenate((1/np.linspace(1/Zstar,1/5.05,200),1/np.linspace(1/5,1/Z_end,100)))
Z_default = np.concatenate((np.linspace(Z_start,Zstar+0.1,2000),Z_cd))
#-----------------------------------------------------------------------

#Recombination related

Lam_H = 8.22458 #The H 2s–1s two photon rate in s^−1
A_rec,b_rec,c_rec,d_rec = 4.309, -0.6166, 0.6703, 0.53
Feff = 1.14 #This extra factor gives the effective 3-level recombination model
lam_alpha = 121.5682e-9 #Wavelength of Lya photon in m
nu_alpha = cE/lam_alpha #Frequency in Hz
B2 = 3.4*eC #Binding energy of level 2 in J
B1 = 13.6*eC #Binding energy of level 1 in J
Ea = B1-B2  #Energy of Lya photon in J
A_alpha = 6.25e8 #Spontaneous emission coeffecient in s^−1
alpha_B = 2.941e-19 #Case-B recombination coefficient (m^3/s) at T= 10^4 K
#------------------------------------------------------------------------------

#Others
T_se = 0.4 #Spin exchange correction (in Kelvin; Chuzhoy & Shapiro 2006)
Tstar = 0.068 #Hyperfine energy difference in temperature (K)
A10 = 2.85e-15 # Einstein's spontaneous emission rate, sec^-1
Pn=np.array([0.2609,0.3078,0.3259,0.3353,0.3410,0.3448,0.3476,0.3496,0.3512,0.3524,0.3535,0.3543,0.355,0.3556,0.3561,0.3565,0.3569,0.3572,0.3575,0.3578])

#------------------------------------------------------------------------------
# Star formation related defaults

phy_sfrd_default_model = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':1e4}
emp_sfrd_default_model = {'type':'emp','a':0.257,'b':4}


#========================================================================================================