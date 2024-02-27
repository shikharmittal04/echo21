#This has all the numbers required for computing the global 21-cm signal. I like to work in SI units so all the numbers are in SI units.

import numpy as np

#Universal constants
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

#-------------------------------------------------------------
#Cosmology related 
Mpc2km = 3.0857e19
Msolar = 1.989e30 #Mass of sun in kg

Msolar_by_Mpc3_to_kg_by_m3 = Msolar*(1000*Mpc2km)**-3

fnu = 0.68 #neutrino contribution to energy density in relativistic species; 3 massless nu's 
Zstar = 60 #redshift of the beginning of star formation

Z_start = 1501
Z_end = 6
Ngrid = 1500
Z_default = np.linspace(Z_start,Z_end,Ngrid)

#-------------------------------------------------------------
#Recombination related
Lam_H = 8.22458 #The H 2s–1s two photon rate in s^−1
A,b,c,d = 4.309, -0.6166, 0.6703, 0.53
Feff = 1.14 #This extra factor gives the effective 3-level recombination model
lam_alpha = 121.5682e-9 #Wavelength of Lya photon in m
nu_alpha = cE/lam_alpha #Frequency in Hz
B2 = 3.4*eC #Bind energy of level 2 in J
B1 = 13.6*eC #Bind energy of level 1 in J
Ea = B1-B2  #Energy of Lya photon in J
A_alpha = 6.25e8 #Spontaneous emission coeffecient in Hz
#-------------------------------------------------------------
#Others
Tstar = 0.068 #Hyperfine energy difference in temperature (K)
A10 = 2.85e-15 # Einstein's spontaneous emission rate, sec^-1
Pn=np.array([0.2609,0.3078,0.3259,0.3353,0.3410,0.3448,0.3476,0.3496,0.3512,0.3524,0.3535,0.3543,0.355,0.3556,
    0.3561,0.3565,0.3569,0.3572,0.3575,0.3578])
