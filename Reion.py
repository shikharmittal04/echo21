#Evolution of ionised fraction in HII region, denoted by Q.
#Reionisation model parameters and functions, for e.g. clumping factor,...
#are taken from "https://doi.org/10.3847/1538-4357/aa6af9".
#This code also gives a fitting to the reionisation curve.
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
import scipy.optimize as op

Ho=67.4    #Hubble's constant in units of 1km/s/Mpc
To=2.725
Yp=0.245
Ω_B,Ω_m,Ω_Λ=0.049,0.315,0.685
Ω_R=7.531e-3*To**4/Ho**2

f_esc=1

Z_start=51
#-----------------------------------------------------------------------------------
t_0=13.8
def Z2t(Z):
    return t_0*Z**(-1.5)

def t2Z(t):
    return (t/t_0)**(-2/3)
#------------------------------------------------------------------------------------

def n_H(z):#Hydrogen number density
    return 1.1227e-3*Ho**2*Ω_B*(1-Yp)*(1+z)**3

def H(z):#Hubble function
    return Ho*(Ω_R*(1+z)**4+Ω_m*(1+z)**3+Ω_Λ)**0.5

def SFRD(z):
    return 0.01*(1+z)**2.6/(1+7.38e-4*(1+z)**6.2)

def reionised(z,Q):     #This sets the condition for the completion of reionisation.
    return Q[0]-1.00

def Q_fit(Z,a,b,Zre):
    return (Z/Zre)**-a*np.exp(-(Z-Zre)/b)

#-------------------------------------------------------------------------------------
def Eq(z,Q):
    eq = (99.0325*Q*(1+z)**(-1.1)*n_H(z)-f_esc*28.925e4*SFRD(z)/n_H(0))/H(z)
    return eq

Z=np.linspace(Z_start,1,10001)     #This is actually (1+z)

#Solve_ivp can only integrate forward in "time", therefore make a variable change as  a=1/(1+z)=1/Z
#and since all functions require z (not a), z=1/a-1
reionised.terminal = True #This stops the integrator when the reionisation is complete.
Sol=scint.solve_ivp(lambda a, Q: -Eq(1/a-1,Q)/a, [1/Z_start, 1], [0],t_eval=1/Z, events=reionised) 

Q_HII=Sol.y[0]
ζ=1/Sol.t

print('Reionisation completed when z=',1/np.array(Sol.t_events)[0]-1)

popt,pcov=op.curve_fit(Q_fit,ζ,Q_HII)     #This is the main step. popt contain the parameters a,b,z_re
Q=Q_fit(ζ,*popt)                    
print('Parameters ',popt)
print('Error ',np.sqrt(np.diag(pcov)))
'''
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig,ax= plt.subplots()
ax.plot(ζ,Q_HII,'b')
#ax.plot(ζ,Q,'r')
ax.invert_xaxis()
ax.set_xlabel(r'$1+z$',fontsize=18)
ax.set_ylabel(r'$Q$',fontsize=18)
ax.grid(True)
ax.minorticks_on()
#ax.grid(which='minor', linewidth='0.5', color='black',alpha=0.2)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.ylim([0,1])
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.show()

secax1 = ax.secondary_xaxis('top', functions=(z2t,t2z))
secax1.set_xlabel('Time [Gyr]',fontsize=16,labelpad=12)
secax1.tick_params(which='major', labelsize=18)
'''
np.savetxt("Q_HII.txt", Q_HII)
np.savetxt("z.txt", ζ)

