#This has eveything related to recombination physics.

from .const import *
from .basic_cosmo import H, nH
import numpy as np

def alpha(Tk): #Case-B recombination coefficient for hydrogen
    t=Tk/10000
    return (1e-19)*Feff*A*t**b/(1+c*t**d)

def beta(Tk): #Photo-ionization rate
    return alpha(Tk)*(2*np.pi*me*kB*Tk/hP**2)**1.5*np.exp(-B2/(kB*Tk))

def Krr(Z,Ho,Om_m, Tcmbo): #Redshifting rate
    return lam_alpha**3/(8*np.pi*H(Z,Ho,Om_m, Tcmbo))

def Peebles_C(Z,xe,Tk, Ho,Om_m,Om_b,Tcmbo,Yp):
    return (1+Krr(Z,Ho,Om_m, Tcmbo)*Lam_H*nH(Z, Ho,Om_b,Yp)*(1-xe))/(1+Krr(Z, Ho,Om_m, Tcmbo)*(Lam_H+beta(Tk))*nH(Z, Ho,Om_b,Yp)*(1-xe))

def Saha_xe(Z,Tk, Ho,Om_b,Yp):
    S=1/nH(Z, Ho,Om_b,Yp)*(2*np.pi*me*kB*Tk/hP**2)**1.5*np.exp(-B1/(kB*Tk))
    return (np.sqrt(S**2+4*S)-S)/2
