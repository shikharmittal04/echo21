#My own plotting style.

import matplotlib.pyplot as plt
import numpy as np

#The following two functions will add a secondary x-axis showing the frequency
np.seterr(divide='ignore')
def Z2nu(Z):
    return 1420/Z

def nu2Z(nu):
    return 1420/nu

def plotter(x,y,xlog=True,ylog=False,quant_name='',add_edges=False,xlow=6,xhigh=200):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig,ax=plt.subplots(figsize=(8.3,7.5),dpi=300)
    fig.subplots_adjust(left=0.12, bottom=0.07, right=0.88, top=0.97)
    clr=['b','r','limegreen']
    linsty=['-','--',':']
    if type(y)==dict:
        leng = len(y)
        keys = list(y.keys())
        for i in range(leng):
            if keys[i]=='Tk':
                lbl = '$T_{\mathrm{k}}$'
            elif keys[i]=='Ts':
                lbl = '$T_{\mathrm{s}}$'
            elif keys[i]=='Tcmb':
                lbl = '$T_{\mathrm{cmb}}$'
            else:
                print('Warning: unknown quantity given in the dictionary!')
            ax.plot(x,y[keys[i]],color=clr[i],ls=linsty[i],label=lbl)
        if leng>1:
            ax.set_ylabel(r'$T\,$(K)',fontsize=20)
            ax.legend(fontsize=18,frameon=False)
        else:
            if quant_name=='Tk':
                ax.set_ylabel(r'$T_{\mathrm{k}}\,$(K)',fontsize=20)
            elif quant_name=='Tcmb':
                ax.set_ylabel(r'$T_{\mathrm{cmb}}\,$(K)',fontsize=20)
            elif quant_name=='Ts':
                ax.set_ylabel(r'$T_{\mathrm{s}}\,$(K)',fontsize=20)
            
             
    elif type(y)==np.ndarray:
        ax.plot(x,y,'b')
        if quant_name=='xe':
            ax.set_ylabel(r'$x_{\mathrm{e}}$',fontsize=20)
        elif quant_name=='Tk':
            ax.set_ylabel(r'$T_{\mathrm{k}}\,$(K)',fontsize=20)
        elif quant_name=='Tcmb':
            ax.set_ylabel(r'$T_{\mathrm{cmb}}\,$(K)',fontsize=20)
        elif quant_name=='Ts':
            ax.set_ylabel(r'$T_{\mathrm{s}}\,$(K)',fontsize=20)
        elif quant_name=='sfrd' or quant_name=='SFRD':
            ax.set_ylabel(r'$\dot{\rho}_{\star}\,(\mathrm{kg\,m^{-3}s^{-1}})$',fontsize=20)
        elif quant_name=='T21':
            if add_edges==True:
                nu_edges=np.load('nu_edges.npy')
                Z_edges=1420/nu_edges
                T21_edges=np.load('T21_edges.npy')
                res=np.load('residue.npy')
                ax.plot(Z_edges,1000*(T21_edges+res),'r--',lw=1.5)
                ax.legend(['Theory','EDGES'],fontsize=18,frameon=False)
                secax = ax.secondary_xaxis('top', functions=(Z2nu,nu2Z))
                secax.set_xlabel(r'$\nu\,(\mathrm{MHz})$',fontsize=20, labelpad=12)
                secax.minorticks_on()
                secax.tick_params(axis='x',which='major', length=5, width=1, labelsize=20,direction='in')
                secax.tick_params(axis='x',which='minor', length=3, width=1, direction='in')
            ax.axhline(y=0,ls=':',color='k')
            ax.set_xlim([xlow,xhigh])
            ax.set_ylabel(r'$T_{21}\,$(mK)',fontsize=20)
        else:
            print("Warning: enter the quantity name. Use argument 'quant_name'")
        
    else:
        print("Error: incorrect syntax! Give y as an array or dictionary of arrays. eg. {'Tk':Tk,'Ts':Ts,'Tcmb':Tcmb}")
        sys.exit()

    if xlog==True:
        ax.set_xscale('log')
    if ylog==True:
        ax.set_yscale('log')
    
    ax.set_xlabel(r'$1+z$',fontsize=20)    
    ax.tick_params(axis='both', which='major', length=5, width=1, labelsize=20,direction='in')
    ax.tick_params(axis='both', which='minor', length=3, width=1, direction='in')
    ax.minorticks_on()
    ax.invert_xaxis()
    ax.yaxis.set_ticks_position('both')
    if add_edges==False:
        ax.xaxis.set_ticks_position('both')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.show()
    #plt.savefig(quant_name+'.pdf')
    return 
