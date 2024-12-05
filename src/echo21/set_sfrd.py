'''
This module has different sets for SFRD models. Use this to select your SFRD model and specify the corresponding parameters as a dictionary.
'''
class phy_sfrd():
    def __init__(self,hmf='press74',sfrd_para={'Tmin_vir':1e4}):
        '''
        Use this class to set your parameters for a physically-motivated SFRD.

        hmf : str, optional
            HMF model to use. Default value ``press74``. Other available HMFs are
            - sheth99 (for Sheth & Tormen 1999)

        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.          
        '''
        self.name = 'phy'
        self.hmf = hmf
        self.sfrd_para = sfrd_para
        return None

class emp_sfrd():
    def __init__(self,sfrd_para = {'a':0.01,'b':2.6,'c':3.2,'d':6.2}):
        '''
        Use this class to set your parameters for an empirically-motivated SFRD. See eq.(15) from `Madau & Dickinson (2014) <https://www.annualreviews.org/content/journals/10.1146/annurev-astro-081811-125615>`__.

        Note that the value of a should be given in units of Msun/yr/Mpc^3.
        '''
        self.name = 'emp'
        self.sfrd_para = sfrd_para
        return None

