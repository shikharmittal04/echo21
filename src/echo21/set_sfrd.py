'''
This module has different sets for SFRD models. Use this to select your SFRD model and specify the corresponding parameters as a dictionary.
'''
class phy_sfrd():
    def __init__(self,hmf='press74',mdef='fof',para={'Tmin_vir':1e4}):
        '''
        Use this class to set your parameters for a physically-motivated SFRD.

        hmf : str, optional
            HMF model to use. Default value ``press74``. Other commonly used HMFs are
            - sheth99 (for Sheth & Tormen 1999)
            - tinker08 (for Tinker et al 2008)

            For the full list see `colossus <https://bdiemer.bitbucket.io/colossus/lss_mass_function.html#lss.mass_function.massFunction>`__ page.

        mdef: str, optional
            Definition for halo mass. Default is ``fof``. For most HMFs such as Press-Schechter or Sheth-Tormen friends-of-friends (``fof``) algorithm is used. For Tinker, it is an integer times mean matter density (``<int>m``). See colossus definition `page <https://bdiemer.bitbucket.io/colossus/halo_mass.html>`_
            
        Tmin_vir : float, optional
            Minimum virial temperature (in units of kelvin) for star formation. Default value ``1e4``.          
        '''
        self.name = 'phy'
        self.hmf = hmf
        self.mdef = mdef
        self.para = para
        return None

class emp_sfrd():
    def __init__(self,para = {'a':0.01,'b':2.6,'c':3.2,'d':6.2}):
        '''
        Use this class to set your parameters for an empirically-motivated SFRD. See eq.(15) from `Madau & Dickinson (2014) <https://www.annualreviews.org/content/journals/10.1146/annurev-astro-081811-125615>`__.

        Note that the value of a should be given in units of Msun/yr/Mpc^3.
        '''
        self.name = 'emp'
        self.para = para
        return None

