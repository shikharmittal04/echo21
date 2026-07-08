"""
``funcs``
=========
This module contains class :class:`funcs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from . import function_container as fc

class funcs():
    '''
    ``funcs`` is a wrapper around the function_container and thus, provides a common point of access to all the functions in the container.

    Arguments
    ---------
    config : :class:`config`
        A configuration object for a particular model which gives access to all the model parameters, such as :math:`H_0`.
    
    '''
    def __init__(self, config):
        #level 1
        self.basic = fc.basic(config)

        #level 2
        self.halo = fc.halo(config, self.basic)
        self.recomb = fc.recomb(config, self.basic)
        self.idm = fc.idm(config, self.basic)

        #level 3
        self.eor = fc.eor(config, self.basic, self.halo)
        self.lya = fc.lyman_alpha(config, self.basic, self.halo)
        self.hyfi = fc.hyfi(config, self.basic, self.lya)

        #level 4
        self.heating = fc.heating(config, self.basic, self.halo, self.lya)
        self.uvlf = fc.uvlf(config, self.basic, self.halo)

        #level 5
        self.ivp = fc.ivp(config, self.basic, self.recomb, self.halo, self.idm, self.heating)
