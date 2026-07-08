"""
``funcs``
=========
This sub-package holds the physics of ECHO21 split by concern:

- :mod:`basic`       -- background LCDM cosmology (Hubble, Tcmb, nH, ...)
- :mod:`recomb`      -- recombination physics (Peebles C, Saha, alpha, beta)
- :mod:`halo`        -- halo mass function, collapse fraction, SFRD
- :mod:`heating`     -- IGM heating terms
- :mod:`hyfi`        -- 21-cm / hyperfine physics (spin temperature, T21)
- :mod:`lyman_alpha` -- Lyman-alpha specific intensity
- :mod:`eor`         -- reionization (QHii, optical depth)
- :mod:`idm`         -- interacting dark matter terms
- :mod:`ivp`         -- IGM ODE system and integrator
- :mod:`uvlf`        -- UV luminosity function
"""

from .basic import basic
from .recomb import recomb
from .halo import halo
from .heating import heating
from .hyfi import hyfi
from .lyman_alpha import lyman_alpha
from .eor import eor
from .idm import idm
from .ivp import ivp
from .uvlf import uvlf

__all__ = [
    'basic',
    'recomb', 'halo', 'idm',
    'eor', 'lyman_alpha', 'hyfi', 'uvlf',
    'heating',
    'ivp'
]
