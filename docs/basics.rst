Basics
======

Overview
--------

:Name: Exploring Cosmos with Hydrogen Observation
:Author: `Shikhar Mittal <https://sites.google.com/view/shikharmittal/home>`_
:Homepage: https://github.com/shikharmittal04/echo21
:Paper: `Mittal et al. (2024) <https://arxiv.org/abs/2406.17031>`_

Why do you need this code?
--------------------------

Use this code to generate the global 21-cm signal(s) for a given set of astrophysical and cosmological parameters.

Read more about it in the paper `Mittal et al (2024) <https://arxiv.org/abs/2406.17031>`_.

Installation and requirements
-----------------------------

This package can be installed as

.. code:: bash

   pip install echo21

We recommend working on a Python version > 3.8. Packages required are 

- `numpy <https://pypi.org/project/numpy/>`_
- `scipy <https://pypi.org/project/scipy/>`_
- `matplotlib <https://pypi.org/project/matplotlib/>`_
- `mpi4py <https://pypi.org/project/mpi4py/>`_
- `colossus <https://pypi.org/project/colossus/>`

Quick start
-----------

There are only two steps to use `echo21`:

-  Give your choice of parameters
-  Run the solver.

The following code captures the main functionalities of this package.

.. code:: python

   from echo21 import echo

   #Step-1 Set you parameter choices
   cosmo = {'Ho':67,'Om_m':0.315,'Om_b':0.049,'Tcmbo':2.725,'Yp':0.245}
   astro = {'falp':0.1,'fX':0.01,'fesc':0.1,'Tmin_vir':10000} 

   #Step-2 Initialise the object and run
   pipe = echo.pipeline(cosmo=cosmo,astro=astro,path='/path/where/you/want/your/outputs/',hmf_name='press74')
   pipe.glob_sig()

   #That's it.


Save the above code as (say) ``eg_script.py`` and run it as

.. code:: bash

    python eg_script.py

Running the above will generate several files. To visualise your outputs use the jupyter notebook `make_figures.ipynb`. To learn more about the output files and code working see :ref:`detexp`. To learn about the physics of this package see our `paper <https://arxiv.org/abs/2406.17031>`_.

License and citation
--------------------
The software is free to use on the MIT open-source license. If you use the software then please consider citing `Mittal et al (2024) <https://arxiv.org/abs/2406.17031>`_.

Contact
-------

In case of any confusion or suggestions for improvement please do not hesitate to contact me.