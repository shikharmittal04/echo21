Basics
======

Overview
--------

:Name: Exploring Cosmos with Hydrogen Observation
:Author: `Shikhar Mittal <https://sites.google.com/view/shikharmittal/home>`_
:Paper: `Mittal (2025) <https://arxiv.org/abs/2503.11762>`_

Why do you need this code?
--------------------------

Use this code to generate the global 21-cm signal(s) for a given set of astrophysical and cosmological parameters.

Read more about it in the paper `Mittal (2025) <https://arxiv.org/abs/2503.11762>`_.

Installation and requirements
-----------------------------

This package can be installed as

.. code:: bash

   pip install echo21

We recommend working on a Python version > 3.8. Packages required are 

- `numpy <https://pypi.org/project/numpy/>`_ (recommended version 2.1.3)
- `scipy <https://pypi.org/project/scipy/>`_ (recommended version 1.14.1)
- `mpi4py <https://pypi.org/project/mpi4py/>`_ (recommended version 4.0.1)
- `colossus <https://pypi.org/project/colossus/>`_ (recommended version 1.3.6)

Quick start
-----------

The following code more or less captures the main functionalities of this package.

.. code:: python

   from echo21 import echopipeline

   pipe = echopipeline.pipeline()
   pipe.glob_sig()

Save the above code as (say) ``my_echo_script.py`` and run it as

.. code:: bash

    python my_echo_script.py

Running the above will generate an output folder with the name `output_<YYYYMMDD-hhmmss>` which contains several files. To learn how to set the astrophysical or cosmological parameters, halo mass function, star formation model, redshifts at which to evaluate the global signal, and structure of the output files see the documentation. To learn about the physics of this package see our `paper <https://arxiv.org/abs/2503.11762>`_. 

Once you have an understanding of the structure of output files, you can write your own scripts to create figures. To help you get started, see the jupyter notebook `make_figures.ipynb` in the example folder.

Documentation
-------------
For more details on the working of the package and understanding the output files refer to the 
`documentation <https://echo21.readthedocs.io/en/latest/index.html>`_.

License and citation
--------------------
The software is free to use on the MIT open-source license. If you use the software then please consider citing `Mittal (2025) <https://arxiv.org/abs/2503.11762>`_.

Contact
-------

In case of any confusion or suggestions for improvement please do not hesitate to contact me.