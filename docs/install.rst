.. _install:

Installation and requirements
=============================

``ECHO21`` package can be installed as

.. code:: bash

   pip install echo21

I have tested the code on Python==3.10.20. Other special packages required: 

- `numpy <https://pypi.org/project/numpy/>`_ (tested for 1.26.4)
- `scipy <https://pypi.org/project/scipy/>`_ (tested for 1.15.3)
- `mpi4py <https://pypi.org/project/mpi4py/>`_ (tested for 4.1.2)
- `tqdm <https://pypi.org/project/tqdm/>`_ (tested for 4.68.1)
- `colossus <https://pypi.org/project/colossus/>`_ (tested for 1.4.0)
- `pandas <https://pypi.org/project/pandas/>`_ (tested for 2.3.3)
- `tables <https://pypi.org/project/tables/>`_ (tested for 3.10.1)

The following two additional packages are required for simulation of the interacting dark matter (IDM) model. Because ``CLASS`` requires an older version of ``Cython``, I recommend creating a dedicated environment (such as conda) to avoid conflicts with your existing packages.

- `Cython <https://pypi.org/project/Cython/>`_ (tested for 0.29.37; required for IDM)
- `classy <https://github.com/kboddy/class_public.git>`_ (tested for 2.9.4; required for IDM)



Follow the `Anaconda website <https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install#wget>`_ for the steps to install Miniconda. Then follow these steps:

.. code:: bash

   conda create -n echoenv python
   conda install numpy scipy matplotlib cython=0.29 gcc_linux-64 gxx_linux-64 make

   git clone -b dmeff https://github.com/kboddy/class_public.git
   cd class_public
   make clean
   make

Now the ``CLASS`` code should hopefully be installed in your ``echoenv`` environment.