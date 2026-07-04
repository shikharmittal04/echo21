.. _detexp:

In-depth usage
--------------

``ECHO21`` can be used to generate the thermal and ionization history of the intergalactic medium and hence, the cosmological global 21-cm signal. Addionally, one can use this code to study a simple analytical model of reionization and compute the CMB optical depth. Finally, you can compute the UV luminosity functions of galaxies at high redshifts. 

.. _single:

Single realization of IGM histories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are just two steps to use ``ECHO21``:

-  Give your choice of parameters
-  Run the solver

Thus, you first set up your cosmological and astrophysical parameters. These have to be supplied as a dictionary. Then specify your star formation rate density (SFRD) model, again as a dictionary. Once this is done, we can move on to creating a :class:`pipeline` object. Finally, the function :py:func:`run_simulation()` in the :class:`pipeline` object runs the code and produces the outputs. The following script (say ``run_echo21.py``) helps you get started.

.. code:: python
   
   import echo21

   #Step-1 Set you parameter choices
   cosmo = {'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'sig8':0.811,'ns':0.965,'Tcmbo':2.725,'Yp':0.245}
   astro = {'fLy':1,'sLy':2.64,'fX':1,'wX':1.5,'fesc':0.01}

   #and choose your SFRD model type by defining a dictionary
   sfrd = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':1e4}

   #Step-2 Create an object and run
   pipe = echo21.pipeline(cosmo=cosmo,astro=astro,sfrd=sfrd,path='/path/where/you/want/your/outputs/')
   pipe.run_simulation()

   #That's it.

Running the above script will generate an output folder in the path you gave in the ``path`` argument. Suppose you ran the script at 3:00:00 PM on 26th February 2025, then the output folder will have the name ``output_20250226-150000``. To understand output structure, see :ref:`output_format` below.

Please see the paper for an understanding of parameters ``fLy``, ``sLy``, etc. In brief, ``fLy``, ``sLy``, ``fX``, ``wX``, and ``fesc``, are the Ly :math:`\alpha` emissivity normalisation, power-law index of Ly :math:`\alpha` SED, X-ray emissivity normalisation, power-law index of X-ray SED, and escape fraction of ionizing photons, respectively. ``Tmin_vir`` is the minimum virial temperature of the star-forming haloes.

.. _multi:

Charting a parameter space of IGM: running `ECHO21` in parallel mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now suppose you want to generate many gas temperatures, 21-cm signals, neutral hydrogen fraction, optical depths, etc. for different astrophysical (or cosmological) parameter. For this you simply have to provide your choice of parameters as a dictionary of lists (or arrays).

**Grid or no grid?** there is an additional consideration. You can choose to generate the results for all possible combinations of the parameters (i.e., a grid) or only for the combinations of parameters at the same index in the lists/arrays. For example, if you have two parameters, each with three values, then in the grid case you will have 9 models corresponding to all possible combinations of the parameters, but in the no-grid case you will have only 3 models corresponding to the combinations of parameters at the same index. The choice is yours. By default, ``ECHO21`` generates signals assuming no grid. To generate results for all possible combinations of parameters (i.e., grid) set ``grid_on=True`` when defining your :class:`pipeline` object.


The following example shows you how to run `ECHO21` for multi-valued parameters in parallel mode. Replace ``astro = {'fLy':1,'sLy':2.64,'fX':1,'wX':1.5,'fesc':0.01}`` in ``run_echo21.py`` by

.. code:: python

   astro = {'fLy':np.logspace(-2,2,5),'sLy':[-1,0,1],'fX':np.logspace(-2,2,5),'wX':[0,1,2],'fesc':[0.01,0.1,1]}

(Note that the parameters supplied can be lists or arrays.)

For minimum virial temperature you should modify the SFRD dictionary. So now instead of ``sfrd = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':1e4}`` you should have something like this

.. code:: python

   sfrd = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':np.logspace(2,6,5)}

Thus, **the complete script to generate a large space of** :math:`T_{21}`, :math:`x_{\mathrm{HI}}`, :math:`\tau_{\mathrm{e}}`, :math:`\mathrm{d}\Phi/\mathrm{d}M_{\mathrm{UV}}` (and more) **with varying astrophysical parameters** now looks like

.. code:: python
   
   import numpy as np
   import echo21

   #Step-1 Set you parameter choices
   cosmo = {'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'sig8':0.811,'ns':0.965,'Tcmbo':2.725,'Yp':0.245}
   astro = {'fLy':np.logspace(-2,2,5),'sLy':[-1,0,1],'fX':np.logspace(-2,2,5),'wX':[0,1,2],'fesc':[0.01,0.1,1]}

   #and choose your SFRD model type by defining a dictionary
   sfrd = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':np.logspace(2,6,5)}

   #Step-2 Create an object and run
   pipe = echo21.pipeline(cosmo=cosmo,astro=astro,sfrd=sfrd, grid_on=True, path='/path/where/you/want/your/outputs/')
   pipe.run_simulation()

Now a total of :math:`5\times3\times5\times3\times3\times5=3375` models will be generated corresponding to 5 values of :math:`f_{\mathrm{Ly}}`, 3 values of :math:`s_{\mathrm{Ly}}`, 5 values of :math:`f_{\mathrm{X}}`, 3 values of :math:`w_{\mathrm{X}}`, 3 values of :math:`f_{\mathrm{esc}}`, and 5 values of :math:`T_{\mathrm{vir}}`. (In the paper, I have used :math:`s` for ``sLy`` and :math:`w` for ``wX``.)

Similarly, you can change the ``cosmo`` parameter in the above script to **generate a large space of** :math:`T_{21}`, :math:`x_{\mathrm{HI}}`, :math:`\tau_{\mathrm{e}}`, etc **with varying cosmological parameters**. Further, ``ECHO21`` is not limited to varying either astrophysical or cosmological parameters; both can be simultaneously varied.

In the above example we set `grid_on` to True. If you choose ``grid_on=False``, *then the parameters you wish to vary should have the same number of values*. In this case, the code will generate models for combinations of parameters at the same index in the lists/arrays. As an example

.. code:: python
   
   import numpy as np
   import echo21

   #Step-1 Set you parameter choices
   cosmo = {'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'sig8':0.811,'ns':0.965,'Tcmbo':2.725,'Yp':0.245}
   astro = {'fLy':np.logspace(-2,2,3),'sLy':[-1,0,1],'fX':np.logspace(-2,2,3),'wX':[0,1,2],'fesc':[0.01,0.1,1]}

   #and choose your SFRD model type by defining a dictionary
   sfrd = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':np.logspace(2,6,3)}

   #Step-2 Create an object and run
   pipe = echo21.pipeline(cosmo=cosmo,astro=astro,sfrd=sfrd, grid_on=False, path='/path/where/you/want/your/outputs/')
   pipe.run_simulation()

The above script will generate 3 models only.



You can run the above script on your local PC as usual *but with more than one CPU*, as ``ECHO21`` uses a master-worker CPU distribution. Thus, if you provide N CPUs, one CPU will act as the master CPU and remaining N-1 will act as worker CPUs. In general, generating a large number of models on a single CPU can be time consuming. To save time, you should utilize the **parallel** feature of ``ECHO21`` and run the script ``run_echo21.py`` as (say on four CPUs)

.. code:: bash
   
   mpirun -np 4 python run_echo21.py

Using a similar strategy you can now generate thousands of models in a few minutes with an appropriate choice of HPC resources.


For multi-valued parameter run, ``params_df`` is a useful object. It is a dataframe, where each row in ``params_df`` corresponds to a unique model. For the ":math:`r^{\mathrm{th}}`" model, i.e., for the set of parameters at :math:`r^{\mathrm{th}}` row, the corresponding 21-cm signal is at :math:`r^{\mathrm{th}}` row, i.e., ``T21[r,:]`` (similarly for other quantities). This gives you a convenient way to access the outputs for a particular model and hence useful for training.


Choosing a different HMF
^^^^^^^^^^^^^^^^^^^^^^^^

Until now we have been using the Press & Schechter (1974) HMF. In ``ECHO21`` you can choose a different HMF also. Suppose you want to generate a signal for Sheth & Tormen (1999) HMF. Then set ``'sheth99'`` for the ``hmf`` keyword in the SFRD dictionary. For some HMFs you will have to change your definition of halo mass which is done by the keyword ``mdef``. For example both Press & Schechter (1974) and Sheth & Tormen (1999) are based on the friends-of-friends definition (which is why we set ``'fof'`` for  ``mdef``), but Tinker et al. (2008) is based on an integer multiple of mean or critical matter density of the Universe. So you can give, say, ``'200m'`` for ``mdef``. For a complete list of available HMFs see the `COLOSSUS <https://bdiemer.bitbucket.io/colossus/lss_mass_function.html#mass-function-models>`_ page.

Below is an example syntax for SFRD dictionary using Tinker et al. (2008) HMF.

.. code:: python

   sfrd = {'type':'phy','hmf':'tinker08','mdef':'200m','Tmin_vir':1e4}


Choosing a different SFRD model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Until now we have been working with physically-motivated star formation rate density (SFRD) models, which is why we had ``'phy'`` for ``type`` in the SFRD dictionary. ``ECHO21`` offers two additional models of SFRD -- *semi-empirical* model and an *empirically-motivated* SFRD model. Let us first look at the semi-empirical model. The dictionary looks mostly the same as for the physically-motivated case, except now we use ``'semi-emp'`` for ``type``. Further, for this case now you also have an additional free parameter, ``tstar`` (default value 0.5). The dictionary now looks like

.. code:: python
   
   sfrd = {'type':'semi-emp','hmf':'press74','mdef':'fof','Tmin_vir':1e4, 'tstar':0.5}

Let us now implement an empirically-motivated SFRD model. For this you need to set your SFRD type as ``'emp'`` and choose the :math:`a` parameter. 

.. code:: python
   
   sfrd = {'type':'emp','a':0.257}

See section 2.2 from our paper for more details on SFRD.


.. _output_format:

Output structure
^^^^^^^^^^^^^^^^

Let us first understand the three output files ``ECHO21`` produces. 

   1. ``pipeline.pkl``: a pickle file which holds the pipeline object (``pipe`` in the above examples). This object has all your inputs and outputs as its attributes. Output inclues gas temperature, spin temperature, bulk IGM electron fraction, volume-filling factor, globally-averaged neutral hydrogen fraction, 21-cm signal, and UV luminosity function. For IDM simulations, DM temperature and baryon-DM relative velocity are also included. The gas and spin temperatures are in units of kelvin and 21-cm signal is in units of milli kelvin (mK).

   2. ``summary_<timestamp>.txt``: a text file containing the summary of your simulation. This file contains the timestamp, execution time, and cosmological & astrophysical parameters you provided.

   3. ``echo21_output.h5``: an HDF5 file containing the simulation results. Same outputs part of the pipeline object are stored in this file also. Additionally, the HDF5 file contains the parameters (only multi-valued ones) as a pandas dataframe, redshifts (:math:`1+z`, **not** :math:`z`) from dark ages to today and cosmic dawn to today, and absolute AB magnitude. If :math:`N_{\mathrm{p}}, N_{z}, N_{\mathrm{mag}}`, and :math:`N_{\mathrm{mod}}` are the number of parameters that are multi-valued, number of redshift points, number of magnitude points, and number of models, respectively then ``params_df`` is a table of shape :math:`N_{\mathrm{mod}} \times N_{\mathrm{p}}`, ``one_plus_z`` is a 1D numpy array of length :math:`N_{z}`, ``xe``, ``Tk``, ``Ts``, ``T21`` & ``xHI`` are 2D arrays of shape :math:`N_{\mathrm{mod}} \times N_{z}`, and ``tau`` is a 1D array of length :math:`N_{\mathrm{mod}}`.

      Volume-filling factor and UV luminosity function are 2D and 3D arrays but correspond to redshifts from cosmic dawn to today. So they are of shape :math:`N_{\mathrm{mod}} \times N_{z_{\mathrm{c\,d}}}` and :math:`N_{\mathrm{mod}} \times N_{\mathrm{mag}} \times N_{z_{\mathrm{c\,d}}}`, respectively, where :math:`N_{z_{\mathrm{c\,d}}}` is the number of redshift points from cosmic dawn to today.

      For IDM simulations, DM temperature and baryon-DM relative velocity are also stored in the HDF5 file as 2D arrays of shape :math:`N_{\mathrm{mod}} \times N_{z}`.


**Read the output:**

There are *two* ways to work with outputs. You can access the quantities from the pipeline object after you have run the command ``pipe.run_simulation()``. For example, ``pipe.Tk`` gives the gas temperature. Alternatively, you can use the :py:func:`load_results` function to read the results from an hdf5 file.
Second method has an extra advantage. It allows you to choose a custom redshift range and evaluate the quantities at those redshifts. The example below shows you how to evaluate the quantities between :math:`1+z=30` and :math:`1+z=10` with 100 evenly spaced values.

Use the following lines of code to load the output:

.. code:: python

   import echo21

   filename = "output_20250226-150000"

   #Method-1: Load the pipeline object
   pipe = echo21.load_pipeline(filename)
   one_plus_z = pipe.one_plus_z
   T21 = pipe.T21
   #and more such quantities are available as attributes of the pipeline object

   #Method-2: Load the results from the hdf5 file
   #additionally, you can specify your choice of redshifts at which you want to evaluate the quantities.
   myZs = np.linspace(30,10,100)
   results_dic = echo21.load_results(filename, Z_eval=myZs)
   #The results_dic is a dictionary with keys 'one_plus_z', 'params_df', 'Tk', 'Ts', 'xe', 'xHI', 'T21',etc.
   #Z_eval is optional. If not provided, the quantities will be evaluated at the default redshifts.

Save it as (say) ``load_echo21.py`` and run it as

.. code:: bash

   python load_echo21.py

