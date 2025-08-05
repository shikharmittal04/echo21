.. _detexp:

In-depth usage
--------------

``ECHO21`` can be used to generate the thermal and ionization history of the intergalactic medium and hence, the cosmological global 21-cm signal. Addionally, one can use this code to study a simple analytical model of reionization and compute the CMB optical depth.

.. _single:

Single realization of 21-cm signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are just two steps to use ``ECHO21``:

-  Give your choice of parameters
-  Run the solver

Thus, you first set up your cosmological and astrophysical parameters. These have to be supplied as a dictionary. Then specify your star formation rate density (SFRD) model, again as a dictionary. Once this is done, we can move on to creating a :class:`pipeline` object. Finally, the function :py:func:`glob_sig()` in the :class:`pipeline` object runs the code and produces the outputs. The following script (say ``my_echo_script.py``) helps you get started.

.. code:: python
   
   from echo21 import echopipeline

   #Step-1 Set you parameter choices
   cosmo = {'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'sig8':0.811,'ns':0.965,'Tcmbo':2.725,'Yp':0.245}
   astro = {'fLy':1,'sLy':2.64,'fX':1,'wX':1.5,'fesc':0.0106}

   #and choose your SFRD model type by defining a dictionary
   sfrd = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':1e4}

   #Step-2 Create an object and run
   myobj = echopipeline.pipeline(cosmo=cosmo,astro=astro,sfrd_dic=sfrd,path='/path/where/you/want/your/outputs/')
   myobj.glob_sig()

   #That's it.

Running the above script will generate an output folder in the path you gave in the ``path`` argument. Suppose you ran the script at 3:00:00 PM on 26th February 2025, then the output folder will have the name ``output_20250226-150000``. To understand output structure, see :ref:`output_format` below.

Please see the paper for an understanding of parameters ``fLy``, ``sLy``, etc. In brief, ``fLy``, ``sLy``, ``fX``, ``wX``, and ``fesc``, are the Ly :math:`\alpha` emissivity normalisation, power-law index of Ly :math:`\alpha` SED, X-ray emissivity normalisation, power-law index of X-ray SED, and escape fraction of ionizing photons, respectively. ``Tmin_vir`` is the minimum virial temperature of the star forming haloes.

.. _multi:

Exploring a large space of 21-cm signals: running ECHO21 in parallel mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now suppose you want to generate many 21-cm signals for different astrophysical (or cosmological) parameter. For this you simply have to provide your choice of parameters as a dictionary of lists or arrays. The following example shows you how. You simply have to replace ``astro = {'fLy':1,'sLy':2.64,'fX':1,'wX':1.5,'fesc':0.0106}`` in ``my_echo_script.py`` by

.. code:: python

   astro = {'fLy':np.logspace(-2,2,5),'sLy':[-1,0,1],'fX':np.logspace(-2,2,5),'wX':[0,1,2],'fesc':[0.01,0.1,1]}

For minimum virial temperature you should modify the SFRD dictionary. So now instead of ``sfrd = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':1e4}`` you should have something like this

.. code:: python

   sfrd = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':np.logspace(2,6,5)}

Thus, **the complete script to generate a large space of 21-cm signals with varying astrophysical parameters** now looks like

.. code:: python
   
   import numpy as np
   from echo21 import echopipeline

   #Step-1 Set you parameter choices
   cosmo = {'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'sig8':0.811,'ns':0.965,'Tcmbo':2.725,'Yp':0.245}
   astro = {'fLy':np.logspace(-2,2,5),'sLy':[-1,0,1],'fX':np.logspace(-2,2,5),'wX':[0,1,2],'fesc':[0.01,0.1,1]}

   #and choose your SFRD model type by defining a dictionary
   sfrd = {'type':'phy','hmf':'press74','mdef':'fof','Tmin_vir':np.logspace(2,6,5)}

   #Step-2 Create an object and run
   myobj = echopipeline.pipeline(cosmo=cosmo,astro=astro,sfrd_dic=sfrd,path='/path/where/you/want/your/outputs/')
   myobj.glob_sig()

Now a total of :math:`5\times3\times5\times3\times3\times5=3375` models will be generated corresponding to 5 values of :math:`f_{\mathrm{Ly}}`, 3 values of :math:`s_{\mathrm{Ly}}`, 5 values of :math:`f_{\mathrm{X}}`, 3 values of :math:`w_{\mathrm{X}}`, 3 values of :math:`f_{\mathrm{esc}}`, and 5 values of :math:`T_{\mathrm{vir}}`. (In the paper, I have used :math:`s` for ``sLy`` and :math:`w` for ``wX``.)

Similarly, you can change the ``cosmo`` parameter in the above script to **generate a large space of 21-cm signals with varying cosmological parameters**. Further, ``ECHO21`` is not limited to varying either astrophysical or cosmological parameters; both can be simultaneously varied.



You can run the above script on your local PC as usual but with more than one CPU as ``ECHO21`` uses a master-worker CPU distribution. Thus, if you provide N CPUs, 1 CPU will act as the master CPU and remaining N-1 will act as worker CPUs. In general, generating a large number of models on a single CPU can be time consuming. To save time, you should utilize the **parallel** feature of ``ECHO21`` and run the script ``my_echo_script.py`` as (say on four CPUs)

.. code:: bash
   
   mpirun -np 4 python my_echo_script.py

Using a similar strategy you can now generate thousands of models in a few minutes with an appropriate choice of HPC resources.


Choosing a different HMF
^^^^^^^^^^^^^^^^^^^^^^^^

Until now we have been using the Press & Schechter (1974) HMF. In ``ECHO21`` you can choose a different HMF also. Suppose you want to generate a signal for Sheth & Tormen (1999) HMF. Then set ``'sheth99'`` for the ``hmf`` keyword in the SFRD dictionary. For some HMFs you will have to change your definition of halo mass which is done by the keyword ``mdef``. For example both Press & Schechter (1974) and Sheth & Tormen (1999) are based on the friends-of-friends definition (which is why we set ``'fof'`` for  ``mdef``), but Tinker et al. (2008) is based on an integer multiple of mean matter density of the Universe. So you can give, say, ``'200m'`` for ``mdef``. For a complete list of available HMFs see the `COLOSSUS <https://bdiemer.bitbucket.io/colossus/lss_mass_function.html#mass-function-models>`_ page.

Below is an example syntax for SFRD dictionary using Tinker et al. (2008) HMF.

.. code:: python

   sfrd = {'type':'phy','hmf':'tinker08','mdef':'200m','Tmin_vir':1e4}


Choosing a different SFRD model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Until now we have been working with physically-motivated star formation rate density (SFRD) models, which is why we had ``'phy'`` for ``type`` in the SFRD dictionary. ``ECHO21`` offers two additional models of SFRD -- semi-empirical model and an empirically-motivated SFRD model. Let us first look at the semi-empirical model. The dictionary looks mostly the same as for the physically-motivated case, except now we use ``'semi-emp'`` for ``type``. Further, for this case now you also have an additional free parameter, ``t_star`` (default value 0.5). The dictionary now looks like

.. code:: python
   
   sfrd = {'type':'semi-emp','hmf':'press74','mdef':'fof','Tmin_vir':1e4, 't_star':0.5}

Let us now implement an empirically-motivated SFRD model. For this you simply need to set your type as ``'emp'`` and choose the :math:`a` parameter. 

.. code:: python
   
   sfrd = {'type':'emp','a':0.257}


See our section 2.2 from our paper for more details on SFRD.


Choosing the redshifts at which you want to evaluate global signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before anything I want to clarify that I always work with :math:`1+z` and NOT :math:`z`. So wherever, I write redshift I talk about :math:`1+z`. To avoid confusion I have used the capital letter zed ('Z') to represent :math:`1+z`.

Moving on to the main content of this section, when you do not specify the redshift range the code will evaluate the quantities at default redshifts. This default has 2300 values defined by the array ``Z_default`` given below.

.. code:: python
   
   import numpy as np
   Z_cd = np.concatenate((1/np.linspace(1/60,1/5.05,200),1/np.linspace(1/5,1,100)))
   Z_default = np.concatenate((np.linspace(1501,60.1,2000),Z_cd))

When you run the code for a single set of parameters or vary cosmological parameters (irrespective of astrophysical ones) then the code will output the signal at redshits defined by ``Z_default`` by default. When you vary only astrophysical parameters then the code will output the signals at cosmic dawn redshifts defined by ``Z_cd``.

**How to give redshift values of your choice?** Simple, just give your choice through the argument ``Z_eval`` when defining the ``pipeline`` object. For example, if you want to generate signal between :math:`1+z=30` and :math:`1+z=10` with 100 evenly spaced values then you should do the following

.. code:: python

   myZs = np.linspace(30,10,100)
   myobj = echopipeline.pipeline(cosmo=cosmo,astro=astro,sfrd_dic=sfrd,path='/path/where/you/want/your/outputs/',Z_eval=myZs)

Note: you don't have to worry about giving redshifts in decreasing order. Whichever order you give, ``ECHO21`` will always generate outputs for decreasing redshifts. When you are varying the astrophysical parameters only, the highest value of :math:`1+z` should not be above 60. 

.. _output_format:

Output structure
^^^^^^^^^^^^^^^^

When you run ``ECHO21`` for a single parameter the output folder will contain 9 files. These are redshifts (:math:`1+z`, **not** :math:`z`), CMB temperature (Tcmb.npy), gas temperature (Tk.npy), spin temperature (Ts.npy), bulk IGM electron fraction (xe.npy), volume-filling factor (Q.npy), 21-cm signal (T21.npy), a text file ``glob_sig_20250226-150000.txt``, and the class object ``echopipeline.pipeline`` as ``pipe.pkl``. All ``.npy`` files are 1D arrays. They are evaluated at redshifts in the ``.npy`` file ``one_plus_z.npy``. The CMB, gas, and spin temperatures are in units of kelvin and 21-cm signal is in units of milli kelvin (mK).

The text file contains all the basic information regarding your simulation such as the timestamp, execution time, cosmological & astrophysical parameters you provided. This file also contains the redshift when the Universe was 50% ionized and 100% ionized, and the total CMB optical depth. Also, the file mentions the strongest 21-cm signal and the corresponding redshift.

In case of multiple values of parameter(s), only global signal, redshift, the text file, and the object file are generated. When you vary astrophysical parameter(s), then T21.py will be a 7D array. Consider the example in section :ref:`multi`. In this case T21 will be of shape :math:`5\times3\times5\times3\times3\times5\times300` (assuming you did not give your own redshift values. If you did, then in the last dimension, instead of 300 it will be your number of values.). The first dimension will correspond to ``fLy``, second to ``sLy``, third to ``fX``, fourth to ``wX``, fifth to ``fesc``, and sixth to ``Tmin_vir``. The sixth dimension will correspond to ``Tmin_vir`` if you choose the physically-motivated SFRD model, otherwise the sixth dimension will correspond to ``a`` -- relevant to empirically-motivated SFRD. Seventh index corresponds to global signal values. Continuing with the example in section :ref:`multi`, suppose you want to access the global signal corresponding to :math:`f_{\mathrm{Ly}} = 10^{-1}`, :math:`s = 1`, :math:`f_{\mathrm{X}}=10^2`, :math:`w=0`, :math:`f_{\mathrm{esc}}=1`, and min (:math:`T_{\mathrm{vir}})=10^5\,` K, then you should run the following code in your ``output_*`` folder.

.. code:: python

   import numpy as np
   T21 = np.load('T21.npy')
   T21[1,2,4,0,2,3,:] # is the required signal.


Similarly, when you vary only the cosmological parameters, the global signal will be an 8D array. The first to 7th dimension will correspond to parameters ``Ho``, ``Om_m``, ``Om_b``, ``sig8``, ``ns``, ``Tcmbo``, and ``Yp``, respectively.

Finally, if you vary cosmological as well as astrophysical parameters then the output will be a 14D array. As before, first 7 dimensions will correspond to cosmological parameters, next 6 dimensions will correspond to astrophysical parameters and finally the last dimension corresponds to the 21-cm signal.

Often you may not want to vary all parameters. In this case corresponding to the parameter you want fixed the dimension will be of size 1 in ``T21`` array. To get rid of these redundant dimensions, simply use the ``numpy.squeeze`` function.