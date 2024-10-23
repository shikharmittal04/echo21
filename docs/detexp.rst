.. _detexp:

Detailed explanation
--------------------

This package can be used to generate the number density maps for the extragalactic point sources, angular power spectrum of number overdensity, brightness temperature map corresponding to extragalactic point (radio) sources and the antenna temperature. Hence, one can reproduce results similar to the ones seen in figures 1, 2, 3, 4, and 5 from our paper. Given the beam directivity one can also add to the sky-averaged foregrounds due to extragalactic point sources plot, the antenna temperature (as in the upper panel of figure 8). [#f1]_


Although, in our paper we have simulated for the both kinds of point sources, resolved and unresolved, as such there is no difference between the mechanics of them except on their detectability. Unresolved point sources are discrete sources on the sky which are usually smaller than the telescope beam width. Typically, the bright resolved source will be trivial to peel off the foregrounds. It is usually the unresolved sources that are problematic. Thus, if users are only interested in unresolved radio sources then simply changing the value of :math:`S_{\mathrm{max}}`, which demarcates the unresolved and resolved sources, will suffice.


Initialisation
^^^^^^^^^^^^^^
The first step to use this package is to initialise the properties of the extragalactic point sources. This is done using the class :class:`meps.eps`. If you give no arguments default settings are assumed.

There are total 11 available optional arguments which include all of the six physical parameters listed in table 2 from our paper. If you want to choose a different set of parameters, then your python script should have the following initialisation

.. code:: python

   from epspy import meps
   obj = meps.eps(log2Nside=6, logSmin=-2,logSmax=-1,dndS_form=0, nu_o=150e6, beta_o=2.681,sigma_beta=0.5, amp=7.8e-3,gam=0.821, path='', lbl='')

Replace the above values by values of your choice. For more details see the :ref:`api`.

.. _ref-freq:


.. code:: bash

    mpirun -np 4 python eg_script.py

However, in general, and for more realistic flux density ranges and high-resolution maps, we recommend using the package on HPCs.


Reference frequency
^^^^^^^^^^^^^^^^^^^

The next step is to run :func:`ref_freq`. The function :func:`ref_freq` does 3 tasks:-

1. It calculates the total number of point sources (according to equation 6 from the paper) corresponding to your specified ``logSmin`` and ``logSmax``. Then it creates a 'clustered' number density of point sources on the sky (according to equation 9 from the paper), fluctuation for which follows the 2PACF whose parameters are set by ``amp`` and ``gam``.
   
2. Next, it visits each pixel on the sky and assigns each source a flux density chosen from a flux density distribution function, :math:`\mathrm{d}n/\mathrm{d}S`. The default choice of the form of :math:`\mathrm{d}n/\mathrm{d}S` is the sum-of-two-inverse-double-power-laws by `Gervasi et al (2008) <https://iopscience.iop.org/article/10.1086/588628>`_. However, the package gives one the functionality to work with a different form of :math:`\mathrm{d}n/\mathrm{d}S`. Currently, there are 2 additional available forms, namely the :math:`5^{\mathrm{th}}` `(Intema et al 2017) <https://www.aanda.org/articles/aa/full_html/2017/02/aa28536-16/aa28536-16.html>`_ and :math:`7^{\mathrm{th}}` `(Mandal et al 2021) <https://www.aanda.org/articles/aa/full_html/2021/04/aa39998-20/aa39998-20.html>`_ order log-log polynomial fit to :math:`S^{-2.5}\mathrm{d}n/\mathrm{d}S`. (The quantity :math:`S^{-2.5}\mathrm{d}n/\mathrm{d}S` is called the Euclidean number count).

3. Finally, it assigns a spectral index to each source from a normal distribution (see equation 5 from the paper). The normal distribution is set by ``beta_o`` and ``sigma_beta``.

The function does not return anything, but produces four files, namely ``n_ps.npy``, ``Tb_o_individual.npy``, ``Tb_o_map.npy``, and ``beta.npy``. The files will be put in the path specified by the keyword argument ``path`` during initialisation. The files are described below.

- ``n_ps.npy`` is a 1D array which stores number density of the point sources in units of number per pixel. ``n_ps[i]`` gives the number of sources on the :math:`i^{\mathrm{th}}` pixel, where :math:`i=0,1,\ldots,N_{\mathrm{pix}}-1`. Throughout this package we work with the standard ``HEALPix`` RING-ordered format. Following the notation of our paper, ``n_ps`` represents :math:`n_{\mathrm{ps}}(\hat{n})`, LHS in equation (9). (Note that in general ``n_ps[i]`` will not be a natural number; we simulate for a rounded-off value.)

- ``Tb_o_individual.npy`` is an array of arrays of unequal size. For 'realistic' values of point sources parameters, this files will be huge (but for default settings it will of size ~ 17 MB each). Each of ``Tb_o_individual[0]``, ``Tb_o_individual[1]``, ..., is an array and they are total :math:`N_{\mathrm{pix}}` in number corresponding to :math:`N_{\mathrm{pix}}` pixels. The last array is ``Tb_o_individual[Npix-1]``. The length of array ``Tb_o_individual[i]`` is equal to the number of sources on the :math:`i^{\mathrm{th}}` pixel, which is ``round(n_ps[i])``. The values itself are the brightness temperature contributed by each source in kelvin at reference frequency, :math:`\nu_0`. Following the notation of our paper, it represents :math:`T_{\mathrm{ps},ij}(\nu_0)`, LHS in equation (10).

- The structure of ``beta.npy`` is same as ``Tb_o_individual.npy``. The values itself are the spectral indices assigned to each source. Following the notation of our paper, it represents :math:`\beta_{ij}`. (Note that both ``Tb_o_individual.npy`` and ``beta.npy`` are 'Object' arrays. If you want to load them yourself then set ``allow_pickle=True`` in ``numpy.load()``.)

- ``Tb_o_map`` is an array similar in structure to ``n_ps``. It is the pixel wise brightness temperature contributed by the extragalactic point sources at the reference frequency, :math:`\nu_0`. Following the notation of our paper, it represents :math:`T_{\mathrm{ps},i}(\nu_0)`. Thus, ``Tb_o_map[i] = numpy.sum(Tb_o_individual[i])``.

To run up to :func:`ref_freq` the following should be your python script

.. code:: python

   from epspy import meps
   
   obj = meps.eps()
   
   obj.ref_freq()


Note that the default value of :math:`S_{\mathrm{min}}` and the number of pixels in the package are different from the fiducial model values used in our paper, which are :math:`S_{\mathrm{min}}=10^{-6}\,\mathrm{Jy}` and :math:`N_{\mathrm{pix}}=3145728`. This is done so as to lessen the cost of default runs and enable the user to try out the package on a personal computer. Accordingly, in the package we have set :math:`S_{\mathrm{min}}=0.01\,\mathrm{Jy}` and ``nside`` :math:`=2^6` so that :math:`N_{\mathrm{pix}}=49152`. However, to simulate a realistic scenario we recommend setting these parameters to the fiducial values as in our paper. Since fiducial run can be expensive, the code should be run on high performance clusters. We have made this package parallel with message passing interface (MPI).


.. _gen-freq:

General frequency
^^^^^^^^^^^^^^^^^

The next important task is performed by the function :func:`gen_freq`. It scales the brightness temperature at reference frequency for each source according to a power law to a desired range of frequencies. The desired frequencies should be supplied (in Hz) as a :mod:`numpy` array to this function. For example, the following should be your python script

.. code:: python

   from epspy import meps
   
   obj = meps.eps()
   
   obj.ref_freq()

   obj.gen_freq(nu = 1e6*numpy.arange(50,201))

The default value of frequencies at which :func:`gen_freq` will scale is :math:`\nu=50,51,\ldots,200\,` MHz. This function does not return anything but produces three files namely ``Tb_nu_map.npy``, ``Tb_nu_glob.npy``, and ``nu_glob.npy`` in the path specified by the keyword argument ``path`` during initialisation. The files are described below.

1. ``Tb_nu_map`` is a 2D array of shape :math:`N_{\mathrm{pix}}\times N_{\nu}`, so that ``Tb_nu_map[i,k]`` gives the brightness temperature due to extragalactic point sources on the :math:`i^{\mathrm{th}}` pixel at ``nu[k]`` frequency. :math:`N_{\nu}` is the number of frequencies. Following the notation of our paper, it represents :math:`T_{\mathrm{ps}}(\hat{n},\nu)`, the LHS in equation (11). This quantity is perhaps the most important output of this package. This is the quantity data analysts from different 21-cm experiments can to add to their simulated foregrounds model.

2. ``Tb_nu_glob`` is derived directly from ``Tb_nu_map``. It is the sky average of the map at each frequency and is thus a 1D array. It is calculated as ``Tb_nu_glob = numpy.mean(Tb_nu_map,axis=0)``. Following the notation of our paper, it represents :math:`\langle T_{\mathrm{ps}}\rangle(\nu)`, the LHS in equation (12).

3. ``nu_glob.npy`` is simply the frequency array you gave, else it is the default value.

Note that :func:`ref_freq` and :func:`gen_freq` functions deal with ``Tb_o_individual.npy`` and ``beta.npy``. These data can easily be 10s of GB in size for 'realistic' ``logSmin`` and ``logSmax``. Common PCs have at least ~ 4 GB RAM. We therefore recommend to run this package on supercomputers. For users who use a slurm job schedular must specify ``#SBATCH --mem-per-cpu=[size in MB]`` in their job submission scipt. A recommendation for 'size in MB' will be printed when you initialise your class object if the requirements are more than 2 GB. We emphasize that the default values are chosen such that the package can be run on a PC. In the paper we worked with ``logSmin=-6`` for which both ``Tb_o_individual`` and ``beta`` are ~ 34 GB in size. We used ``mem-per-cpu=80000``.


Chromatic distortions
^^^^^^^^^^^^^^^^^^^^^

Until now the results generated have been experiment independent. So ``Tb_nu_map`` and hence ``Tb_nu_glob`` generated do NOT account for chromatic distortions. They are simply the model outputs for foregrounds due to extragalactic point sources. However, in reality because of the chromatic nature of the antenna beam the actual foregrounds spectrum registered will be different. Use the function :func:`couple2D()` to account for the chromaticity. It essentially couples the foregrounds to the beam directivity, i.e., it will multiply the point sources map to beam directivity, and average over the pixels. See equation (14) from our paper.

Since the antenna temperature is experiment specific, you will need to provide an external data file: the beam directivity pattern, :math:`D=D(\hat{n},\nu)`. Its structure should be the same as ``Tb_nu_map``, i.e., it should be a 2D array of shape :math:`N_{\mathrm{pix}}\times N_{\nu}`, such that ``D[i,k]`` should give the beam directivity at :math:`i^{\mathrm{th}}` pixel at ``nu[k]`` frequency. The pixel ordering should be the standard HEALPix RING ordering. The frequencies at which you generate your data :math:`D` should be the same as the frequencies you gave in ``gen_freq()``. (In case you forgot, :func:`gen_freq` will have saved the frequency array in your ``obj.path`` path by the name of ``nu_glob.npy``.) Put this array :math:`D` in your ``obj.path`` path by the name of ``D.npy``. ``obj.path`` is the default path where the code will look for a file named 'D.npy'. You can always choose a different path and name; use the optional argument ``bd`` for this purpose.

As a consistency check it should be noted that 

.. math::

   \frac{1}{4\pi}\int D(\hat{n},\nu)\,\mathrm{d}\Omega=1\,,

for any frequency in the range.

Only after running :func:`ref_freq` and :func:`gen_freq`, run :func:`couple2D` as

.. code:: python

   from epspy import meps

   obj = meps.eps()

   obj.ref_freq()

   obj.gen_freq()
   
   #If you have already ran ref_freq and gen_freq previously then comment
   #obj.ref_freq() and obj.gen_freq(). 
   obj.couple2D(bd='full/path/to/beam_directivity.npy')

The return value is ``None``. This function will generate a file called ``T_ant.npy`` in your ``obj.path``. Following our notation from the paper the obtained quantity is :math:`T_{\mathrm{A,ps}}(\nu)`, the LHS in equation (14). This will be a 1D array with length being the number of frequencies, :math:`N_{\nu}`. 

This function will also print the best-fitting parameters (along with :math:`1\sigma` uncertainty) :math:`T_{\mathrm{f}}, \beta_{\mathrm{f}}` and :math:`\Delta\beta_{\mathrm{f}}` based on a simple least-squares fitting of power-law-with-a-running-spectral-index function (given in equation 15 in our paper) to the antenna temperature data :math:`T_{\mathrm{A,ps}}(\nu)`.

Visualisation
^^^^^^^^^^^^^

The final part of the package is to visualise the results. Users can always write their own scripts to produce figures. However, best efforts have been made as part of this package to produce publication-ready plots. Main data for inspection is in the file ``Tb_nu_map.npy``. Each of ``Tb_nu_map[:,k]`` is an array in the standard RING-ordered ``HEALPix`` format and is thus ready for visualisation as a Mollweide projection. If you are interested in inspecting the global spectrum of extragalactic emission, i.e, temperature as a function of frequency check for the data file ``Tb_nu_glob.npy`` which was generated by :func:`gen_freq`.

Use the function :func:`visual` for both the above purposes. It is possible to make several other additional figures by simply setting the optional arguments to ``True``. This function is again a method of class object :class:`meps.eps` and thus your python script should contain

.. code:: python
   
   from epspy import meps
   
   obj = meps.eps()

   obj.ref_freq()

   obj.gen_freq()

   obj.couple2D()

   #comment out obj.ref_freq(), obj.gen_freq(), obj.couple2D() if you have already run them.
   obj.visual()

For all the available options for this function see the :ref:`api`. This function will produce figures in the path specified during initialisation.


.. rubric:: Footnotes

.. [#f1] Note that this package does not contain the functionality for generating the brightness temperature corresponding to galactic emission, using the customary Global Sky Maps (GSM) modelling. Also, Bayesian inference is not a part of this package. For these we used the *REACH* data analysis pipeline developed by `Anstey et al (2021) <https://academic.oup.com/mnras/article/506/2/2041/6307526?login=true>`_.

