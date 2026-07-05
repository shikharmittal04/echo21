Basics
======

Overview
--------

:Name: Exploring Cosmos with Hydrogen Observation
:Author: `Shikhar Mittal <https://sites.google.com/view/shikharmittal/home>`_
:Homepage: https://github.com/shikharmittal04/echo21
:Paper: `Mittal et al (2026) <https://doi.org/10.1093/rasti/rzag001>`_

Why do you need this code?
--------------------------

``ECHO21`` is a fast and flexible Python package for modelling the global 21-cm signal across cosmic history - from the dark ages to reionization. Given a set of astrophysical and cosmological parameters, the code self-consistently generates the global 21-cm signal, neutral hydrogen fraction, and CMB optical depth.

Designed for both precision studies and large-scale parameter inference, ``ECHO21`` combines physical realism with computational efficiency. Its key features include:

- simultaneous variation of astrophysical and cosmological parameters,
- flexible prescriptions for halo mass functions and star formation models,
- inclusion of Ly :math:`\alpha` heating and detailed IGM thermal evolution,
- support for non-standard cosmologies such as interacting dark matter `(Mittal et al 2026) <https://arxiv.org/abs/2605.00991>`_
- UV luminosity functions

A single realization of the signal can be generated in :math:`\sim 1` second, making the code ideally suited for Bayesian inference, emulator training, and large parameter-space explorations.

``ECHO21`` is MPI-parallelized and scalable - equally at home on a laptop or a high-performance computing cluster.

Read more about it in the paper `Mittal et al (2026) <https://doi.org/10.1093/rasti/rzag001>`_.

For installation instructions, see :ref:`install`.

Quick start
-----------

The following code more or less captures the main functionalities of this package.

.. code:: python

   import echo21

   pipe = echo21.pipeline()
   pipe.run_simulation()

Save the above code as (say) ``run_echo21.py`` and run it as

.. code:: bash

    python run_echo21.py

Running the above will generate an output folder with the name `output_<YYYYMMDD-hhmmss>` which contains three files. To learn how to set the astrophysical or cosmological parameters, halo mass function, star formation model, redshifts at which to evaluate the global signal, and structure of the output files see :ref:`detexp`. To learn about the physics of this package see our `paper <https://doi.org/10.1093/rasti/rzag001>`_. 

Once you have an understanding of the structure of output files, you can write your own scripts to create figures. To help you get started, see the jupyter notebooks in the example folder.

To see what parameters you are running for, you can add ``pipe.print_input()`` to ``run_echo21.py``.

License and citation
--------------------
The software is free to use on the MIT open-source license. If you use the software then please consider citing `Mittal et al (2026) <https://doi.org/10.1093/rasti/rzag001>`_.

If the code is used in a project where the author has provided significant scientific input, guidance on methodology, or assistance with interpretation of results, then co-authorship on resulting publications is expected, following standard academic practice.

Users are encouraged to contact the author when using the code for new scientific applications or major projects.