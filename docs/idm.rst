
Beyond standard model cosmologies
---------------------------------

:Authors: `Shikhar Mittal <https://sites.google.com/view/shikharmittal/home>`_ and Prakhar Bansal
:Paper: `Mittal et al (2026) <https://arxiv.org/abs/2605.00991>`_


Interacting dark matter model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ECHO21`` can be used to study a special class of interacting dark matter (IDM) model: Coulomb-like DM. We assume that *all DM particles are Coulomb-like* and that they can interact with *all baryons*. The interaction cross-section for such a system is given by :math:`\sigma(v_{\mathrm{b}\chi}) = \sigma_0(v_{\mathrm{b}\chi}/c)^{-4}`, where :math:`v_{\mathrm{b}\chi}` is the baryon-DM relative velocity. In addition to cooling, IDM leads to a suppression and delay in star formation.

IDM requires two additional parameters, namely :math:`m_{\chi}` (mass of Coulomb-like DM) and :math:`\sigma_{45}` (dimensionless cross-section, expressed as :math:`\sigma_0/10^{-45}\,\mathrm{m^2}`). If you want work with IDM, supply both these parameters in your ``cosmo`` dictionary. Note that :math:`m_{\chi}` should be provided in units of GeV. The ``cosmo`` dictionary now looks like 

.. code:: python
   
   cosmo = {'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'sig8':0.811,'ns':0.965,'Tcmbo':2.725,'Yp':0.245, 'mx_gev':1.0,'sigma45':1.0}

``ECHO21`` interfaces with a modified version of `CLASS <https://github.com/kboddy/class_public/tree/dmeff>`_ to get the (linear) matter power spectrum for IDM (at :math:`z=0`) which is then fed into ``COLOSSUS`` to obtain variance :math:`\sigma(M)`. We translate the power spectrum and variance to an arbitrary redshift linearly with a scale-independent growth factor. Accordingly, we can obtain HMF at an arbitrary redshift.

A few points to note on which the latest version of IDM implementation of ``ECHO21`` differ from the one described in `Mittal et al (2026) <https://arxiv.org/abs/2605.00991>`_.

   - Rather than fixing the primordial spectrum amplitude :math:`A_\mathrm{s}`, we fix the normalisation  :math:`\sigma_8` by the user defined value.

   - We adopt a top-hat filter to smoothen the matter density field for the calculation of variance :math:`\sigma(M)`. Previously, a sharp-k filter was used. (Top-hat make the simulation a bit faster and is more commonly used in the literature.)

   - Previously, only Tinker et al (2008) HMF was available for IDM simulations. Now, the user can choose any standard HMF. Moreover, the halo mass definition was restricted to :math:`M_{500\mathrm{c}}` (mass enclosed within a radius where the mean density is 500 times the critical matter density). Now, the user can choose any halo mass definition supported by `COLOSSUS <https://pypi.org/project/colossus/>`_.