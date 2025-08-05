
Beyond standard model cosmologies
---------------------------------

Interacting dark matter model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ECHO21`` can be used to study a special class of interacting dark matter model: Coulomb-like DM. We assume that all DM particles are Coulomb-like and that they can interact with all baryons. The interaction cross-section for such a system is given by :math:`\sigma(v_{\mathrm{b}\chi}) = \sigma_0(v_{\mathrm{b}\chi}/c)^{-4}`, where :math:`v_{\mathrm{b}\chi}` is the baryon-DM relative velocity. In addition to cooling, IDM leads to a suppression and delay in star formation. We follow the strategy outlined by `Driskell et al. (2022) <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.103525>`__.

IDM requires two additional parameters, namely :math:`m_{\chi}` (mass of Coulomb-like DM) and :math:`\sigma_{45}` (dimensionless cross-section, expressed as :math:`\sigma_0/10^{-45}\,\mathrm{m^2}`). If you want work with IDM, supply both these parameters in your ``cosmo`` dictionary. Note that :math:`m_{\chi}` should be provided in units of GeV. The ``cosmo`` dictionary now looks like 

.. code:: python
   
   cosmo = {'Ho':67.4,'Om_m':0.315,'Om_b':0.049,'sig8':0.811,'ns':0.965,'Tcmbo':2.725,'Yp':0.245, 'mx_gev':1.0,'sigm45':1.0}

For the SFRD, you can work either with physically-motivated or semi-empirical model. The arguments ``'hmf'`` and ``'mdef'`` in the dictionary will be ignored since in the current version we work with only ``'tinker08'`` HMF. Further, IDM can be run only for some specific values of :math:`m_{\chi}` and :math:`\sigma_{45}`. These are 15 numbers starting from :math:`10^{-4}` to :math:`10^{3}` separated by 0.5 dex for both, :math:`m_{\chi}` and :math:`\sigma_{45}`. For any other number, the code will run for the nearest number. For example, if you provide :math:`m_{\chi}=15`, then the code will actually compute for :math:`m_{\chi}=10`.