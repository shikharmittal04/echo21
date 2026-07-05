Additional features
===================

Ultraviolet Luminosity Function
-------------------------------

In ECHO21 you can compute the Ultraviolet Luminosity Function (UVLF) and other derived quantities such as gradient of number count of the bright galaxies given a limiting magnitude and the total number of galaxies upto a given redshift a survey can see. 

Note that throughout we have assumed that the star formation efficiency :math:`f_{\star}` and star formation timescale :math:`t_{\star}` are independent of halo mass and redshift. Further, UVLF computation is relevant only for semi-empirical SFRD model. When you choose the semi-empirical SFRD model, UVLF is automatically computed along with related quantities otherwise it is ignored.

**Theory in brief**

We define the luminosity of galaxies hosted by a halo of mass :math:`M` as

:math:`{L}_{\mathrm{UV}} = \dot{M}_{\star} \mathcal{L}_{\mathrm{UV}}`,

where

:math:`\dot{M}_{\star} = f_{\star} \frac{\Omega_\mathrm{b}}{\Omega_\mathrm{m}} \frac{M}{t_{\star}H(z)^{-1}}`

and :math:`\mathcal{L}_{\mathrm{UV}}=8.695\times10^{20}\,\mathrm{WHz^{-1}(\mathrm{M}_{\odot}yr^{-1})^{-1}}`. `(Madau & Dickinson 2014) <https://doi.org/10.1146/annurev-astro-081811-125615>`_

The absolute magnitude (AB) is given by `(Oke 1974) <http://dx.doi.org/10.1086/190287>`_

:math:`M_\mathrm{UV}=-2.5\log_{10}\left(\frac{L_\mathrm{UV}}{4\pi d^2_{10}}\right)-56.1\,,`

where luminosity is in units of :math:`\mathrm{WHz^{-1}}` and distance :math:`d_{10}` is in units of m. (When :math:`d_{10}=10\,\mathrm{pc}` is replaced by the actual distance to the star we get apparent magnitude.)

We define the UVLF as (typically as a function of absolute magnitude)

:math:`\frac{\mathrm{d}\phi}{\mathrm{d}M_\mathrm{UV}} = \frac{\mathrm{d}n}{\mathrm{d}M}\frac{\mathrm{d}M}{\mathrm{d}M_\mathrm{UV}}`,

where :math:`\frac{\mathrm{d}n}{\mathrm{d}M}` is the halo mass function (HMF) and :math:`\frac{\mathrm{d}M}{\mathrm{d}M_\mathrm{UV}}` is obtained from the above equations.

The gradient of number count of galaxies brighter than a limiting magnitude :math:`m_\mathrm{UV}^{\mathrm{lim}}` (surveys typically give their limiting *apparent* magnitudes) per unit survey area is given by

:math:`\frac{\mathrm{d}N(m_{\mathrm{UV}}^{\mathrm{lim}})}{\mathrm{d}z}=\frac{c}{H(z)}\left(\frac{d_\mathrm{L}}{1+z}\right)^2\int_{-\infty}^{m_{\mathrm{UV}}^{\mathrm{lim}}}\ \frac{\mathrm{d}\phi}{\mathrm{d}m_\mathrm{UV}} \mathrm{d} m_{\mathrm{UV}}\,,`

We have used the relation between apparent and absolute magnitudes

:math:`m_\mathrm{UV}-M_\mathrm{UV}=5\log_{10}(d_\mathrm{L}/d_{10})`,

where :math:`d_\mathrm{L}` is the luminosity distance. Accordingly, at a fixed redshift :math:`\mathrm{d} m_{\mathrm{UV}} = \mathrm{d} M_{\mathrm{UV}}`.

Integrating the above equation over redshift gives the total number of galaxies brighter than a limiting magnitude upto a given redshift :math:`z_\mathrm{max}` per unit survey area.

Star formation rate, couplings, & reionization history
------------------------------------------------------

Related to 21-cm signal computation, ``ECHO21`` can be used to calculate Ly :math:`\alpha` or collisional coupling and SFRD. Further, ``ECHO21`` can be utilized to generate the reionization history and also the CMB optical depth. The best way to see this in action is to look at the jupyter notebook in the `example <https://github.com/shikharmittal04/echo21/tree/master/examples>`_ folder.