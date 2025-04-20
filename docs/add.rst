Additional features
-------------------

Star formation rate, couplings, & reionization history
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Related to 21-cm signal computation, ``ECHO21`` can be used to calculate Ly :math:`\alpha` or collisional coupling and SFRD. Further, ``ECHO21`` can be utilized to generate the reionization history and also the CMB optical depth. The best way to see this in action is to look at the jupyter notebook in the `example <https://github.com/shikharmittal04/echo21/tree/master/examples>`_ folder.

Saving & loading pipeline object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose you want to save the :class:`echopipeline.pipeline` class object ``myobj`` (see section :ref:`single`) and reuse it for later purpose or simply want to check what parameters you gave. You can do this using the saving and loading functions available in the module :mod:`echopipeline`. To save, put the following in your script

.. code:: python
    
    save_pipeline(myobj, 'myechoobj')

where ``myechoobj`` is the name of the object file. If you check your ``output_*`` folder it will contain a file ``myechoobj.pkl``. Note that when you define class object :class:`echopipeline.pipeline`, the object is automatically saved, so you may never need to save it by yourself.

Now if you want to load it later, then put the following in your script

.. code:: python

    myobj=load_pipeline('/full/path/to/myechoobj.pkl')

where you need to supply the full path to your file ``myechoobj.pkl``.