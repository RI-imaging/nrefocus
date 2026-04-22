Setting the NDArray Backend
===========================

.. _sec_doc_ndarray_backend:

Since version 0.6.0, `nrefocus` allows the user to leverage their CUDA GPU
via the :class:`.RefocusCupy` refocus interface and the ``CuPy`` library.
Additionally, you can control the desired ndarray backend.
An "ndarray backend" is defined as the library used to define the
ndarrays in nrefocus during runtime. By default it is set to ``'numpy'``.

If you are using the :class:`.RefocusCupy` refocus interface, it is
recommended to set the backend to ``'cupy'``. See the script below 
for details on how to do this.
For more info, see the `CuPy library <https://cupy.dev/>`_.

There are currently two available ndarray backends: ``'numpy'`` and ``'cupy'``.

Controlling the ndarray backend
-------------------------------

``nrefocus`` allows users to swap between these backends with the
:func:`nrefocus.set_ndarray_backend()` function. To check which backend is
currently in use just run :func:`nrefocus.get_ndarray_backend()`.

.. admonition:: Matching the NDArray Backend with the Refocus interface

    Always try to match the NDArray Backend with the Refocus interface, as shown in
    the example below, otherwise you will run into warnings or errors.

    To summarise:
        - 	``'numpy'`` (default) backend works as expected with the
            :class:`.RefocusNumpy` and :class:`.RefocusPyFFTW` classes.
        - 	``'cupy'`` backend works as expected with the :class:`.RefocusCupy`
            and :class:`.RefocusNumpy` classes. This is because NumPy is
            `quite clever <https://numpy.org/doc/stable/user/basics.interoperability.html#example-cupy-arrays>`_.
            The :class:`.Refocus interfacePyFFTW` class will raise an Error if used
            with the ``'cupy'`` backend.


.. code-block:: python

    import nrefocus

    print(nrefocus.get_ndarray_backend())
    # <module 'numpy' from '~\\numpy\\__init__.py'>

    nrefocus.set_ndarray_backend('cupy')  # swap to the 'cupy' backend
    print(nrefocus.get_ndarray_backend())
    # <module 'cupy' from '~\\cupy\\__init__.py'>


Example use of 'cupy' backend for Field Refocussing
---------------------------------------------------

.. code-block:: python

    import nrefocus

    # set the ndarray backend to cupy
    nrefocus.set_ndarray_backend("cupy")
    xp = nrefocus.get_ndarray_backend()

    # load your experimental data here
    # some fake data:
    pixel_size = 1e-6
    rf = nrefocus.RefocusCupy(field=xp.arange(256).reshape(16, 16),
                              wavelength=8.25 * pixel_size,
                              pixel_size=pixel_size,
                              medium_index=1.533,
                              distance=0,
                              kernel="helmholtz",
                              padding=False)

    refocused = rf.propagate(distance=2.13 * pixel_size)
    # get the array back on the CPU (numpy array)
    refocused_cpu = refocused.get()
