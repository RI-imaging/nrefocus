==================
Basics of nrefocus
==================

.. _sec_getting_started_basics:

This section gives a minimal example for numerical refocussing of a single 2D
complex field with ``nrefocus``.

If you want to use the GPU via CuPy, see :ref:`sec_doc_ndarray_backend`.
If you want to refocus stacks shaped ``(..., y, x)``, see :ref:`sec_3d_stacks`.

.. code-block:: python

    import numpy as np
    import nrefocus

    # A single complex 2D field: (y, x)
    ny, nx = 128, 96
    field = (np.random.randn(ny, nx) + 1j*np.random.randn(ny, nx)
             ).astype(np.complex128)

    pixel_size = 1e-6
    # you can instead use `RefocusPyFFTW` or `RefocusCupy` if you want
    rf = nrefocus.RefocusNumpy(
        field=field,
        wavelength=8.25 * pixel_size,
        pixel_size=pixel_size,
        medium_index=1.533,
        padding=True,
    )

    refocused = rf.propagate(distance=2.13 * pixel_size)
    print(refocused.shape)  # (128, 96)
