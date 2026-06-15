.. _sec_3d_stacks:

=========
3D stacks
=========

Since version 0.7.0, the :class:`.RefocusNumpy`, :class:`.RefocusPyFFTW` and
:class:`.RefocusCupy` interfaces accept either a single 2D field shaped
``(y, x)`` or a stack shaped ``(..., y, x)`` (for example ``(n, y, x)``).
The last two axes are always interpreted as the spatial axes and any leading
axes are treated as batch dimensions. If you pass an
``n``-dimensional input, you get an output array with the same
``n``-dimensional shape.

.. admonition:: `nrefocus.refocus_stack`

	The convenience functions `nrefocus.refocus_stack` and
	`nrefocus.autofocus_stack` may be removed in the
	future, as  `nrefocus` now accepts in 3D data. See
	`Issue #28 <https://github.com/RI-imaging/nrefocus/issues/28>`_


.. code-block:: python

    import numpy as np
    import nrefocus

    # stack of complex 2D fields: (n, y, x)
    field = (np.random.randn(4, 128, 96) + 1j*np.random.randn(4, 128, 96)
             ).astype(np.complex128)

    pixel_size = 1e-6
    rf = nrefocus.RefocusNumpy(
        field=field,
        wavelength=8.25 * pixel_size,
        pixel_size=pixel_size,
        medium_index=1.533,
        padding=True,
    )
    refocused = rf.propagate(distance=2.13 * pixel_size)
    print(refocused.shape)  # (4, 128, 96)
