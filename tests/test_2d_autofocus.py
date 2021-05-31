"""Test 2D autorefocusing"""
import numpy as np

import nrefocus

from test_helper import load_cell


def test_2d_autofocus_cell_helmholtz_average_gradient():
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )

    # attempt to autofocus with standard arguments
    d = rf.autofocus(metric="average gradient",
                     minimizer="lmfit",
                     interval=(-5e-6, 5e-6))
    assert np.allclose(d, -8.781356558557544e-07, atol=0)

    nfield = rf.propagate(d)
    assert np.allclose(nfield[10, 10],
                       1.0455603934920419 - 0.020475662236633177j,
                       atol=0)


def test_2d_autofocus_return_field():
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )

    # attempt to autofocus with standard arguments
    d, nfield = rf.autofocus(
        metric="average gradient",
        minimizer="lmfit",
        interval=(-5e-6, 5e-6),
        minimizer_kwargs={"ret_field": True}
        )

    nfield2 = rf.propagate(d)
    assert np.all(nfield == nfield2)
