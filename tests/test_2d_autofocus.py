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


def test_2d_autofocus_cell_helmholtz_std_gradient():
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )

    # attempt to autofocus with standard arguments
    d = rf.autofocus(metric="std gradient",
                     minimizer="lmfit",
                     interval=(-5e-6, 5e-6))
    assert np.allclose(d, -1.027262300918845e-06, atol=0)

    nfield = rf.propagate(d)
    assert np.allclose(nfield[10, 10],
                       1.0397563526680962 - 0.023285740461459085j,
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
        ret_field=True,
    )

    nfield2 = rf.propagate(d)
    assert np.all(nfield == nfield2)


def test_2d_autofocus_return_grid_field():
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )

    # attempt to autofocus with standard arguments
    d, (dgrid, mgrid), nfield = rf.autofocus(
        metric="average gradient",
        minimizer="lmfit",
        interval=(-5e-6, 5e-6),
        ret_grid=True,
        ret_field=True,
    )

    idx_metric_min = np.argmin(mgrid)
    idx_distance = np.argmin(np.abs(dgrid - d))
    assert idx_metric_min == idx_distance


def test_2d_autofocus_small_interval():
    """
    Test for IndexError failure for brute method
    (brute_step too small).

                for k in range(N - 1, -1, -1):
        >           thisN = Nshape[k]
        E           IndexError: tuple index out of range

        /scipy/optimize/optimize.py:3276: IndexError
    """
    wavelength = 647e-9
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=wavelength,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )

    # attempt to autofocus with interval smaller than brute_step
    rf.autofocus(
        metric="average gradient",
        minimizer="lmfit",
        interval=(0, 1.9 * wavelength),
    )
