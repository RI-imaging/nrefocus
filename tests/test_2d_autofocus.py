"""Test 2D autorefocusing"""
import numpy as np
import pytest

import nrefocus
from nrefocus.roi_handling import ROIValueError
from nrefocus.metrics.mt_spectrum import MetricSpectrumValueError

from test_helper import load_cell


@pytest.mark.parametrize(
    "metric, roi, expected_d, expected_field_point",
    [
        ("average gradient", None,
         -8.781356558557544e-07,
         1.0455603934920419 - 0.020475662236633177j),
        ("average gradient", [10, 10, 100, 100],
         -8.795139152651752e-07,
         1.0455078513415523 - 0.020478153810279703j),
        ("rms contrast", None,
         4.999999999974661e-06,
         1.0505454858452632 - 0.022163822293036956j),
        ("rms contrast", [10, 10, 100, 100],
         4.999999999974661e-06,
         1.0505454858452632 - 0.022163822293036956j),
        ("spectrum", None,
         -5e-06,
         1.026638094465674 - 0.02923150877539289j),
        # roi doesn't work with spectrum, see test below
        ("std gradient", None,
         -8.781518791456171e-07,
         1.0455597758297575 - 0.02047568956477154j),
        ("std gradient", [10, 10, 100, 100],
         -8.796339535413204e-07,
         1.0455032687136372 - 0.02047838714147014j),
        ("med gradient", None,
         2.8018890771670997e-07,
         1.0422687231100252 - 0.01375193149358192j),
        ("med gradient", [10, 10, 100, 100],
         2.18410667763866e-07,
         1.042245183739662 - 0.015481320442455255j),
    ])
def test_2d_autofocus_cell_helmholtz_metric_roi(
        metric, roi, expected_d, expected_field_point):
    """Check that roi works for the relevant metrics."""
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )

    # attempt to autofocus with standard arguments
    d = rf.autofocus(metric=metric,
                     minimizer="lmfit",
                     interval=(-5e-6, 5e-6),
                     roi=roi)
    assert np.allclose(d, expected_d, atol=0, rtol=1e-4)

    nfield = rf.propagate(d)
    assert np.allclose(nfield[10, 10],
                       expected_field_point,
                       atol=0)


def test_2d_autofocus_cell_helmholtz_average_gradient():
    """attempt to autofocus with standard arguments"""
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )
    d = rf.autofocus(metric="average gradient",
                     minimizer="lmfit",
                     interval=(-5e-6, 5e-6))
    assert np.allclose(d, -8.781335587979859e-07, atol=0)

    nfield = rf.propagate(d)
    assert np.allclose(nfield[10, 10],
                       1.0455597758297575 - 0.02047568956477154j,
                       atol=0)


def test_2d_autofocus_cell_helmholtz_std_gradient():
    """attempt to autofocus with std gradient"""
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )
    d = rf.autofocus(metric="std gradient",
                     minimizer="lmfit",
                     interval=(-5e-6, 5e-6))
    assert np.allclose(d, -8.781518791456171e-07, atol=0)

    nfield = rf.propagate(d)
    assert np.allclose(nfield[10, 10],
                       1.0455597758297575 - 0.02047568956477154j,
                       atol=0)


def test_2d_autofocus_cell_helmholtz_med_gradient():
    """attempt to autofocus with med gradient"""
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )
    d = rf.autofocus(metric="med gradient",
                     minimizer="lmfit",
                     interval=(-5e-6, 5e-6))
    assert np.allclose(d, 2.8018890771670997e-07, atol=0)

    nfield = rf.propagate(d)
    assert np.allclose(nfield[10, 10],
                       1.0422687231100252 - 0.01375193149358192j,
                       atol=0)


def test_2d_autofocus_cell_helmholtz_spectrum_roi():
    """Show that the spectrum metric doesn't allow roi."""
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )

    # attempt to autofocus with spectrum metric with an roi
    with pytest.raises(MetricSpectrumValueError):
        # doesn't allow an roi with spectrum
        d = rf.autofocus(metric="spectrum",  # noqa: F841
                         minimizer="lmfit",
                         interval=(-5e-6, 5e-6),
                         roi=[10, 100, 10, 100])


def test_2d_autofocus_return_field():
    """Basic check of the field after autofocus"""
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
    """Check the calculation of the metric after autofocus"""
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


def test_2d_autofocus_cell_roi():
    """Compare the roi given to rf.autofocus with numpy indexing"""

    field = load_cell("HL60_field.zip")
    roi = [10, 20, 100, 200]
    slice_roi = (slice(10, 100), slice(20, 200))

    rf_1 = nrefocus.iface.RefocusNumpy(field=field,
                                       wavelength=647e-9,
                                       pixel_size=0.139e-6,
                                       kernel="helmholtz")

    roi_1 = rf_1.parse_roi(roi=roi)
    assert roi_1 == slice_roi

    field_1 = field[slice_roi]
    field_2 = field[roi_1]
    field_3 = field[roi[0]: roi[2], roi[1]: roi[3]]

    assert np.array_equal(field_1, field_2)
    assert np.array_equal(field_1, field_3)


def test_2d_autofocus_cell_roi_list_of_slices():
    """Use a list of slices to slice the array."""

    field = load_cell("HL60_field.zip")
    roi = [10, 20, 100, 200]
    slice_roi = [slice(10, 100), slice(20, 200)]

    rf_1 = nrefocus.iface.RefocusNumpy(field=field,
                                       wavelength=647e-9,
                                       pixel_size=0.139e-6,
                                       kernel="helmholtz")

    slice_roi_1 = rf_1.parse_roi(roi=slice_roi)
    assert slice_roi_1 == tuple(slice_roi)

    field_1 = field[slice_roi_1]
    field_2 = field[roi[0]: roi[2], roi[1]: roi[3]]

    assert np.array_equal(field_1, field_2)


def test_2d_autofocus_cell_roi_nones():
    """Use None when slicing with roi."""

    field = load_cell("HL60_field.zip")
    roi = [None, None, 10, 100]
    slice_roi = (slice(None, 10), slice(None, 100))

    rf_1 = nrefocus.iface.RefocusNumpy(field=field,
                                       wavelength=647e-9,
                                       pixel_size=0.139e-6,
                                       kernel="helmholtz")

    roi_1 = rf_1.parse_roi(roi=roi)
    assert roi_1 == slice_roi

    field_1 = field[slice_roi]
    field_2 = field[roi_1]
    field_3 = field[roi[0]: roi[2], roi[1]: roi[3]]

    assert np.array_equal(field_1, field_2)
    assert np.array_equal(field_1, field_3)


def test_2d_autofocus_cell_roi_fail():
    """Give bad roi arguments"""

    field = load_cell("HL60_field.zip")

    rf = nrefocus.iface.RefocusNumpy(field=field,
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz")

    # wrong sequence type
    roi_1 = {10, 100, 10, 100}
    with pytest.raises(ROIValueError):
        d = rf.autofocus(metric='average gradient',
                         minimizer="lmfit",
                         interval=(-5e-6, 5e-6),
                         roi=roi_1)

    # wrong element type
    roi_2 = ['3', 100, 10, 100]
    with pytest.raises(ROIValueError):
        d = rf.autofocus(metric='average gradient',
                         minimizer="lmfit",
                         interval=(-5e-6, 5e-6),
                         roi=roi_2)

    # wrong length
    roi_3 = [10, 100, 100]
    with pytest.raises(ROIValueError):
        d = rf.autofocus(metric='average gradient',  # noqa: F841
                         minimizer="lmfit",
                         interval=(-5e-6, 5e-6),
                         roi=roi_3)

    # list of lists not allowed
    roi_4 = [[10, 100], [100, 200]]
    with pytest.raises(ROIValueError):
        d = rf.autofocus(metric='average gradient',  # noqa: F841
                         minimizer="lmfit",
                         interval=(-5e-6, 5e-6),
                         roi=roi_4)
