"""Test 2D autorefocusing of legacy minimizer"""
import numpy as np

import nrefocus
import pytest

from test_helper import load_cell


@pytest.mark.filterwarnings('ignore::nrefocus.minimizers.mz_legacy.'
                            'LegacyDeprecationWarning')
def test_2d_autofocus_cell_helmholtz_average_gradient():
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )

    # attempt to autofocus with standard arguments
    d = rf.autofocus(metric="average gradient",
                     minimizer="legacy",
                     interval=(-5e-6, 5e-6))
    assert np.allclose(d, -8.69809203142537e-07, atol=0)

    nfield = rf.propagate(d)
    assert np.allclose(nfield[10, 10],
                       1.045874817165857-0.020467790949516538j,
                       atol=0)


@pytest.mark.filterwarnings('ignore::nrefocus.minimizers.mz_legacy.'
                            'LegacyDeprecationWarning')
def test_2d_autofocus_helmholtz_average_gradient():
    field = 1*np.exp(1j*np.linspace(.1, .5, 256)).reshape(16, 16)
    d = 5
    nm = 1.533
    res = 8.25
    method = "helmholtz"

    # first propagate the field
    rfield = nrefocus.refocus(field=field,
                              d=d,
                              nm=nm,
                              res=res,
                              method=method)
    # then try to refocus it
    d, nfield = nrefocus.autofocus(
        field=rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric="average gradient",
        minimizer="legacy",
        padding=True,
        num_cpus=1,
    )
    assert np.allclose(d, -3.263187429854096)
    assert np.allclose(0, np.angle(nfield/rfield), atol=.047)
    assert np.allclose(1, np.abs(nfield/rfield), atol=.081)


@pytest.mark.filterwarnings('ignore::nrefocus.minimizers.mz_legacy.'
                            'LegacyDeprecationWarning')
def test_2d_autofocus_helmholtz_average_gradient_zero():
    field = 1*np.exp(1j*np.linspace(.1, .5, 256)).reshape(16, 16)
    d = 0
    nm = 1.533
    res = 8.25
    method = "helmholtz"

    # first propagate the field
    rfield = nrefocus.refocus(field=field,
                              d=d,
                              nm=nm,
                              res=res,
                              method=method,
                              padding=False
                              )
    # then try to refocus it
    _, nfield = nrefocus.autofocus(
        field=rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric="average gradient",
        minimizer="legacy",
        padding=False,  # without padding, result must be exact
        num_cpus=1,
    )
    assert np.allclose(nfield.flatten().view(float),
                       rfield.flatten().view(float))


@pytest.mark.filterwarnings('ignore::nrefocus.minimizers.mz_legacy.'
                            'LegacyDeprecationWarning')
def test_2d_autofocus_fresnel_average_gradient():
    field = 1*np.exp(1j*np.linspace(.1, .5, 256)).reshape(16, 16)
    d = 5
    nm = 1.533
    res = 8.25
    method = "fresnel"

    # first propagate the field
    rfield = nrefocus.refocus(field=field,
                              d=d,
                              nm=nm,
                              res=res,
                              method=method)
    # then try to refocus it
    _, nfield = nrefocus.autofocus(
        field=rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric="average gradient",
        minimizer="legacy",
        padding=True,
        num_cpus=1)
    assert np.allclose(0, np.angle(nfield/rfield), atol=.125)
    assert np.allclose(1, np.abs(nfield/rfield), atol=.147)


@pytest.mark.filterwarnings('ignore::nrefocus.minimizers.mz_legacy.'
                            'LegacyDeprecationWarning')
def test_2d_autofocus_return_grid_field():
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )

    # attempt to autofocus with standard arguments
    d, (dgrid, mgrid), nfield = rf.autofocus(
        metric="average gradient",
        minimizer="legacy",
        interval=(-5e-6, 5e-6),
        ret_grid=True,
        ret_field=True,
        )

    idx_metric_min = np.argmin(mgrid)
    idx_distance = np.argmin(np.abs(dgrid - d))
    assert idx_metric_min == idx_distance


@pytest.mark.filterwarnings('ignore::nrefocus.minimizers.mz_legacy.'
                            'LegacyDeprecationWarning')
def test_2d_autofocus_stack_same_dist_nopadding():
    d = 5.5
    nm = 1.5133
    res = 6.25
    method = "helmholtz"
    size = 10
    metric = "average gradient"
    stack = 1*np.exp(1j*np.linspace(.1, .5, size**3)).reshape(size, size, size)
    rfield = nrefocus.refocus_stack(fieldstack=stack,
                                    d=d,
                                    nm=nm,
                                    res=res,
                                    method=method)
    ds, nfield = nrefocus.autofocus_stack(
        fieldstack=rfield.copy(),
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric=metric,
        minimizer="legacy",
        padding=False,
        same_dist=False,
        num_cpus=1,
        copy=True)

    # reconstruction distance is same in above case
    ds_same, nfield_same = nrefocus.autofocus_stack(
        fieldstack=rfield.copy(),
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric=metric,
        minimizer="legacy",
        padding=False,
        same_dist=True,
        num_cpus=1,
        copy=True)
    assert np.allclose(np.mean(ds), -4.867283950617284)
    assert np.all(np.array(ds) == ds_same)
    assert np.all(np.array(ds) == np.mean(ds))
    assert np.allclose(nfield.flatten().view(float),
                       nfield_same.flatten().view(float),
                       atol=.000524)


@pytest.mark.filterwarnings('ignore::nrefocus.minimizers.mz_legacy.'
                            'LegacyDeprecationWarning')
def test_2d_autofocus_stack_same_dist():
    d = 5.5
    nm = 1.5133
    res = 6.25
    method = "helmholtz"
    size = 10
    metric = "average gradient"
    stack = 1*np.exp(1j*np.linspace(.1, .5, size**3)).reshape(size, size, size)
    rfield = nrefocus.refocus_stack(fieldstack=stack,
                                    d=d,
                                    nm=nm,
                                    res=res,
                                    method=method,
                                    padding=True)
    ds, nfield = nrefocus.autofocus_stack(
        fieldstack=1*rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric=metric,
        minimizer="legacy",
        padding=True,
        same_dist=False,
        num_cpus=1,
        copy=True)

    assert np.allclose(np.array(rfield).flatten().view(float),
                       np.array(nfield).flatten().view(float),
                       atol=.013)

    # reconstruction distance is same in above case
    ds_same, nfield_same = nrefocus.autofocus_stack(
        fieldstack=1*rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric=metric,
        minimizer="legacy",
        padding=True,
        same_dist=True,
        num_cpus=1,
        copy=True)

    assert np.allclose(nfield[0][8][8],
                       0.9900406072155992+0.1341183159587472j)
    assert np.allclose(nfield[0][2][8],
                       0.9947454248517085+0.11020637810883656j)
    assert np.allclose(np.mean(ds), -4.8240740740740735)
    assert np.allclose(np.array(ds), np.mean(ds))
    assert np.allclose(np.array(ds), ds_same)

    assert np.allclose(nfield, nfield_same)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
