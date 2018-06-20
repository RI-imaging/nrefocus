"""Test 2D autorefocusing"""
import numpy as np

import nrefocus


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
    nfield = nrefocus.autofocus(
        field=rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric="average gradient",
        padding=True,
        ret_d=False,
        ret_grad=False,
        num_cpus=1,
    )
    assert np.allclose(0, np.angle(nfield/rfield), atol=.047)
    assert np.allclose(1, np.abs(nfield/rfield), atol=.081)


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
    nfield = nrefocus.autofocus(
        field=rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric="average gradient",
        padding=False,  # without padding, result must be exact
        ret_d=False,
        ret_grad=False,
        num_cpus=1,
    )
    assert np.allclose(nfield.flatten().view(float),
                       rfield.flatten().view(float))


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
    nfield = nrefocus.autofocus(
        field=rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric="average gradient",
        padding=True,
        ret_d=False,
        ret_grad=False,
        num_cpus=1)
    assert np.allclose(0, np.angle(nfield/rfield), atol=.125)
    assert np.allclose(1, np.abs(nfield/rfield), atol=.147)


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
    nfield = nrefocus.autofocus_stack(
        fieldstack=rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric=metric,
        padding=False,
        same_dist=False,
        ret_ds=False,
        ret_grads=False,
        num_cpus=1,
        copy=True)

    # reconstruction distance is same in above case
    nfield_same = nrefocus.autofocus_stack(
        fieldstack=rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric=metric,
        padding=False,
        same_dist=True,
        ret_ds=False,
        ret_grads=False,
        num_cpus=1,
        copy=True)
    assert np.allclose(nfield.flatten().view(float),
                       nfield_same.flatten().view(float),
                       atol=.000524)


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
    nfield = nrefocus.autofocus_stack(
        fieldstack=1*rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric=metric,
        padding=True,
        same_dist=False,
        ret_ds=False,
        ret_grads=False,
        num_cpus=1,
        copy=True)

    assert np.allclose(np.array(rfield).flatten().view(float),
                       np.array(nfield).flatten().view(float),
                       atol=.013)

    # reconstruction distance is same in above case
    nfield_same = nrefocus.autofocus_stack(
        fieldstack=1*rfield,
        nm=nm,
        res=res,
        ival=(-1.5*d, -0.5*d),
        roi=None,
        metric=metric,
        padding=True,
        same_dist=True,
        ret_ds=False,
        ret_grads=False,
        num_cpus=1,
        copy=True)

    assert np.allclose(nfield, nfield_same)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
