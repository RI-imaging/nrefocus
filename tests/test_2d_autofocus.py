#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests 2D refocusing
"""
from __future__ import division, print_function

import multiprocessing as mp
import numpy as np
import os
from os.path import abspath, dirname, join, split
import sys



# Add parent directory to beginning of path variable
DIR = dirname(abspath(__file__))
sys.path = [split(DIR)[0]] + sys.path

import nrefocus


def test_2d_autofocus_helmholtz_average_gradient():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    field=1*np.exp(1j*np.linspace(.1,.5, 256)).reshape(16,16)
    d = 5
    nm = 1.533
    res = 8.25
    method = "helmholtz"
    
    # first propagate the field
    rfield = nrefocus.refocus(field=field,
                              d=d,
                              nm = nm,
                              res = res,
                              method = method)
    # then try to refocus it
    nfield, dnew = nrefocus.autofocus(
            field=rfield,
            nm=nm,
            res=res,
            ival=(-1.5*d, -0.5*d),
            roi=None,
            metric="average gradient",
            padding=True,
            ret_d=True,
            ret_grad=False,
            num_cpus=1,
            )
    print("  correct / expected / refocused distances:", -1*d, -3.25757575758, dnew)
    assert np.allclose(0, np.angle(nfield/rfield), atol=.047)
    assert np.allclose(1, np.abs(nfield/rfield), atol=.081)


def test_2d_autofocus_helmholtz_average_gradient_zero():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    field=1*np.exp(1j*np.linspace(.1,.5, 256)).reshape(16,16)
    d = 0
    nm = 1.533
    res = 8.25
    method = "helmholtz"
    
    # first propagate the field
    rfield = nrefocus.refocus(field = field,
                              d = d,
                              nm=nm,
                              res=res,
                              method=method,
                              padding=False
                              )
    # then try to refocus it
    nfield, dnew = nrefocus.autofocus(
            field=rfield,
            nm=nm,
            res=res,
            ival=(-1.5*d, -0.5*d),
            roi=None,
            metric="average gradient",
            padding=False, # without padding, result must be exact
            ret_d=True,
            ret_grad=False,
            num_cpus=1,
            )
    print("  correct / expected / refocused distances:", 0, 0, dnew)
    assert np.allclose( nfield.flatten().view(float),
                        rfield.flatten().view(float))


def test_2d_autofocus_fresnel_average_gradient():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    field=1*np.exp(1j*np.linspace(.1,.5, 256)).reshape(16,16)
    d = 5
    nm = 1.533
    res = 8.25
    method = "fresnel"
    
    # first propagate the field
    rfield = nrefocus.refocus(field = field,
                              d = d,
                              nm = nm,
                              res = res,
                              method = method)
    # then try to refocus it
    nfield, dnew = nrefocus.autofocus(
            field=rfield,
            nm=nm,
            res=res,
            ival=(-1.5*d, -0.5*d),
            roi=None,
            metric="average gradient",
            padding=True,
            ret_d=True,
            ret_grad=False,
            num_cpus=1)
    print("  correct / expected / refocused distances:", -1*d, -5.68181818182, dnew)
    assert np.allclose(0, np.angle(nfield/rfield), atol=.125)
    assert np.allclose(1, np.abs(nfield/rfield), atol=.147)


def test_2d_autofocus_stack_same_dist_nopadding():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    d = 5.5
    nm = 1.5133
    res = 6.25
    method = "helmholtz"
    size = 10
    metric = "average gradient"
    stack = 1*np.exp(1j*np.linspace(.1,.5, size**3)).reshape(size,size,size)
    rfield = nrefocus.refocus_stack(fieldstack=stack,
                                    d=d,
                                    nm=nm,
                                    res=res,
                                    method=method)
    nfield, dnew = nrefocus.autofocus_stack(
            fieldstack=rfield,
            nm=nm,
            res=res,
            ival=(-1.5*d, -0.5*d),
            roi=None,
            metric=metric,
            padding=False,
            same_dist=False,
            ret_ds=True,
            ret_grads=False,
            num_cpus=1,
            copy=True)

    # reconstruction distance is same in above case
    nfield_same, dnewsame = nrefocus.autofocus_stack(
            fieldstack=rfield,
            nm=nm,
            res=res,
            ival=(-1.5*d, -0.5*d),
            roi=None,
            metric=metric,
            padding=False,
            same_dist=True,
            ret_ds=True,
            ret_grads=False,
            num_cpus=1,
            copy=True)
    assert np.allclose(nfield.flatten().view(float),
                       nfield_same.flatten().view(float),
                       atol=.000524)


def test_2d_autofocus_stack_same_dist():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    d = 5.5
    nm = 1.5133
    res = 6.25
    method = "helmholtz"
    size = 10
    metric = "average gradient"
    stack = 1*np.exp(1j*np.linspace(.1,.5, size**3)).reshape(size,size,size)
    rfield = nrefocus.refocus_stack(fieldstack=stack,
                                    d=d,
                                    nm=nm,
                                    res=res,
                                    method=method,
                                    padding=True)
    nfield, dnew = nrefocus.autofocus_stack(
            fieldstack=1*rfield,
            nm=nm,
            res=res,
            ival=(-1.5*d, -0.5*d),
            roi=None,
            metric=metric,
            padding=True,
            same_dist=False,
            ret_ds=True,
            ret_grads=False,
            num_cpus=1,
            copy=True)

    assert np.allclose(np.array(rfield).flatten().view(float),
                       np.array(nfield).flatten().view(float),
                       atol=.013)

    # reconstruction distance is same in above case
    nfield_same, ds2 = nrefocus.autofocus_stack(
            fieldstack=1*rfield,
            nm=nm,
            res=res,
            ival=(-1.5*d, -0.5*d),
            roi=None,
            metric=metric,
            padding=True,
            same_dist=True,
            ret_ds=True,
            ret_grads=False,
            num_cpus=1,
            copy=True)

    assert np.allclose(nfield, nfield_same)

    
# Get results
results = dict()
datadir = join(DIR, "data")
for f in os.listdir(datadir):
    #np.savetxt('outfile.txt', np.array(r).flatten().view(float))
    #np.savetxt('outfile.txt', np.array(r).flatten().view(float), fmt="%.9f")
    glob = globals()
    if f.endswith(".txt") and f[:-4] in list(glob.keys()):
        results[f[:-4]] = np.loadtxt(join(datadir, f))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
