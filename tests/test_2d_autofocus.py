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
            ret_d=True,
            ret_grad=False,
            num_cpus=1)
    print("correct / expected / refocused distances:", -1*d, -3.08922558923, dnew)
    assert np.allclose(0, np.angle(nfield/rfield), atol=.047)
    assert np.allclose(1, np.abs(nfield/rfield), atol=.081)

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
            ret_d=True,
            ret_grad=False,
            num_cpus=1)
    print("correct / expected / refocused distances:", -1*d, -7.56172839506, dnew)
    assert np.allclose(0, np.angle(nfield/rfield), atol=.124)
    assert np.allclose(1, np.abs(nfield/rfield), atol=.15)






def test_2d_refocus_stack():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    size = 10
    stack = np.arange(size**3).reshape(size, size, size)
    rfield = nrefocus.refocus_stack(fieldstack=stack,
                                    d = 2.13,
                                    nm = 1.533,
                                    res = 8.25,
                                    method = "helmholtz")
    assert np.allclose(np.array(rfield).flatten().view(float), results[myname])
    
    
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
    
