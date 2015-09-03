#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests 2D refocusing
"""
from __future__ import division, print_function

import numpy as np
import os
from os.path import abspath, dirname, join, split
import sys



# Add parent directory to beginning of path variable
DIR = dirname(abspath(__file__))
sys.path = [split(DIR)[0]] + sys.path

import nrefocus


def test_2d_refocus1():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    rfield = nrefocus.refocus(field=np.arange(256).reshape(16,16),
                              d = 2.13,
                              nm = 1.533,
                              res = 8.25,
                              method = "helmholtz",
                              padding=False)
    assert np.allclose(np.array(rfield).flatten().view(float), results[myname])


def test_2d_refocus_stack():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    size = 10
    stack = np.arange(size**3).reshape(size, size, size)
    rfield = nrefocus.refocus_stack(fieldstack=stack,
                                    d = 2.13,
                                    nm = 1.533,
                                    res = 8.25,
                                    method = "helmholtz",
                                    padding = False)
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
    
