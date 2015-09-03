#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests 1D refocusing
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


def test_1d_refocus1():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    rfield = nrefocus.refocus(field=np.arange(200),
                              d = 42.13,
                              nm = 1.333,
                              res = 3.25,
                              method = "helmholtz",
                              padding = False)
    assert np.allclose(np.array(rfield).flatten().view(float), results[myname])
    
    
# Get results
results = dict()
datadir = join(DIR, "data")
for f in os.listdir(datadir):
    #np.savetxt('outfile.txt', np.array(r).flatten().view(float))
    glob = globals()
    if f.endswith(".txt") and f[:-4] in list(glob.keys()):
        results[f[:-4]] = np.loadtxt(join(datadir, f))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
