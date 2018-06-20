"""Test 2D refocusing"""
import pathlib
import sys

import numpy as np

import nrefocus


def test_2d_refocus1():
    myname = sys._getframe().f_code.co_name
    rfield = nrefocus.refocus(field=np.arange(256).reshape(16, 16),
                              d=2.13,
                              nm=1.533,
                              res=8.25,
                              method="helmholtz",
                              padding=False)
    assert np.allclose(np.array(rfield).flatten().view(float), results[myname])


def test_2d_refocus_stack():
    myname = sys._getframe().f_code.co_name
    size = 10
    stack = np.arange(size**3).reshape(size, size, size)
    rfield = nrefocus.refocus_stack(fieldstack=stack,
                                    d=2.13,
                                    nm=1.533,
                                    res=8.25,
                                    method="helmholtz",
                                    padding=False)
    assert np.allclose(np.array(rfield).flatten().view(float), results[myname])


# Get results
results = {}
datadir = pathlib.Path(__file__).parent / "data"
for ff in datadir.glob("*.txt"):
    # np.savetxt('outfile.txt', np.array(r).flatten().view(float))
    # np.savetxt('outfile.txt', np.array(r).flatten().view(float), fmt="%.9f")
    glob = globals()
    if ff.name[:-4] in list(glob.keys()):
        results[ff.name[:-4]] = np.loadtxt(str(ff))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
