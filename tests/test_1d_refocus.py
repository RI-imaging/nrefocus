"""Test 1D refocusing"""
import pathlib
import sys

import numpy as np

import nrefocus


def test_1d_refocus1():
    myname = sys._getframe().f_code.co_name
    rfield = nrefocus.refocus(field=np.arange(200),
                              d=42.13,
                              nm=1.333,
                              res=3.25,
                              method="helmholtz",
                              padding=False)
    assert np.allclose(np.array(rfield).flatten().view(float), results[myname])


# Get results
results = {}
datadir = pathlib.Path(__file__).parent / "data"
for ff in datadir.glob("*.txt"):
    # np.savetxt('outfile.txt', np.array(r).flatten().view(float))
    glob = globals()
    if ff.name[:-4] in list(glob.keys()):
        results[ff.name[:-4]] = np.loadtxt(str(ff))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
