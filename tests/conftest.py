import os.path as op
import zipfile
import numpy as np
import pytest

import nrefocus


@pytest.fixture(autouse=True)
def set_ndarray_backend_to_numpy():
    """Ensures that the backend is set to numpy as the end of each test."""
    yield
    # always reset to numpy, even if test fails
    nrefocus.set_ndarray_backend('numpy')


@pytest.fixture()
def set_ndarray_backend_to_cupy():
    """Ensures that the backend is set to cupy when desired."""
    nrefocus.set_ndarray_backend('cupy')
    yield
    # always reset to numpy, even if test fails
    nrefocus.set_ndarray_backend('numpy')


@pytest.fixture()
def cell_field(fname="HL60_field.zip"):
    """Load zip file and return complex field"""
    here = op.dirname(op.abspath(__file__))
    data = op.join(here, "data")
    arc = zipfile.ZipFile(op.join(data, fname))
    for f in arc.filelist:
        with arc.open(f) as fd:
            if f.filename.count("imag"):
                imag = np.loadtxt(fd)

            elif f.filename.count("real"):
                real = np.loadtxt(fd)

    field = real + 1j * imag
    return field
