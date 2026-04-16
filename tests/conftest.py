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
