import pytest

import nrefocus
from nrefocus._ndarray_backend import xp, NDArrayBackendWarning

from .helper_methods import skip_if_missing


def test_ndarray_backend_numpy_default():
    """should return numpy"""

    assert xp.is_numpy()
    nrefocus.set_ndarray_backend('numpy')
    assert xp.is_numpy()


def test_ndarray_backend_bad():
    """should raise an ImportError"""
    bad_backend = "funpy"
    match_err_str = (f"The backend '{bad_backend}' is not installed. "
                     f"Either install it or use the default backend: 'numpy'.")

    with pytest.raises(ImportError, match=match_err_str):
        nrefocus.set_ndarray_backend(bad_backend)


@skip_if_missing("cupy")
def test_ndarray_backend_cupy():
    """should return cupy"""
    assert xp.is_numpy()
    nrefocus.set_ndarray_backend('cupy')
    assert xp.is_cupy()


@skip_if_missing("cupy")
def test_ndarray_backend_swap():
    """should return the correct set backend"""
    nrefocus.set_ndarray_backend('cupy')
    assert xp.is_cupy()
    nrefocus.set_ndarray_backend('numpy')
    assert xp.is_numpy()
    nrefocus.set_ndarray_backend('cupy')
    assert xp.is_cupy()


def test_ndarray_backend_expected_numpy():
    """should return cupy"""
    assert nrefocus.RefocusNumpy.backend_expected == "numpy"
    assert nrefocus.RefocusNumpy.backend_incompatible == "cupy"
    assert nrefocus.RefocusPyFFTW.backend_expected == "numpy"
    assert nrefocus.RefocusPyFFTW.backend_incompatible == "cupy"


@skip_if_missing("cupy")
def test_ndarray_backend_expected_cupy():
    """should return cupy"""
    assert nrefocus.RefocusCupy.backend_expected == "cupy"
    assert nrefocus.RefocusCupy.backend_incompatible is None


@skip_if_missing("cupy")
def test_refocus_backend_mismatch(cell_field):
    """Shows how a Refocus and ndarray backend mismatch creates a warning"""
    pixel_size = 1e-6
    wavelength = 8.25 * pixel_size

    # this works but provides a user warning
    wrong_backend = "numpy"
    backend_expected = "cupy"
    rf_interface = nrefocus.RefocusCupy
    nrefocus.set_ndarray_backend(wrong_backend)
    with pytest.warns(
            NDArrayBackendWarning,
            match=rf"You are using `{rf_interface.__name__}` "
                  rf"with the '{wrong_backend}' ndarray backend. This might "
                  rf"limit the Refocussing speed. To set the correct "
                  rf"ndarray backend, use "
                  rf"`nrefocus.set_ndarray_backend\('{backend_expected}'\)`"
    ):
        _ = rf_interface(cell_field, wavelength, pixel_size)

    # this fails because currently padding mask
    backend_expected = "numpy"
    wrong_backend = "cupy"
    rf_interface = nrefocus.RefocusNumpy
    nrefocus.set_ndarray_backend(wrong_backend)
    with pytest.raises(
            NDArrayBackendWarning,
            match=rf"You cannot use the '{wrong_backend}' "
                  rf"ndarray backend with `{rf_interface.__name__}`. "
                  rf"To set the correct ndarray backend, use "
                  rf"`nrefocus.set_ndarray_backend\('{backend_expected}'\)`"
    ):
        _ = rf_interface(cell_field, wavelength, pixel_size)

    # this fails because pyfftw arrays don't work with cupy
    wrong_backend = "cupy"
    rf_interface = nrefocus.RefocusPyFFTW
    nrefocus.set_ndarray_backend(wrong_backend)
    with pytest.raises(
            NDArrayBackendWarning,
            match=rf"You cannot use the '{wrong_backend}' "
                  rf"ndarray backend with `{rf_interface.__name__}`. "
                  rf"To set the correct ndarray backend, use "
                  rf"`nrefocus.set_ndarray_backend\('{backend_expected}'\)`"
    ):
        _ = rf_interface(cell_field, wavelength, pixel_size)
