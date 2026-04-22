"""Test the two available propagation kernels 'Helmholtz' and 'Fresnel'."""

import pathlib
import numpy as np

import pytest

import nrefocus

from .helper_methods import skip_if_missing

data_path = pathlib.Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "kernel, reference_file",
    [
        ("helmholtz", data_path / "test_2d_refocus1.txt"),
        ("fresnel", data_path / "test_2d_refocus1_fresnel.txt"),
    ]
)
def test_prop_kernel(kernel, reference_file):
    pixel_size = 1e-6
    distance = 2.13 * pixel_size

    rf = nrefocus.RefocusNumpy(field=np.arange(256).reshape(16, 16),
                               wavelength=8.25 * pixel_size,
                               pixel_size=pixel_size,
                               medium_index=1.533,
                               distance=0,
                               kernel=kernel,
                               padding=False)
    # this codeblock is just `rf.propagate(distance=2.13*pixel_size)`
    fft_kernel = rf.get_kernel(distance=distance)
    refocused = np.fft.ifft2(rf.fft_origin * fft_kernel)

    reference = np.loadtxt(reference_file)
    assert np.allclose(np.array(refocused).flatten().view(float), reference)


@skip_if_missing("pyfftw")
@pytest.mark.parametrize(
    "kernel", [("helmholtz"), ("fresnel"), ]
)
def test_prop_kernel_refocus_interfaces(kernel):
    """compare refocus interface outputs (non-cupy) for kernels"""
    pixel_size = 1e-6
    distance = 2.13 * pixel_size
    field = np.arange(256).reshape(16, 16)
    refocus_iface = [nrefocus.RefocusNumpy, nrefocus.RefocusPyFFTW]

    fft_kernels = []
    for iface in refocus_iface:
        rf = iface(field, 8.25 * pixel_size, pixel_size, medium_index=1.533,
                   distance=0, kernel=kernel, padding=False)
        fft_kernels.append(rf.get_kernel(distance=distance))

    assert np.allclose(fft_kernels[0], fft_kernels[1])


@skip_if_missing("cupy")
@pytest.mark.parametrize(
    "kernel", [("helmholtz"), ("fresnel"), ]
)
def test_prop_kernel_backends(kernel):
    """compare cupy and numpy backend outputs for kernels"""
    pixel_size = 1e-6
    distance = 2.13 * pixel_size

    nrefocus.set_ndarray_backend("numpy")
    xp = nrefocus.get_ndarray_backend()
    rf_np = nrefocus.RefocusNumpy(
        xp.arange(256).reshape(16, 16), 8.25 * pixel_size, pixel_size,
        medium_index=1.533, distance=0, kernel=kernel, padding=False)
    fft_kernel_np = rf_np.get_kernel(distance=distance)

    nrefocus.set_ndarray_backend("cupy")
    xp = nrefocus.get_ndarray_backend()
    rf_cp = nrefocus.RefocusCupy(
        xp.arange(256).reshape(16, 16), 8.25 * pixel_size, pixel_size,
        medium_index=1.533, distance=0, kernel=kernel, padding=False)
    fft_kernel_cp = rf_cp.get_kernel(distance=distance)

    assert np.allclose(fft_kernel_np, fft_kernel_cp)


def test_prop_kernel_bad():
    pixel_size = 1e-6
    distance = 2.13 * pixel_size
    kernel = "laplandian"

    rf = nrefocus.RefocusNumpy(field=np.arange(256).reshape(16, 16),
                               wavelength=8.25 * pixel_size,
                               pixel_size=pixel_size,
                               medium_index=1.533,
                               distance=0,
                               kernel=kernel,
                               padding=False)
    with pytest.raises(KeyError):
        # at this point the kernel name is checked
        _ = rf.get_kernel(distance=distance)
