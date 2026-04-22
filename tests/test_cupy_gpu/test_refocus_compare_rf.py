"""Compare outputes of cupy and numpy runs"""
import pathlib
import numpy as np

import nrefocus
from nrefocus._ndarray_backend import xp

from ..helper_methods import skip_if_missing

data_path = pathlib.Path(__file__).parent.parent / "data"


@skip_if_missing("cupy")
def test_2d_refocus1():
    pixel_size = 1e-6

    # run with numpy
    xp.assert_numpy()
    rf_np = nrefocus.RefocusNumpy(field=np.arange(256).reshape(16, 16),
                                  wavelength=8.25 * pixel_size,
                                  pixel_size=pixel_size,
                                  medium_index=1.533,
                                  distance=0,
                                  kernel="helmholtz",
                                  padding=False)
    refocused_np = rf_np.propagate(distance=2.13 * pixel_size)

    # run with pyfftw
    xp.assert_numpy()
    rf_fftw = nrefocus.RefocusPyFFTW(field=np.arange(256).reshape(16, 16),
                                     wavelength=8.25 * pixel_size,
                                     pixel_size=pixel_size,
                                     medium_index=1.533,
                                     distance=0,
                                     kernel="helmholtz",
                                     padding=False)
    refocused_fftw = rf_fftw.propagate(distance=2.13 * pixel_size)

    # run with cupy
    nrefocus.set_ndarray_backend("cupy")
    rf_cp = nrefocus.RefocusCupy(field=np.arange(256).reshape(16, 16),
                                 wavelength=8.25 * pixel_size,
                                 pixel_size=pixel_size,
                                 medium_index=1.533,
                                 distance=0,
                                 kernel="helmholtz",
                                 padding=False)
    refocused_cp = rf_cp.propagate(distance=2.13 * pixel_size)
    refocused_cp_cpu = refocused_cp.get()

    assert np.allclose(np.array(refocused_np).flatten().view(float),
                       np.array(refocused_cp_cpu).flatten().view(float))
    assert np.allclose(np.array(refocused_fftw).flatten().view(float),
                       np.array(refocused_cp_cpu).flatten().view(float))
