import pathlib

import numpy as np

import nrefocus

from ..helper_methods import skip_if_missing

data_path = pathlib.Path(__file__).parent.parent / "data"


@skip_if_missing("cupy")
def test_2d_refocus1(set_ndarray_backend_to_cupy):
    pixel_size = 1e-6
    rf = nrefocus.RefocusCupy(field=np.arange(256).reshape(16, 16),
                              wavelength=8.25 * pixel_size,
                              pixel_size=pixel_size,
                              medium_index=1.533,
                              distance=0,
                              kernel="helmholtz",
                              padding=False)

    refocused = rf.propagate(distance=2.13 * pixel_size)
    refocused_cpu = refocused.get()
    reference = np.loadtxt(data_path / "test_2d_refocus1.txt")
    assert np.allclose(np.array(refocused_cpu).flatten().view(float),
                       reference)
