import pathlib

import numpy as np

import nrefocus


data_path = pathlib.Path(__file__).parent / "data"


def test_2d_refocus1():
    pixel_size = 1e-6
    rf = nrefocus.RefocusPyFFTW(field=np.arange(256).reshape(16, 16),
                                wavelength=8.25*pixel_size,
                                pixel_size=pixel_size,
                                medium_index=1.533,
                                distance=0,
                                kernel="helmholtz",
                                padding=False)

    refocused = rf.propagate(distance=2.13*pixel_size)
    reference = np.loadtxt(data_path / "test_2d_refocus1.txt")
    assert np.allclose(np.array(refocused).flatten().view(float), reference)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
