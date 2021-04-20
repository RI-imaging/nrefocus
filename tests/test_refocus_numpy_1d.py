import pathlib

import numpy as np

import nrefocus


data_path = pathlib.Path(__file__).parent / "data"


def test_2d_refocus1():
    pixel_size = 1e-6
    rf = nrefocus.RefocusNumpy1D(field=np.arange(200),
                                 wavelength=3.25*pixel_size,
                                 pixel_size=pixel_size,
                                 medium_index=1.333,
                                 distance=0,
                                 kernel="helmholtz",
                                 padding=False)

    refocused = rf.propagate(distance=42.13*pixel_size)
    reference = np.loadtxt(data_path / "test_1d_refocus1.txt")
    assert np.allclose(np.array(refocused).flatten().view(float), reference)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
