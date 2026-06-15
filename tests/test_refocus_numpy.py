import pathlib

import numpy as np

import nrefocus

data_path = pathlib.Path(__file__).parent / "data"


def test_2d_refocus1():
    pixel_size = 1e-6
    rf = nrefocus.RefocusNumpy(field=np.arange(256).reshape(16, 16),
                               wavelength=8.25 * pixel_size,
                               pixel_size=pixel_size,
                               medium_index=1.533,
                               distance=0,
                               kernel="helmholtz",
                               padding=False)

    refocused = rf.propagate(distance=2.13 * pixel_size)
    reference = np.loadtxt(data_path / "test_2d_refocus1.txt")
    assert np.allclose(np.array(refocused).flatten().view(float), reference)


def test_refocus_numpy_stack_matches_per_slice():
    rng = np.random.default_rng(0)
    stack = (rng.normal(size=(5, 16, 12)) + 1j * rng.normal(size=(5, 16, 12))
             ).astype(np.complex128)

    pixel_size = 1e-6
    dist = 2.13 * pixel_size

    rf_stack = nrefocus.RefocusNumpy(
        field=stack,
        wavelength=8.25 * pixel_size,
        pixel_size=pixel_size,
        medium_index=1.533,
        distance=0,
        kernel="helmholtz",
        padding=True,
    )
    out_stack = rf_stack.propagate(distance=dist)

    out_expected = np.empty_like(stack)
    for i in range(stack.shape[0]):
        rf = nrefocus.RefocusNumpy(
            field=stack[i],
            wavelength=8.25 * pixel_size,
            pixel_size=pixel_size,
            medium_index=1.533,
            distance=0,
            kernel="helmholtz",
            padding=True,
        )
        out_expected[i] = rf.propagate(distance=dist)

    assert out_stack.shape == stack.shape
    assert np.allclose(out_stack, out_expected, rtol=1e-12, atol=1e-12)


def test_refocus_numpy_nonsquare_nopadding():
    rng = np.random.default_rng(0)
    stack = (rng.normal(size=(5, 16, 12)) + 1j * rng.normal(size=(5, 16, 12))
             ).astype(np.complex128)
    pixel_size = 1e-6
    dist = 2.13 * pixel_size

    rf_stack = nrefocus.RefocusNumpy(
        field=stack,
        wavelength=8.25 * pixel_size,
        pixel_size=pixel_size,
        medium_index=1.533,
        distance=0,
        kernel="helmholtz",
        padding=False,
    )
    out_stack = rf_stack.propagate(distance=dist)

    assert out_stack.shape == stack.shape


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
