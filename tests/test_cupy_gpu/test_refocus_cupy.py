import pathlib

import numpy as np

import nrefocus

from ..helper_methods import skip_if_missing

data_path = pathlib.Path(__file__).parent.parent / "data"


@skip_if_missing("cupy")
def test_2d_refocus1_cupy(set_ndarray_backend_to_cupy):
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


@skip_if_missing("cupy")
def test_refocus_cupy_stack_matches_per_slice(set_ndarray_backend_to_cupy):
    rng = np.random.default_rng(0)
    stack = (rng.normal(size=(5, 16, 12)) + 1j * rng.normal(size=(5, 16, 12))
             ).astype(np.complex128)

    pixel_size = 1e-6
    dist = 2.13 * pixel_size

    rf_stack = nrefocus.RefocusCupy(
        field=stack,
        wavelength=8.25 * pixel_size,
        pixel_size=pixel_size,
        medium_index=1.533,
        distance=0,
        kernel="helmholtz",
        padding=True,
    )
    out_stack = rf_stack.propagate(distance=dist).get()

    out_expected = np.empty_like(stack)
    for i in range(stack.shape[0]):
        rf = nrefocus.RefocusCupy(
            field=stack[i],
            wavelength=8.25 * pixel_size,
            pixel_size=pixel_size,
            medium_index=1.533,
            distance=0,
            kernel="helmholtz",
            padding=True,
        )
        out_expected[i] = rf.propagate(distance=dist).get()

    assert out_stack.shape == stack.shape
    assert np.allclose(out_stack, out_expected, rtol=1e-12, atol=1e-12)


@skip_if_missing("cupy")
def test_autofocus_cupy_stack_per_frame(set_ndarray_backend_to_cupy):
    """Autofocus each 2D frame on GPU without multiprocessing."""
    rng = np.random.default_rng(0)
    plane = rng.normal(size=(16, 12)) + 1j * rng.normal(size=(16, 12))
    stack = np.stack([plane * np.exp(1j * phase)
                      for phase in (0.0, 0.2, 0.4)])

    distances, focused = nrefocus.autofocus_stack_cupy(
        stack, nm=1.333, res=8.25, ival=(-4, 4),
        metric="average gradient", padding=False,
        roi=(1, 1, 15, 11), num_streams=2)

    focused_cpu = focused.get()
    assert len(distances) == stack.shape[0]
    assert focused_cpu.shape == stack.shape
    assert np.all(np.isfinite([float(d.get()) for d in distances]))
    # check that the autofocus prop works as expected
    for frame, distance in enumerate(distances):
        reference = nrefocus.RefocusCupy(
            field=stack[frame], wavelength=8.25, pixel_size=1,
            medium_index=1.333, kernel="helmholtz", padding=False
        ).propagate(distance).get()
        assert np.allclose(focused_cpu[frame], reference)

    # Convert GPU results before switching the active backend for the CPU run.
    distances_gpu = np.asarray([float(distance.get())
                                for distance in distances])
    focused_gpu = focused.get()
    nrefocus.set_ndarray_backend("numpy")

    # Compare against the regular CPU multiprocessing implementation.
    distances2, focused2 = nrefocus.autofocus_stack(
        stack, nm=1.333, res=8.25, ival=(-4, 4),
        metric="average gradient", minimizer="legacy", padding=False,
        roi=(1, 1, 15, 11), num_cpus=1)
    assert np.allclose(distances2, distances_gpu)
    assert np.allclose(focused2, focused_gpu)
