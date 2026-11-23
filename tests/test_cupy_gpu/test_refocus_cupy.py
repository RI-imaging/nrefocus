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

    distances_cupy, focused_cupy = nrefocus.autofocus_stack_cupy(
        stack, nm=1.333, res=8.25, ival=(-4, 4),
        metric="average gradient", minimizer="cupy", padding=False,
        roi=(1, 1, 15, 11), num_streams=2)

    assert len(distances_cupy) == stack.shape[0]
    assert focused_cupy.get().shape == stack.shape

    # check that the autofocus prop works as expected
    for frame, distance in enumerate(distances_cupy):
        reference = nrefocus.RefocusCupy(
            field=stack[frame], wavelength=8.25, pixel_size=1,
            medium_index=1.333, kernel="helmholtz", padding=False
        ).propagate(distance).get()
        assert np.allclose(focused_cupy.get()[frame], reference)

    nrefocus.set_ndarray_backend("numpy")
    # Compare against the regular CPU multiprocessing implementation.
    distances2, focused2 = nrefocus.autofocus_stack(
        stack, nm=1.333, res=8.25, ival=(-4, 4),
        metric="average gradient", minimizer="legacy", padding=False,
        roi=(1, 1, 15, 11), num_cpus=1)

    # convert from list
    distances_cupy = np.asarray([float(distance.get())
                                 for distance in distances_cupy])
    assert np.allclose(distances2, distances_cupy)
    assert np.allclose(focused2, focused_cupy.get())


@skip_if_missing("cupy")
def test_cupy_minimizer_matches_legacy_minimizer(set_ndarray_backend_to_cupy):
    """cupy minimizer is the same as the legacy one."""
    rng = np.random.default_rng(5)
    field = rng.normal(size=(18, 14)) + 1j * rng.normal(size=(18, 14))
    common = dict(nm=1.333, res=8.25, ival=(-4, 4),
                  metric="average gradient", padding=False,
                  roi=(1, 1, 17, 13))
    rf = nrefocus.RefocusCupy(
        field=field, wavelength=common["res"], pixel_size=1,
        medium_index=common["nm"], kernel="helmholtz",
        padding=common["padding"])
    autofocus_args = dict(interval=common["ival"],
                          metric=common["metric"], roi=common["roi"],
                          ret_field=True,
                          minimizer_kwargs={"coarse_acc": 2})
    batched_distance, batched_field = rf.autofocus(
        **autofocus_args, minimizer="cupy")
    legacy_distance, legacy_field = rf.autofocus(
        **autofocus_args, minimizer="legacy")
    assert np.isclose(float(batched_distance.get()),
                      float(legacy_distance.get()), atol=1e-8)
    assert np.allclose(batched_field.get(), legacy_field.get(), atol=1e-8)


@skip_if_missing("cupy")
def test_lmfit_minimizer_with_cupy_matches_numpy(set_ndarray_backend_to_cupy):
    """lmfit can optimize a GPU-evaluated metric through scalar transfers."""
    rng = np.random.default_rng(9)
    field = rng.normal(size=(18, 14)) + 1j * rng.normal(size=(18, 14))
    common = dict(wavelength=8.25, pixel_size=1, medium_index=1.333,
                  kernel="helmholtz", padding=False)
    autofocus_args = dict(interval=(-4, 4), metric="average gradient",
                          roi=(1, 1, 17, 13), minimizer="lmfit",
                          ret_field=True)

    rf_cupy = nrefocus.RefocusCupy(field=field, **common)
    distance_cupy, focused_cupy = rf_cupy.autofocus(**autofocus_args)

    nrefocus.set_ndarray_backend("numpy")
    rf_numpy = nrefocus.RefocusNumpy(field=field, **common)
    distance_numpy, focused_numpy = rf_numpy.autofocus(**autofocus_args)

    assert np.isclose(distance_cupy, distance_numpy, atol=1e-8)
    assert np.allclose(focused_cupy.get(), focused_numpy, atol=1e-8)
