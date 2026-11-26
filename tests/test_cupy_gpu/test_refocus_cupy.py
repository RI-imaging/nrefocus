import pathlib

import numpy as np
import pytest

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
        metric="average gradient", minimizer="legacy_cupy", padding=False,
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
def test_legacy_cupy_minimizer_matches_legacy_minimizer(
        set_ndarray_backend_to_cupy):
    """The batched legacy CuPy minimizer matches the NumPy legacy result."""
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
        **autofocus_args, minimizer="legacy_cupy")
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


@skip_if_missing("cupy")
def test_lmfit_cupy_finds_quadratic_minimum_and_respects_bounds(
        set_ndarray_backend_to_cupy):
    """The CuPy LM update converges on a known one-dimensional objective."""
    import cupy as cp

    class QuadraticRefocus:
        wavelength = 1.0

        @staticmethod
        def propagate(distance):
            distance = cp.asarray(distance).reshape(-1, 1, 1)
            y = cp.arange(8, dtype=cp.float64).reshape(1, 8, 1)
            x = cp.arange(8, dtype=cp.float64).reshape(1, 1, 8)
            amplitude = cp.sqrt((distance - 0.37) ** 2 + 0.01)
            return amplitude * (x + 2 * y)

    rf = QuadraticRefocus()
    distance = nrefocus.minimizers.minimize_lmfit_cupy(
        rf, nrefocus.metrics.metric_average_gradient, (-1, 1),
        coarse_step=0.25)
    assert np.isclose(float(distance.get()), 0.37, atol=1e-6)
    bounded_distance = nrefocus.minimizers.minimize_lmfit_cupy(
        rf, nrefocus.metrics.metric_average_gradient, (0, 0.2),
        coarse_step=0.05)
    assert np.isclose(float(bounded_distance.get()), 0.2, atol=1e-6)


@skip_if_missing("cupy")
def test_lmfit_cupy_agrees_with_lmfit(set_ndarray_backend_to_cupy,
                                      cell_field):
    """GPU-native fitting agrees with the lmfit reference within tolerance."""
    wavelength = 647e-9
    rf = nrefocus.RefocusCupy(
        field=cell_field, wavelength=wavelength, pixel_size=0.139e-6,
        kernel="helmholtz")
    kwargs = dict(interval=(-5e-6, 5e-6), metric="average gradient")
    distance_gpu, (grid, values), focused = rf.autofocus(
        **kwargs, minimizer="lmfit_cupy", ret_grid=True, ret_field=True)
    distance_lmfit = rf.autofocus(**kwargs, minimizer="lmfit")

    normalized_gpu = float(distance_gpu.get()) / wavelength
    normalized_lmfit = distance_lmfit / wavelength
    assert np.isclose(normalized_gpu, normalized_lmfit,
                      rtol=1e-4, atol=1e-7)
    assert grid.ndim == values.ndim == 1
    assert grid.shape == values.shape
    assert np.allclose(focused.get(), rf.propagate(distance_gpu).get())


@skip_if_missing("cupy")
@pytest.mark.parametrize("metric", nrefocus.metrics.METRICS)
def test_cupy_metrics_evaluate_batched_distances(set_ndarray_backend_to_cupy,
                                                  metric):
    """Each registered metric returns one score per propagated plane."""
    import cupy as cp

    rng = np.random.default_rng(11)
    field = rng.normal(size=(18, 14)) + 1j * rng.normal(size=(18, 14))
    rf = nrefocus.RefocusCupy(
        field=field, wavelength=8.25, pixel_size=1, medium_index=1.333,
        kernel="helmholtz", padding=False)
    roi = None if metric == "spectrum" else (slice(1, 17), slice(1, 13))
    metric_func = nrefocus.metrics.METRICS[metric]
    distances = cp.asarray([-2.0, 0.0, 2.0])

    scores = metric_func(rf, distances, roi=roi)
    scalar_scores = np.asarray([
        float(metric_func(rf, distance, roi=roi).get())
        for distance in (-2.0, 0.0, 2.0)
    ])
    assert scores.shape == (len(distances),)
    assert np.allclose(scores.get(), scalar_scores)


@skip_if_missing("cupy")
@pytest.mark.parametrize("minimizer", ("legacy_cupy", "lmfit_cupy"))
@pytest.mark.parametrize("metric", nrefocus.metrics.METRICS)
def test_cupy_minimizers_accept_registered_metrics(
        set_ndarray_backend_to_cupy, minimizer, metric):
    """Both CuPy minimizers accept all metrics, respecting spectrum ROI rules."""
    rng = np.random.default_rng(12)
    field = rng.normal(size=(18, 14)) + 1j * rng.normal(size=(18, 14))
    rf = nrefocus.RefocusCupy(
        field=field, wavelength=8.25, pixel_size=1, medium_index=1.333,
        kernel="helmholtz", padding=False)
    roi = None if metric == "spectrum" else (1, 1, 17, 13)
    distance = rf.autofocus(
        interval=(-4, 4), metric=metric, minimizer=minimizer, roi=roi,
        minimizer_kwargs=({"coarse_acc": 2} if minimizer == "legacy_cupy"
                          else {"coarse_step": 1}))
    assert np.isfinite(float(distance.get()))
    assert -4 <= float(distance.get()) <= 4
