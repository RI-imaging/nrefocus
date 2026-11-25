"""Compare CPU and GPU legacy/lmfit autofocus on a 20-frame HL60 stack.

Run from the repository root with::

    python examples/compare_minimizers.py

The figure shows the middle frame's metric search and focused amplitude for
each minimizer. The table reports middle-frame focus results and full-stack
runtime. GPU results are omitted if CuPy or a CUDA device is unavailable.
"""
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np

import nrefocus
from example_helper import load_cell

WAVELENGTH = 647e-9
PIXEL_SIZE = 0.139e-6
RES = WAVELENGTH / PIXEL_SIZE
INTERVAL = (-5e-6 / PIXEL_SIZE, 5e-6 / PIXEL_SIZE)
ROI = (slice(8, -8), slice(8, -8))
_TIMING_REPEATS = 3
_NFRAMES = 20


def _host_array(value):
    """Return a NumPy view/copy of either a NumPy or CuPy array."""
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def _make_stack(field):
    """Make n distinct planes by propagating the input across focus offsets."""
    nrefocus.set_ndarray_backend("numpy")
    rf = nrefocus.iface.RefocusNumpy(
        field=field, wavelength=RES, pixel_size=1, medium_index=1.333,
        kernel="helmholtz", padding=False)
    return rf.propagate(np.linspace(-2.0, 2.0, _NFRAMES))


def _run_case(fieldstack, backend, minimizer):
    """Run one stack autofocus case and collect median timing and outputs."""
    nrefocus.set_ndarray_backend(backend)
    if backend == "cupy":
        import cupy as cp

    options = ({"coarse_acc": 1, "fine_acc": 0.005}
               if minimizer in ("legacy", "legacy_cupy") else {})

    def run_stack():
        """Run and synchronize one complete stack autofocus call."""
        start = time.perf_counter()
        if backend == "cupy":
            distances, focused = nrefocus.autofocus_stack_cupy(
                fieldstack, nm=1.333, res=RES, ival=INTERVAL, roi=ROI,
                metric="average gradient", minimizer=minimizer,
                minimizer_kwargs=options, padding=False, num_streams=1,
                copy=True)
            cp.cuda.get_current_stream().synchronize()
            output = (cp.asarray(distances).get(), focused.get())
        else:
            output = nrefocus.autofocus_stack(
                fieldstack, nm=1.333, res=RES, ival=INTERVAL, roi=ROI,
                metric="average gradient", minimizer=minimizer,
                minimizer_kwargs=options, padding=False)
        return output, time.perf_counter() - start

    # Warm up GPU kernels and FFT plans before recording comparable timings.
    run_stack()
    timings = []
    for _ in range(_TIMING_REPEATS):
        output, elapsed = run_stack()
        timings.append(elapsed)
    distances, focused = output
    distances = np.asarray(distances, dtype=float)

    representative = _NFRAMES // 2
    rf_cls = (nrefocus.iface.RefocusCupy if backend == "cupy"
              else nrefocus.iface.RefocusNumpy)
    rf = rf_cls(field=fieldstack[representative], wavelength=RES,
                pixel_size=1, medium_index=1.333, kernel="helmholtz",
                padding=False)
    distance, (grid, metric_values) = rf.autofocus(
        interval=INTERVAL, metric="average gradient", minimizer=minimizer,
        minimizer_kwargs=options, roi=ROI, ret_grid=True)
    distance = float(_host_array(distance))
    grid = _host_array(grid)
    metric_values = _host_array(metric_values)
    focused_roi = np.abs(focused[:, ROI[0], ROI[1]])
    gy, gx = np.gradient(focused_roi, axis=(-2, -1))
    metric_at_focus = float(np.mean(
        np.mean(gy ** 2 + gx ** 2, axis=(-2, -1)) / 2))
    return {
        "backend": backend,
        "minimizer": minimizer,
        "distance": distances[representative],
        "metric": metric_at_focus,
        "seconds": statistics.median(timings),
        "grid": grid,
        "metric_values": metric_values,
        "focused": focused[representative],
        "distances": distances,
    }


def main():
    """Run four stack comparisons and display plots plus a summary table."""
    field = load_cell("HL60_field.zip")
    fieldstack = _make_stack(field)
    cases = [("numpy", "legacy"), ("numpy", "lmfit")]
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            cases.extend([("cupy", "legacy_cupy"),
                          ("cupy", "lmfit_cupy")])
        else:
            print("No CUDA device found; showing CPU comparisons only.")
    except Exception as exc:
        print("CuPy/CUDA unavailable ({}); showing CPU comparisons "
              "only.".format(exc))

    results = [_run_case(fieldstack, backend, minimizer)
               for backend, minimizer in cases]
    lmfit_reference = next(row for row in results
                           if row["backend"] == "numpy" and
                           row["minimizer"] == "lmfit")
    reference = lmfit_reference["distances"]

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    layout = fig.add_gridspec(4, 4, height_ratios=(1.1, 0.16, 1.0, 0.8))
    image_axes = [fig.add_subplot(layout[0, i]) for i in range(4)]
    curve_ax = fig.add_subplot(layout[2, :])
    table_ax = fig.add_subplot(layout[3, :])

    markers = ("o", "s", "^", "D")
    for index, result in enumerate(results):
        curve_ax.plot(
            result["grid"] * PIXEL_SIZE * 1e6,
            result["metric_values"],
            marker=markers[index],
            markevery=(index, max(1, len(result["grid"]) // 12)),
            markersize=6,
            markerfacecolor="none",
            markeredgewidth=1.2,
            label='minimizer="{}" (focus {:.3f} µm)'.format(
                result["minimizer"],
                result["distance"] * PIXEL_SIZE * 1e6),
        )
    curve_ax.axvline(
        lmfit_reference["distance"] * PIXEL_SIZE * 1e6,
        color="black", linestyle="--", linewidth=1.2,
        label="lmfit minimum")
    curve_ax.set(title="Autofocus search",
                 xlabel="Focus distance (µm)",
                 ylabel="Metric (Average Gradient)")
    curve_ax.legend(loc="lower right")
    curve_ax.grid(alpha=0.2)

    amplitude_limits = (np.min(np.abs(fieldstack)), np.max(np.abs(fieldstack)))
    images = []
    for ax, result in zip(image_axes, results):
        image = ax.imshow(np.abs(result["focused"]), cmap="gray",
                          vmin=amplitude_limits[0], vmax=amplitude_limits[1])
        images.append(image)
        ax.set_title('minimizer="{}"'.format(result["minimizer"]))
        ax.set_axis_off()
    for ax in image_axes[len(results):]:
        ax.set_axis_off()
    if images:
        fig.colorbar(images[0], ax=image_axes[:len(results)],
                     fraction=0.025, label="Amplitude")

    table_ax.axis("off")
    columns = ["Minimizer", "Median focus (µm)", "Mean metric at focus",
               "Stack time (s, median of 3)",
               'Median Δ from minimizer="lmfit" (nm)']
    rows = [[
        'minimizer="{}"'.format(result["minimizer"]),
        "{:.6f}".format(np.median(result["distances"]) * PIXEL_SIZE * 1e6),
        "{:.6g}".format(result["metric"]),
        "{:.3f}".format(result["seconds"]),
        "{:+.2f}".format(np.median(
            (result["distances"] - reference) * PIXEL_SIZE * 1e9)),
    ] for result in results]
    table_ax.table(cellText=rows, colLabels=columns, loc="center",
                   cellLoc="center")
    fig.suptitle("Autofocus Minimizer Comparison \n"
                 f"(plots show the middle frame of {_NFRAMES} frames)")
    plt.savefig("compare_minimizers.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
