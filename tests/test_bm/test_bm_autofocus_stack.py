"""Benchmark serial, multiprocessing, and GPU stack autofocus paths."""
import io
import multiprocessing as mp
from pathlib import Path
import statistics
import time
import zipfile

import numpy as np
import pytest

import nrefocus


_NFRAMES = 32
_NM = 1.333
_RES = 8.25
_INTERVAL = (-5, 5)
_ROI = (8, 8, 104, 104)
_COARSE_ACC = 4
_FINE_ACC = 0.005
_NUM_STREAMS = 1


def _make_stack():
    """Make _NFRAMES distinct inputs from the bundled HL60 field data."""
    nrefocus.set_ndarray_backend("numpy")
    data_file = Path(__file__).parents[1] / "data" / "HL60_field.zip"
    with zipfile.ZipFile(data_file) as archive:
        real = np.loadtxt(io.TextIOWrapper(archive.open("HL60_real.txt")))
        imag = np.loadtxt(io.TextIOWrapper(archive.open("HL60_imag.txt")))
    field = real + 1j * imag
    offsets = np.linspace(-2.0, 2.0, _NFRAMES)
    return np.stack([
        nrefocus.refocus(field, d=d, nm=_NM, res=_RES,
                         method="helmholtz", padding=False)
        for d in offsets
    ])


@pytest.fixture(scope="module")
def autofocus_stack_data():
    return _make_stack()


@pytest.mark.parametrize(
    "method",
    ["autofocus", "autofocus_stack", "autofocus_stack_gpu_cupy",
     "autofocus_stack_gpu_lmfit"],
    ids=["autofocus", "autofocus_stack_cpu", "autofocus_stack_gpu_batched",
         "autofocus_stack_gpu_lmfit"],
)
def test_bm_autofocus_stack(benchmark, method, autofocus_stack_data):
    """Benchmark equivalent per-frame autofocus workloads.

    The GPU callable reports upload, processing/completion, and download
    stages separately through pytest-benchmark extra info and stdout.
    """
    fieldstack = autofocus_stack_data
    common = dict(
        nm=_NM,
        res=_RES,
        ival=_INTERVAL,
        roi=_ROI,
        metric="average gradient",
        minimizer="legacy",
        minimizer_kwargs={
            "coarse_acc": _COARSE_ACC,
            "fine_acc": _FINE_ACC,
        },
        padding=False,
    )

    if method == "autofocus":
        nrefocus.set_ndarray_backend("numpy")

        def run_serial():
            return [nrefocus.autofocus(frame, **common)
                    for frame in fieldstack]

        result = run_serial()
        assert len(result) == _NFRAMES
        benchmark.pedantic(run_serial, rounds=3, warmup_rounds=0)
        return

    if method == "autofocus_stack":
        nrefocus.set_ndarray_backend("numpy")

        def run_cpu_stack():
            return nrefocus.autofocus_stack(
                fieldstack, **common, num_cpus=mp.cpu_count())

        distances, focused = run_cpu_stack()
        assert len(distances) == _NFRAMES
        assert focused.shape == fieldstack.shape
        benchmark.pedantic(run_cpu_stack, rounds=3, warmup_rounds=0)
        return

    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device is available.")
    except Exception as exc:
        pytest.skip("CuPy could not initialize CUDA: {}".format(exc))

    nrefocus.set_ndarray_backend("cupy")
    gpu_common = dict(common)
    if method == "autofocus_stack_gpu_lmfit":
        gpu_common["minimizer"] = "lmfit"
        gpu_common["minimizer_kwargs"] = {}
    else:
        gpu_common["minimizer"] = "cupy"
    upload_start = time.perf_counter()
    gpu_stack = cp.asarray(fieldstack)
    cp.cuda.get_current_stream().synchronize()
    upload_seconds = time.perf_counter() - upload_start

    # Preflight and warm up CuPy/cuFFT outside measured rounds.
    gpu_distances, gpu_fields = nrefocus.autofocus_stack_cupy(
        gpu_stack, **gpu_common, num_streams=_NUM_STREAMS, copy=False)
    cp.cuda.get_current_stream().synchronize()
    assert len(gpu_distances) == _NFRAMES
    assert gpu_fields.shape == fieldstack.shape
    cp.asarray(gpu_distances).get()
    gpu_fields.get()

    processing_samples = []
    download_samples = []

    def run_gpu_stack():
        start = time.perf_counter()
        distances, focused = nrefocus.autofocus_stack_cupy(
            gpu_stack, **gpu_common, num_streams=_NUM_STREAMS, copy=False)
        # Include completion of output stacking in GPU processing time.
        cp.cuda.get_current_stream().synchronize()
        processing_samples.append(time.perf_counter() - start)

        start = time.perf_counter()
        cp.asarray(distances).get()
        focused.get()
        download_samples.append(time.perf_counter() - start)

    benchmark.pedantic(run_gpu_stack, rounds=3, warmup_rounds=0)

    processing_median = statistics.median(processing_samples)
    download_median = statistics.median(download_samples)
    benchmark.extra_info["GPU input upload (s)"] = upload_seconds
    benchmark.extra_info[
        "GPU processing, synchronized (s)"] = processing_median
    benchmark.extra_info["GPU output download (s)"] = download_median
    benchmark.extra_info["GPU end-to-end (s)"] = (
        upload_seconds + processing_median + download_median)
    print("GPU timing stages (median of 3 rounds): upload={:.3f}s, "
          "processing={:.3f}s, download={:.3f}s".format(
              upload_seconds, processing_median, download_median))
