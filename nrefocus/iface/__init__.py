# flake8: noqa: F401
import warnings

from .rf_numpy import RefocusNumpy
from .rf_numpy_1d import RefocusNumpy1D

try:
    import pyfftw
except ImportError:
    pyfftw = None
    RefocusPyFFTW = None
    warnings.warn("Interface 'RefocusPyFFTW' unavailable!")
else:
    from .rf_pyfftw import RefocusPyFFTW

try:
    import cupy
except ImportError:
    cupy = None
    RefocusCupy = None
    warnings.warn("Interface 'RefocusCupy' unavailable!")
else:
    from .rf_cupy import RefocusCupy


def get_best_interface():
    """Return the fastest refocusing interface available

    If `cupy` is installed, :class:`nrefocus.RefocusCupy`
    is returned. If `cupy` is not installed, then
    if `pyfftw` is installed, :class:`nrefocus.RefocusPyFFTW`
    is returned. The fallback is :class:`nrefocus.RefocusNumpy`.
    """
    ordered_candidates = [
        RefocusCupy,
        RefocusPyFFTW,
        RefocusNumpy,
    ]
    for cand in ordered_candidates:
        if cand is not None:
            return cand
