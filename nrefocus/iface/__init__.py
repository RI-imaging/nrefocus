# flake8: noqa: F401
from .rf_numpy import RefocusNumpy
from .rf_numpy_1d import RefocusNumpy1D

try:
    import pyfftw
except ImportError:
    pyfftw = None
    RefocusPyFFTW = None
else:
    from .rf_pyfftw import RefocusPyFFTW


def get_best_interface():
    """Return the fastest refocusing interface available

    If `pyfftw` is installed, :class:`nrefocus.RefocusPyFFTW`
    is returned. The fallback is :class:`nrefocus.RefocusNumpy`.
    """
    ordered_candidates = [
        RefocusPyFFTW,
        RefocusNumpy,
    ]
    for cand in ordered_candidates:
        if cand is not None:
            return cand
