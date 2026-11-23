# flake8: noqa: F401

from .mz_legacy import minimize_legacy
from .mz_lmfit import minimize_lmfit
from .mz_cupy import minimize_cupy

#: Available minimizers
MINIMIZERS = {
    "legacy": minimize_legacy,
    "lmfit": minimize_lmfit,
    "cupy": minimize_cupy,
}
