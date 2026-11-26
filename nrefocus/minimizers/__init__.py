# flake8: noqa: F401

from .mz_legacy import minimize_legacy
from .mz_lmfit import minimize_lmfit
from .mz_legacy_cupy import minimize_legacy_cupy
from .mz_lmfit_cupy import minimize_lmfit_cupy

#: Available minimizers
MINIMIZERS = {
    "legacy": minimize_legacy,
    "lmfit": minimize_lmfit,
    "legacy_cupy": minimize_legacy_cupy,
    "lmfit_cupy": minimize_lmfit_cupy,
}
