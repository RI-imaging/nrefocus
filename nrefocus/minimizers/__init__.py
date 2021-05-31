# flake8: noqa: F401

from .mz_legacy import minimize_legacy
from .mz_lmfit import minimize_lmfit

#: Available minimizers
MINIMIZERS = {
    "legacy": minimize_legacy,
    "lmfit": minimize_lmfit,
}
