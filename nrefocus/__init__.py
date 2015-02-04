#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Numerical focusing of electric fields.
"""

#try:
#    import fftw3
#except ImportError:
#    raise ImportError("`fftw3` not found. Please install `python-fftw`")
## Number of cores to use for multiprocessing tasks
#import multiprocessing as mp
#_ncores = mp.cpu_count()
## Flags for fftw3
#_fftwflags = ["estimate"]


from .nr_version import __version__
from ._autofocus import *
from ._propagate import *

