#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Obtaining nrefocus 
------------------
If you have Python and :py:mod:`numpy` installed, simply run

    pip install nrefocus

The source code of nrefocus is available at
https://github.com/paulmueller/nrefocus.


Citing nrefocus
---------------
The nrefocus package should be cited like this (replace "x.x.x"
with the actual version of nrefocus that you used and "DD Month YYYY"
with a matching date).

.. topic:: cite

    Paul Müller (2013) *Python algorithms for numerical focusing*
    (Version x.x.x)
    [Computer program].
    Available at https://pypi.python.org/pypi/nrefocus/
    (Accessed DD Month YYYY)


You can find out what version you are using by typing
(in a Python console):


    >>> import nrefocus
    >>> nrefocus.__version__
    '0.1.2'


"""

# try:
#    import fftw3
# except ImportError:
#    raise ImportError("`fftw3` not found. Please install `python-fftw`")
# Number of cores to use for multiprocessing tasks
#import multiprocessing as mp
#_ncores = mp.cpu_count()
# Flags for fftw3
#_fftwflags = ["estimate"]


from ._autofocus import *
from ._propagate import *

from ._version import version as __version__
