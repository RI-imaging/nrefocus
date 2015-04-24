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
Please cite this package if you are using it in a scientific
publication.

This package should be cited like this 
(replace "x.x.x" with the actual version of nrefocus that you used):

.. topic:: cite

    Paul Müller (2013) *nrefocus: Python algorithms for numerical
    focusing* (Version x.x.x) [Software].
    Available at https://pypi.python.org/pypi/nrefocus/


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
