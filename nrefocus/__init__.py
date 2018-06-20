"""
This package provides methods for numerical propagation of a complex
wave in free space. The available propagators are the angular spectrum
method (`helmholtz`) and the Fresnel approximation (`fresnel`). Both
implementations are convolution-based. The angular spectrum method is
suited for near-field propagation (numerical focusing) and yields
better results than the Fresnel approximation.
The single Fourer transform-based Fresnel propagation method which is
suitable for far-field propagation is not implemented in this package.


Obtaining nrefocus
------------------
If you have Python and :py:mod:`numpy` installed, simply run

    pip install nrefocus

The source code of nrefocus is available at
https://github.com/RI-imaging/nrefocus.


Citing nrefocus
---------------
Please cite this package if you are using it in a scientific
publication.

This package should be cited like this [1]_.

You can find out what version you are using by typing
(in a Python console):

    >>> import nrefocus
    >>> nrefocus.__version__
    '0.1.2'


References
----------
.. [1] Paul Müller (2013) *nrefocus: Python algorithms for numerical
       focusing* (Version x.x.x) [Software].
       Available at https://pypi.python.org/pypi/nrefocus/.

"""

# try:
#    import fftw3
# except ImportError:
#    raise ImportError("`fftw3` not found. Please install `python-fftw`")
# Number of cores to use for multiprocessing tasks
# import multiprocessing as mp
# _ncores = mp.cpu_count()
#  Flags for fftw3
# _fftwflags = ["estimate"]


from ._autofocus import autofocus, autofocus_stack  # noqa: F401
from ._propagate import refocus, refocus_stack  # noqa: F401
from . import pad  # noqa: F401

from ._version import version as __version__  # noqa: F401
