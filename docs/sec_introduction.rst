============
Introduction
============
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
You can install nrefocus via::

    pip install nrefocus

If you would like to take advantage of fast Fourer transforms with
`PyFFTW <https://pyfftw.readthedocs.io/>`__, please also install the
`pyfftw` package or use the extras key `FFTW`::

    pip install nrefocus[FFTW]

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


Acknowledgments
---------------
This project has received funding from the European Union’s Seventh Framework
Programme for research, technological development and demonstration under
grant agreement no 282060.


References
----------
.. [1] Paul Müller (2013) *nrefocus: Python algorithms for numerical
       focusing* (Version x.x.x) [Software].
       Available at https://pypi.python.org/pypi/nrefocus/.
