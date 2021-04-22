nrefocus
========

|PyPI Version| |Tests Status| |Coverage Status| |Docs Status|

Numerically (auto)refocus complex wave fields, such as those acquired using
quantitative phase imaging techniques in modern microscopy.


Documentation
-------------

The documentation is available is available at
`nrefocus.readthedocs.io <https://nrefocus.readthedocs.io/en/stable/>`__.


Installation
------------
Install from the Python package index (the `FFTW` extra enables fast
Fourer transforms with `PyFFTW <https://pyfftw.readthedocs.io/>`__)::

    pip install nrefocus[FFTW]

or clone the repository and run::

    pip install -e .


Testing
-------
Testing is done with pytest::

    pip install pytest
    pytest tests



.. |PyPI Version| image:: https://img.shields.io/pypi/v/nrefocus.svg
   :target: https://pypi.python.org/pypi/nrefocus
.. |Tests Status| image:: https://img.shields.io/github/workflow/status/RI-Imaging/nrefocus/Checks
   :target: https://github.com/RI-Imaging/nrefocus/actions?query=workflow%3AChecks
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/RI-imaging/nrefocus/master.svg
   :target: https://codecov.io/gh/RI-imaging/nrefocus
.. |Docs Status| image:: https://readthedocs.org/projects/nrefocus/badge/?version=latest
   :target: https://readthedocs.org/projects/nrefocus/builds/
