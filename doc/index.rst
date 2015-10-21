nrefocus reference
==================
.. toctree::
   :maxdepth: 2

About
:::::

For a quick overview, see :ref:`genindex`.

.. automodule:: nrefocus

Acknowledgments
~~~~~~~~~~~~~~~
This project has received funding from the European Union’s Seventh Framework
Programme for research, technological development and demonstration under
grant agreement no 282060.


Theory
::::::
.. include:: ./content_theory.txt

Refocus 1D/2D fields
::::::::::::::::::::
.. automodule:: nrefocus._propagate
.. currentmodule:: nrefocus
.. autosummary:: 
    fft_propagate
    refocus
    refocus_stack

Fourier-domain propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: fft_propagate

Refocus individual fields
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: refocus

Refocus field stacks
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: refocus_stack


Autofocus 1D/2D fields
::::::::::::::::::::::
.. automodule:: nrefocus._autofocus
.. currentmodule:: nrefocus
.. autosummary:: 
    autofocus
    autofocus_stack

Metrics
~~~~~~~
.. automodule:: nrefocus.metrics
   :members:

.. currentmodule:: nrefocus

Autofocus single fields
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: autofocus

Autofocus field stacks
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: autofocus_stack


Examples
========

.. automodule:: refocus_cell
