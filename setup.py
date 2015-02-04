#!/usr/bin/env python
# To create a distribution package for pip or easy-install:
# python setup.py sdist
from setuptools import setup, find_packages, Command
from os.path import join, dirname, realpath
from warnings import warn

name='nrefocus'

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable, 'tests/runtests.py'])
        raise SystemExit(errno)

try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/"+name)
    # get version number
    from _nr_version import __version__ as version
except:
    version = "unknown"

setup(
    name=name,
    author='Paul Müller',
    #author_email='richard.hartmann...',
    url='https://github.com/paulmueller/nrefocus',
    version=version,
    packages=[name],
    package_dir={name: name},
    license="BSD (3 clause)",
    description='Python library for numerical (auto)refocusing of complex wave fields',
    long_description="""This module provides the means to numerically
refocus complex wave fields, such as those acquired using quantitative 
phase measuring techniques in modern microscopy. The module also comes
with a couple of autofocusing metrics.
""",
    install_requires=["NumPy>=1.5.1"],
#    tests_require=["psutil"],
    keywords=["autofocus", "refocus", "numerical focusing", "DHM",
              "phase imaging", "quantitative phase",
              "digital holography"],
    classifiers= [
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Intended Audience :: Science/Research'
                 ],
    platforms=['ALL'],
    cmdclass = {'test': PyTest},
    )


