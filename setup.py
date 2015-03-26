#!/usr/bin/env python
# -*- coding: utf-8 -*-
# To create a distribution package for pip or easy-install:
# python setup.py sdist
from setuptools import setup, find_packages, Command
from os.path import join, dirname, realpath
import sys
from warnings import warn


author = u"Paul Müller"
authors = [author]
description = 'library for numerical focusing (refocusing, autofocusing) of complex wave fields'
name = 'nrefocus'
year = "2015"


sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
try:
    from _version import version
except:
    version = "unknown"


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


if __name__ == "__main__":
    setup(
        name=name,
        author=author,
        url='https://github.com/paulmueller/nrefocus',
        author_email='paul.mueller@biotec.tu-dresden.de',
        version=version,
        packages=[name],
        package_dir={name: name},
        license="BSD (3 clause)",
        description=description,
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

