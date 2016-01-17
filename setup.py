#!/usr/bin/env python
# -*- coding: utf-8 -*-
# To create a distribution package for pip or easy-install:
# python setup.py sdist
from setuptools import setup
from os.path import exists, dirname, realpath
import sys


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


if __name__ == "__main__":
    setup(
        name=name,
        author=author,
        url='https://github.com/RI-imaging/nrefocus',
        author_email='paul.mueller@biotec.tu-dresden.de',
        version=version,
        packages=[name],
        package_dir={name: name},
        license="BSD (3 clause)",
        description=description,
        long_description=open('README.rst').read() if exists('README.rst') else '',
        install_requires=["NumPy>=1.5.1"],
        setup_requires=['pytest-runner'],
        tests_require=["pytest"],
        keywords=["autofocus", "refocus", "numerical focusing", "DHM",
                  "phase imaging", "quantitative phase",
                  "digital holography"],
        classifiers= [
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Intended Audience :: Science/Research'
                     ],
        platforms=['ALL']
        )

