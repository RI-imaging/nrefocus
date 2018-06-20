from os.path import exists, dirname, realpath
from setuptools import setup
import sys


author = "Paul Müller"
authors = [author]
description = 'numerical focusing (refocusing, autofocusing) of complex wave fields'
name = 'nrefocus'
year = "2015"

sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
from _version import version

setup(
    name=name,
    author=author,
    author_email='dev@craban.de',
    url='https://github.com/RI-imaging/nrefocus',
    version=version,
    packages=[name],
    package_dir={name: name},
    license="BSD (3 clause)",
    description=description,
    long_description=open('README.rst').read() if exists('README.rst') else '',
    install_requires=["numpy>=1.5.1"],
    setup_requires=['pytest-runner'],
    tests_require=["pytest"],
    python_requires='>=3.4, <4',
    keywords=["autofocus",
              "refocus",
              "numerical focusing",
              "quantitative phase imaging",
              "digital holographic microscopy",
              ],
    classifiers= [
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research'
        ],
    platforms=['ALL']
    )
