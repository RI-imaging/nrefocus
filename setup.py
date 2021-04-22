from os.path import exists, dirname, realpath
from setuptools import setup, find_packages
import sys


author = "Paul Müller"
authors = [author]
description = "numerical focusing (refocusing, autofocusing) of " \
    "complex wave fields"
name = "nrefocus"
year = "2015"

sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
from _version import version  # noqa: E402

setup(
    name=name,
    author=author,
    author_email="dev@craban.de",
    url="https://github.com/RI-imaging/nrefocus",
    version=version,
    packages=find_packages(),
    package_dir={name: name},
    license="BSD (3 clause)",
    description=description,
    long_description=open("README.rst").read() if exists("README.rst") else "",
    install_requires=[
        "numexpr",
        "numpy>=1.5.1",
    ],
    extras_require={"FFTW": "pyfftw>=0.12.0"},
    python_requires=">=3.6, <4",
    keywords=["autofocus",
              "refocus",
              "numerical focusing",
              "quantitative phase imaging",
              "digital holographic microscopy",
              ],
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research"
        ],
    platforms=["ALL"]
    )
