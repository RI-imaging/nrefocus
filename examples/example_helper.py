import os.path as op
import zipfile

import numpy as np


def load_cell(fname="HL60_field.zip"):
    "Load zip file and return complex field"
    here = op.dirname(op.abspath(__file__))
    data = op.join(here, "data")
    arc = zipfile.ZipFile(op.join(data, fname))
    for f in arc.filelist:
        with arc.open(f) as fd:
            if f.filename.count("imag"):
                imag = np.loadtxt(fd)

            elif f.filename.count("real"):
                real = np.loadtxt(fd)

    field = real + 1j * imag
    return field
