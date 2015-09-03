#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Data from 2014_11_11/5min_rot_crash_sub1 -> frame 55
#
""" 
2D Refocusing of a live cell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The data shows a live HL60 cell imaged with the camera SID4BIO from
Phasics S.A. (France). The diameter of the cell is about 20µm.


.. figure::  ../examples/refocus_cell_repo.jpg
   :align:   center

   Numerically refocused HL60 cell.
   

Download the :download:`full example <../examples/refocus_cell.py>`.
"""
from __future__ import division, print_function

import numpy as np
from os.path import dirname, abspath
import sys
import zipfile

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import nrefocus


def load_cell(fname="./HL60_field.zip"):
    """Load zip file adn return complex field"""
    arc = zipfile.ZipFile(fname)
    for f in arc.filelist:
        with arc.open(f) as fd:
            if f.filename.count("imag"):
                imag = np.loadtxt(fd)

            elif f.filename.count("real"):
                real = np.loadtxt(fd)
    
    field = real + 1j*imag
    return field

def create_axes():
    fig, axes = plt.subplots(2,3, figsize=(10,6))
    axes = axes.flatten()
    for ax in axes:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    
    return fig, axes

if __name__ == "__main__":
    # We need the unwrap module to perform phase unwrapping
    import unwrap
    # We need matplotlib for plotting
    import matplotlib.pylab as plt
    
    # Load initial cell
    cell1 = load_cell()
    # Refocus to two different positions
    cell2 = nrefocus.refocus(cell1, 15, 1, 1)  # forward
    cell3 = nrefocus.refocus(cell1, -15, 1, 1) # backward

    ## Plot the results
    # amplitude range    
    vmina = np.min(np.abs(np.concatenate((cell1, cell2, cell3))))
    vmaxa = np.max(np.abs(np.concatenate((cell1, cell2, cell3))))
    ampkw = {"cmap": plt.cm.gray,  # @UndefinedVariable
             "vmin": vmina,
             "vmax": vmaxa-(vmaxa-vmina)/3}
    
    # phase range
    cell1p = unwrap.unwrap(np.angle(cell1))
    cell2p = unwrap.unwrap(np.angle(cell2))
    cell3p = unwrap.unwrap(np.angle(cell3))
    vminp = np.min(np.concatenate((cell1p, cell2p, cell3p)))
    vmaxp = np.max(np.concatenate((cell1p, cell2p, cell3p)))
    phakw = {"cmap": plt.cm.coolwarm,  # @UndefinedVariable
             "vmin": vminp,
             "vmax": vmaxp}
    
    # Plots
    fig, axes = create_axes()
    mapamp = axes[0].imshow(np.abs(cell3), **ampkw)
    axes[1].imshow(np.abs(cell1), **ampkw)
    axes[2].imshow(np.abs(cell2), **ampkw)
    mappha = axes[3].imshow(cell3p, **phakw)
    axes[4].imshow(cell1p, **phakw)
    axes[5].imshow(cell2p, **phakw)
    # Text labels
    textkw = {"fontsize": 20,
              "color": "white",
              "horizontalalignment": "left",
              "verticalalignment" : "top"
              }
    axes[0].text(4, 4, "focused backward", **textkw)
    axes[1].text(4, 4, "original image", **textkw)
    axes[2].text(4, 4, "focused forward", **textkw)
    plt.tight_layout(rect=(.07, 0, 1, 1), w_pad=0.055)
    # colorbar amplitude
    pa = axes[0].get_position()
    cbaxes = fig.add_axes([0.060 , pa.y0, .02, pa.y1-pa.y0])
    cb = fig.colorbar(mapamp, cax=cbaxes,
                        orientation="vertical",
                        label="amplitude [a.u.]")
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    # colorbar phase
    pa = axes[3].get_position()
    cbaxes = fig.add_axes([0.060 , pa.y0, .02, pa.y1-pa.y0])
    cb = fig.colorbar(mappha, cax=cbaxes,
                        orientation="vertical",
                        label="phase [rad]")
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    
    DIR = dirname(abspath(__file__))
    plt.savefig(DIR+"/refocus_cell.jpg", dpi=100)
    