"""2D Refocusing of an HL60 cell on the GPU with Cupy

The data show a live HL60 cell imaged with quadriwave lateral shearing
interferometry (SID4Bio, Phasics S.A., France).
The diameter of the cell is about 20µm.
"""
import matplotlib.pylab as plt
from skimage.restoration import unwrap_phase
import numpy as np

import nrefocus

from example_helper import load_cell

# load initial cell
cell1 = load_cell("HL60_field.zip")
pixel_size = 1e-6

# refocus to two different positions
# right now, we must use the `rf = nrefocus.iface.RefocusCupy` syntax for cupy
rf = nrefocus.iface.RefocusPyFFTW(cell1, wavelength=pixel_size,
                                  medium_index=1,
                                  pixel_size=pixel_size)
cell2 = rf.propagate(distance=15 * pixel_size)
cell3 = rf.propagate(distance=-15 * pixel_size)

# amplitude range
vmina = np.min(np.abs(cell1))
vmaxa = np.max(np.abs(cell1))
ampkw = {"cmap": plt.get_cmap("gray"),
         "vmin": vmina,
         "vmax": vmaxa}

# phase range
cell1p = unwrap_phase(np.angle(cell1))
cell2p = unwrap_phase(np.angle(cell2))
cell3p = unwrap_phase(np.angle(cell3))
vminp = np.min(cell1p)
vmaxp = np.max(cell1p)
phakw = {"cmap": plt.get_cmap("coolwarm"),
         "vmin": vminp,
         "vmax": vmaxp}

# plots
fig, axes = plt.subplots(2, 3, figsize=(8, 4.5))
axes = axes.flatten()
for ax in axes:
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

# titles
axes[0].set_title("focused backward")
axes[1].set_title("original image")
axes[2].set_title("focused forward")

# data
mapamp = axes[0].imshow(np.abs(cell3), **ampkw)
axes[1].imshow(np.abs(cell1), **ampkw)
axes[2].imshow(np.abs(cell2), **ampkw)
mappha = axes[3].imshow(cell3p, **phakw)
axes[4].imshow(cell1p, **phakw)
axes[5].imshow(cell2p, **phakw)

# colobars
cbkwargs = {"fraction": 0.045}
plt.colorbar(mapamp, ax=axes[0], label="amplitude [a.u.]", **cbkwargs)
plt.colorbar(mapamp, ax=axes[1], label="amplitude [a.u.]", **cbkwargs)
plt.colorbar(mapamp, ax=axes[2], label="amplitude [a.u.]", **cbkwargs)
plt.colorbar(mappha, ax=axes[3], label="phase [rad]", **cbkwargs)
plt.colorbar(mappha, ax=axes[4], label="phase [rad]", **cbkwargs)
plt.colorbar(mappha, ax=axes[5], label="phase [rad]", **cbkwargs)

plt.suptitle("Refocused cell on CPU with PyFFTW")
plt.tight_layout()
plt.show()
