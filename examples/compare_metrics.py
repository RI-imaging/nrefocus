"""Compare available Metrics for Refocusing

The HL60 cell data used is described in `refocus_cell.py`.
"""
import nrefocus
from nrefocus.metrics import METRICS
import matplotlib.pyplot as plt

from examples.example_helper import load_cell

nrefocus.set_ndarray_backend("cupy")

dataset = load_cell("HL60_field.zip")

# load initial cell and create rf object
rf = nrefocus.iface.RefocusCupy(field=dataset,
                                wavelength=647e-9,
                                pixel_size=0.139e-6,
                                kernel="helmholtz")

# autofocus the image for each metric
my_metrics = list(METRICS.keys())

fig, axes = plt.subplots(1, len(my_metrics), figsize=(12, 5))

for i, mt in enumerate(my_metrics):
    af_vals = rf.autofocus(metric=mt,
                           minimizer="legacy",
                           interval=(-5e-6, 5e-6),
                           ret_field=True,
                           ret_grid=True)
    d, grid, field = af_vals
    d = d.get()
    field = field.get()
    grid_0 = grid[0].get()
    grid_1 = grid[1].get()

    axes[i].plot(grid_0, grid_1)
    axes[i].axvline(d, color='k', ls='--')
    axes[i].set_title(f"Metric {mt}")

fig.suptitle("Comparison of Metrics")
fig.tight_layout()
plt.show()
