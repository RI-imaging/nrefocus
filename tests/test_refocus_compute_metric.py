import pathlib

import numpy as np

import nrefocus


from test_helper import load_cell

data_path = pathlib.Path(__file__).parent / "data"


def test_2d_basic_metric():
    rf = nrefocus.iface.RefocusNumpy(field=load_cell("HL60_field.zip"),
                                     wavelength=647e-9,
                                     pixel_size=0.139e-6,
                                     kernel="helmholtz",
                                     )
    distances, metric_val = rf.compute_metric(
        interval=(-5e-6, 5e-6),
        metric="average gradient",
        roi=None,
        num_steps=50
        )
    # the minimum should be at
    # - metric: 0.0008821168202210964
    # - distance: 5.959024521240163e-07
    idmin = np.argmin(np.abs(metric_val - 0.0008821168202210964))
    idmin2 = np.argmin(np.abs(metric_val - 5.959024521240163e-07))
    assert idmin == idmin2


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
