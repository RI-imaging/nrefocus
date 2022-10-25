import numpy as np


def metric_med_gradient(rfi, distance, roi=None, **kwargs):
    """Compute median gradient norm of the amplitude

    Notes
    -----
    The absolute value of the gradient is returned.
    """
    data = np.abs(rfi.propagate(distance))
    if roi is not None:
        data = data[roi]
    return np.median(np.array(np.gradient(data))**2)
