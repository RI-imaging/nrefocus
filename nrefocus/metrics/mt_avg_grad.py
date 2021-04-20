import numpy as np


def metric_average_gradient(rfi, distance, roi=None, *kwargs):
    """Compute mean average gradient norm of the amplitude

    Notes
    -----
    The absolute value of the gradient is returned.
    """
    data = np.abs(rfi.propagate(distance))
    if roi:
        data = data[roi]
    return np.average(np.array(np.gradient(data))**2)
