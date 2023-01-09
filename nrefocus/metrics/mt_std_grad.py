import numpy as np


def metric_std_gradient(rfi, distance, roi=None, **kwargs):
    """Compute standard deviation (std) gradient of the amplitude

    Notes
    -----
    The absolute value of the gradient is returned.
    """
    data = np.abs(rfi.propagate(distance))
    if roi is not None:
        data = data[roi]
    return np.std(np.array(np.gradient(data)))
