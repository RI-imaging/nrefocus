import cupy as cp


def metric_average_gradient(rfi, distance, roi=None, **kwargs):
    """Compute mean average gradient norm of the amplitude

    Notes
    -----
    The absolute value of the gradient is returned.
    """
    data = cp.abs(rfi.propagate(distance))
    if roi is not None:
        data = data[roi]
    return cp.average(cp.array(cp.gradient(data))**2)
