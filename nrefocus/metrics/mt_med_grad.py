import cupy as cp


def metric_med_gradient(rfi, distance, roi=None, **kwargs):
    """Compute median gradient norm of the amplitude

    Notes
    -----
    The absolute value of the gradient is returned.
    """
    data = cp.abs(rfi.propagate(distance))
    if roi is not None:
        data = data[roi]
    return cp.median(cp.array(cp.gradient(data))**2)
