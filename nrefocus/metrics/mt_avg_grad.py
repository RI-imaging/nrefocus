from .._ndarray_backend import xp


def metric_average_gradient(rfi, distance, roi=None, **kwargs):
    """Compute mean average gradient norm of the amplitude

    Notes
    -----
    The absolute value of the gradient is returned.
    """
    data = xp.abs(rfi.propagate(distance))
    if roi is not None:
        data = data[roi]
    return xp.average(xp.array(xp.gradient(data))**2)
