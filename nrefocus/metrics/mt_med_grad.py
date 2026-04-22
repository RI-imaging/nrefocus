from .._ndarray_backend import xp


def metric_med_gradient(rfi, distance, roi=None, **kwargs):
    """Compute median gradient norm of the amplitude

    Notes
    -----
    The absolute value of the gradient is returned.
    """
    data = xp.abs(rfi.propagate(distance))
    if roi is not None:
        data = data[roi]
    return xp.median(xp.array(xp.gradient(data))**2)
