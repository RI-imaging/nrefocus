from .._ndarray_backend import xp


def metric_std_gradient(rfi, distance, roi=None, **kwargs):
    """Compute standard deviation (std) gradient of the amplitude

    Notes
    -----
    The absolute value of the gradient is returned.
    """
    data = xp.abs(rfi.propagate(distance))
    if roi is not None:
        data = data[roi]
    return xp.std(xp.array(xp.gradient(data)))
