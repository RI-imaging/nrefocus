from .._ndarray_backend import xp


def metric_std_gradient(rfi, distance, roi=None, **kwargs):
    """Compute standard deviation (std) gradient of the amplitude

    Notes
    -----
    The absolute value of the gradient is returned.
    """
    data = xp.abs(rfi.propagate(distance))
    if roi is not None:
        data = data[(slice(None),) * (data.ndim - 2) + roi]
    gradients = xp.gradient(data, axis=(-2, -1))
    return xp.std(xp.stack(gradients, axis=-3), axis=(-3, -2, -1))
