from .._ndarray_backend import xp


def metric_med_gradient(rfi, distance, roi=None, **kwargs):
    """Compute median gradient norm of the amplitude

    Notes
    -----
    The absolute value of the gradient is returned.
    """
    data = xp.abs(rfi.propagate(distance))
    if roi is not None:
        data = data[(slice(None),) * (data.ndim - 2) + roi]
    gradients = xp.gradient(data, axis=(-2, -1))
    return xp.median(xp.stack(gradients, axis=-3) ** 2,
                     axis=(-3, -2, -1))
