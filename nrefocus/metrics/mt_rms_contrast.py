from .._ndarray_backend import xp


def metric_rms_contrast(rfi, distance, roi=None, **kwargs):
    """Compute RMS contrast of the phase

    Notes
    -----
    The negative angle of the field is used for contrast estimation.
    """
    data = -xp.angle(rfi.propagate(distance))
    spatial_axes = (-2, -1)
    av = xp.mean(data, axis=spatial_axes, keepdims=True)
    mal = 1 / (data.shape[-2] * data.shape[-1])
    if roi is not None:
        data = data[(slice(None),) * (data.ndim - 2) + roi]
    return xp.sqrt(mal * xp.sum((data - av)**2, axis=spatial_axes))
