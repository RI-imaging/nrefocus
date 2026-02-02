import cupy as cp


def metric_rms_contrast(rfi, distance, roi=None, **kwargs):
    """Compute RMS contrast of the phase

    Notes
    -----
    The negative angle of the field is used for contrast estimation.
    """
    data = -cp.angle(rfi.propagate(distance))
    av = cp.average(data, *kwargs)
    mal = 1 / (data.shape[0] * data.shape[1])
    if roi is not None:
        data = data[roi]
    return cp.sqrt(mal * cp.sum((data - av)**2))
