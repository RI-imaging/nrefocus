import numpy as np


def metric_rms_contrast(rfi, distance, roi=None, *kwargs):
    """Compute RMS contrast of the phase

    Notes
    -----
    The negative angle of the field is used for contrast estimation.
    """
    data = -np.anlge(rfi.propagate(distance))
    av = np.average(data, *kwargs)
    mal = 1 / (data.shape[0] * data.shape[1])
    if roi:
        data = data[roi]
    return np.sqrt(mal * np.sum((data - av)**2))
