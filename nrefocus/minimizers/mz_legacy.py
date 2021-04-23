import warnings

import numpy as np


def minimize_legacy(rf, metric_func, interval, roi=None,
                    coarse_acc=1, fine_acc=.005,
                    ret_gradient=False, padding=None,
                    return_gradient=None):
    """Find the focus by minimizing the `metric` of an image

    This is the implementation of the legacy nrefocus minimizer.

    Parameters
    ----------
    rf: nrefocus.iface.Refocus
        Refocus interface
    metric_func: callable
        metric called during minimization. The metric should take
        the following arguments: `rf`, `distance`, and `roi`
    interval: tuple of floats
        (minimum, maximum) of interval to search in pixels
    roi: rectangular region of interest (x1, y1, x2, y2)
        Region of interest of `field` for which the metric will be
        minimized. If not given, the entire `field` will be used.
    coarse_acc: float
        accuracy for determination of global minimum in pixels
    fine_acc: float
        accuracy for fine localization percentage of gradient change
    ret_gradient:
        return x and y values of computed gradient
    padding: bool
        perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionchanged:: 0.1.4
           improved padding value and padding location
    return_gradient: bool
        Deprecated, use ret_gradient instead!

    Returns
    -------
    af_field: ndarray
        Autofocused field
    af_dist: float
        Autofocusing distance
    gradients: list of tuples of ndarrays, optional
        Only returned if `ret_gradient` is specified

    """
    if return_gradient is not None:
        warnings.warn("`return_gradient` is deprecated, please use "
                      "`ret_gradient` instead!", DeprecationWarning)
        ret_gradient = return_gradient

    if roi is not None:
        assert len(roi) == len(rf.shape) * \
            2, "ROI must match field dimension"

    if padding is not None:
        warnings.warn("The `padding` argument is deprecated, please only "
                      "specify it in the Refocus interface!")
        if padding != rf.padding:
            raise ValueError("Padding must match padding in `rf`!")

    initshape = rf.shape
    shapelen = len(initshape)

    if roi is None:
        if shapelen == 2:
            roi = (0, 0, rf.shape[0], rf.shape[1])
        else:
            roi = (0, rf.shape[0])

    if shapelen == 2:
        roi_slice = (slice(roi[0], roi[2]), slice(roi[1], roi[3]))
    else:
        roi_slice = slice(roi[0], roi[1])

    ival = interval
    if ival[0] > ival[1]:
        ival = (ival[1], ival[0])
    # set coarse interval
    n = int(100 / coarse_acc)
    zc = np.linspace(ival[0], ival[1], n, endpoint=True)

    # initiate gradient vector
    gradc = np.zeros(zc.shape)
    for i in range(len(zc)):
        d = zc[i]
        gradc[i] = metric_func(rf, distance=d*rf.pixel_size, roi=roi_slice)

    minid = np.argmin(gradc)
    if minid == 0:
        zc -= zc[1] - zc[0]
        minid += 1
    if minid == len(zc) - 1:
        zc += zc[1] - zc[0]
        minid -= 1
    zf = 1*zc

    numfine = 10
    mingrad = gradc[minid]

    while True:
        gradf = np.zeros(numfine)
        ival = (zf[minid - 1], zf[minid + 1])
        zf = np.linspace(ival[0], ival[1], numfine)
        for i in range(len(zf)):
            d = zf[i]
            gradf[i] = metric_func(rf, distance=d*rf.pixel_size, roi=roi_slice)
        minid = np.argmin(gradf)
        if minid == 0:
            zf -= zf[1] - zf[0]
            minid += 1
        if minid == len(zf) - 1:
            zf += zf[1] - zf[0]
            minid -= 1
        if abs(mingrad - gradf[minid]) / 100 < fine_acc:
            break

    minid = np.argmin(gradf)
    af_field = rf.propagate(zf[minid]*rf.pixel_size)

    if ret_gradient:
        return af_field, zf[minid], [(zc, gradc), (zf, gradf)]
    return af_field, zf[minid]
