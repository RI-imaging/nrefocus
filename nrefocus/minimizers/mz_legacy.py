import warnings

import numpy as np


class LegacyDeprecationWarning(DeprecationWarning):
    pass


def minimize_legacy(rf, metric_func, interval, roi=None, coarse_acc=1,
                    fine_acc=.005, ret_grid=False, ret_field=False):
    """Legacy minimizer

    Find the focus by minimizing the `metric` of an image.
    This is the implementation of the legacy nrefocus minimizer.

    Parameters
    ----------
    rf: nrefocus.iface.Refocus
        Refocus interface
    metric_func: callable
        metric called during minimization. The metric should take
        the following arguments: `rf`, `distance`, and `roi`
    interval: tuple of floats
        (minimum, maximum) of interval to search [m]
    roi: tuple of slices or np.ndarray
        Region of interest for which the metric will be minimized.
        If not given, the entire field will be used.
    coarse_acc: float
        accuracy for determination of global minimum in pixels;
        `coarse_acc=1` means that 100 fields are computed in the
        initial step; `coarse_acc=0.5` means 200 fields are computed
    fine_acc: float
        accuracy for fine localization percentage of gradient change
    ret_grid: bool
        return focus positions and metric values of the coarse
        grid search
    ret_field: bool
        return the optimal refocused field for user convenience

    Returns
    -------
    af_dist: float
        Autofocusing distance [m]
    (d_grid, metrid_grid): ndarray
        Coarse grid search values (only if `ret_grid` is True)
    af_field: ndarray
        Autofocused field (only if `ret_field` is True)
    """
    warnings.warn("The 'legacy' minimizer is deprecated, because it is "
                  "slower and not as accurate as the 'lmfit' minimizer! "
                  "Please only use 'legacy' to reproduce previous results.",
                  LegacyDeprecationWarning)

    ival = interval
    # set coarse interval
    n = int(100 / coarse_acc)
    zc = np.linspace(ival[0], ival[1], n, endpoint=True)

    # initiate gradient vector
    gradc = np.zeros(zc.shape)
    for i in range(len(zc)):
        d = zc[i]
        gradc[i] = metric_func(rf, distance=d, roi=roi)

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
            gradf[i] = metric_func(rf, distance=d, roi=roi)
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
    af_dist = zf[minid]

    ret_val = [af_dist]

    if ret_grid:
        ret_val.append((zc, gradc))

    if ret_field:
        ret_val.append(rf.propagate(af_dist))

    if len(ret_val) == 1:
        ret_val = ret_val[0]

    return ret_val
