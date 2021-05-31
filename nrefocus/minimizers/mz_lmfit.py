import copy

import lmfit


def residuals(params, metric_func, rf, roi):
    return metric_func(rf, distance=params["focus"].value, roi=roi)


def minimize_lmfit(rf, metric_func, interval, roi=None, ret_field=False,
                   **lmfitkw):
    """A minimizer that wraps lmfit

    Find the focus by minimizing the `metric` of an image
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
    ret_field:
        return the optimal refocused field for user convenience
    lmfitkw:
        Additional keyword arguments for :func:`lmfit.minimize
        <lmfit.minimizer.minimize>`. The default `method` is "leastsq".

    Returns
    -------
    af_field: ndarray
        Autofocused field
    af_dist: float
        Autofocusing distance [m]
    """
    lmfitkw = copy.deepcopy(lmfitkw)
    if "method" not in lmfitkw:
        lmfitkw["method"] = "leastsq"

    params = lmfit.Parameters()
    params.add("focus", value=0, min=interval[0], max=interval[1])

    res = lmfit.minimize(fcn=residuals,
                         params=params,
                         kws={"metric_func": metric_func,
                              "rf": rf,
                              "roi": roi},
                         **lmfitkw)
    af_dist = res.params["focus"].value

    ret_val = [af_dist]

    if ret_field:
        ret_val.append(rf.propagate(af_dist))

    if len(ret_val) == 1:
        ret_val = ret_val[0]

    return ret_val
