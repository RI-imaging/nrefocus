import copy

import numpy as np
import lmfit


def residuals(params, metric_func, rf, roi):
    return metric_func(rf,
                       distance=params["focus_wl"].value * rf.wavelength,
                       roi=roi)


def minimize_lmfit(rf, metric_func, interval, roi=None, lmfitkw=None,
                   ret_grid=False, ret_field=False):
    """A minimizer that wraps lmfit

    Find the focus by minimizing the `metric` of an image
    A coarse grid search over `interval` with step size
    of `2*rf.wavelength` is performed, followed by a
    "regular" minimization for the best candidate.

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
    lmfitkw:
        Additional keyword arguments for :func:`lmfit.minimize
        <lmfit.minimizer.minimize>` used in the fine grid search.
        The default `method` is "leastsq".
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
    if lmfitkw is None:
        lmfitkw = {}
    lmfitkw = copy.deepcopy(lmfitkw)
    if "method" not in lmfitkw:
        lmfitkw["method"] = "leastsq"

    # normalize fitting interval with wavelength
    interval = np.array(interval, copy=True) / rf.wavelength

    # brute step size is two wavelengths
    brute_step = 2

    # initialize fitter
    params_brute = lmfit.Parameters()
    params_brute.add("focus_wl",
                     value=np.mean(interval),
                     min=interval[0],
                     max=interval[1],
                     brute_step=brute_step)

    fitter = lmfit.Minimizer(
        userfcn=residuals,
        params=params_brute,
        fcn_kws={"metric_func": metric_func,
                 "rf": rf,
                 "roi": roi},
    )

    if np.ptp(interval) <= brute_step:
        # skip the brute step (no step definable)
        fine_params = params_brute
    else:
        # coarse grid search (keep only best result, increasing `keep` does
        # not help)
        res_brute = fitter.minimize(method="brute", keep=1)
        # refine with regular minimizer and new search interval
        fine_params = copy.deepcopy(res_brute).params

    fine_params["focus_wl"].min = max(interval[0],
                                      fine_params["focus_wl"] - 4)
    fine_params["focus_wl"].max = min(interval[1],
                                      fine_params["focus_wl"] + 4)
    res_fine = fitter.minimize(params=fine_params, **lmfitkw)

    # extract focusing distance [m]
    af_dist = res_fine.params["focus_wl"].value * rf.wavelength

    # return values
    ret_val = [af_dist]

    if ret_grid:
        ret_val.append((
            # Representation of the evaluation grid.
            res_brute.brute_grid * rf.wavelength,
            # Function values at each point of the evaluation grid
            res_brute.brute_Jout))

    if ret_field:
        ret_val.append(rf.propagate(af_dist))

    if len(ret_val) == 1:
        ret_val = ret_val[0]

    return ret_val
