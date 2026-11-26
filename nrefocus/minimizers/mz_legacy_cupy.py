"""Batched CuPy implementation of the legacy autofocus search."""

import warnings

from .._ndarray_backend import xp
from .mz_legacy import LegacyDeprecationWarning


def minimize_legacy_cupy(rf, metric_func, interval, roi=None, coarse_acc=1,
                         fine_acc=.005, ret_grid=False, ret_field=False):
    """Minimize an autofocus metric with batched CuPy propagation.

    This follows the legacy minimizer's coarse scan, local fine scans, and
    boundary expansion. Each scan's candidate distances are propagated as a
    single batch, reducing per-candidate Python and GPU launch overhead.

    Parameters
    ----------
    rf :  nrefocus.iface.Refocus
        Refocusing interface backed by CuPy.
    metric_func : callable
        Registered metric function. It must return one score per candidate
        distance when given a distance vector. Spectrum does not support ROI.
    interval : tuple of float
        Inclusive search bounds in the same units accepted by ``rf``.
    roi : tuple of slices, optional
        Spatial region used for metric evaluation.
    coarse_acc : float
        Legacy coarse-grid density control. The grid has ``int(100 / value)``
        candidate distances, so the default evaluates 50 candidates and
        ``coarse_acc=1`` restores the previous 100-candidate density.
    fine_acc : float
        Stop when the change from the coarse minimum divided by 100 is below
        this threshold, matching the legacy minimizer's criterion.
    ret_grid : bool
        Include the coarse distance and metric arrays in the return value.
    ret_field : bool
        Include the field propagated to the selected distance.

    Returns
    -------
    distance : cupy.ndarray
        Estimated best focus distance(s)
    (d_grid, metrid_grid): ndarray
        Coarse grid search values (only if `ret_grid` is True)
    af_field: ndarray
        Autofocused field (only if `ret_field` is True)

    """
    if not xp.is_cupy():
        raise RuntimeError("minimize_legacy_cupy requires the CuPy backend.")
    warnings.warn("The 'legacy_cupy' minimizer follows the legacy "
                  "coarse-to-fine "
                  "search strategy. The legacy minimizer is deprecated; "
                  "use minimizer='lmfit_cupy' for GPU-native LM refinement "
                  "or 'lmfit' for lmfit/SciPy refinement.",
                  LegacyDeprecationWarning)

    n = int(100 / coarse_acc)
    zc = xp.linspace(interval[0], interval[1], n, endpoint=True)

    def evaluate(distances):
        # `distances` is a 1D CuPy array. RefocusCupy broadcasts it against
        # the spatial kernel and runs the inverse FFT across the whole
        # (candidate, y, x) batch. CuPy can process independent candidate
        # planes concurrently here, instead of launching one propagation per
        # candidate from a Python loop as 'legacy' minimizer does.
        return metric_func(rf, distance=distances, roi=roi)

    # Coarse candidates are independent, so evaluate them together with CuPy
    gradc = evaluate(zc)
    # Selecting a single best candidate requires its index on the host
    # .item() synchronizes the stream at this point.
    minid = int(xp.argmin(gradc).item())
    step = zc[1] - zc[0]
    if minid == 0:
        zc -= step
        minid = 1
    elif minid == len(zc) - 1:
        zc += step
        minid -= 1

    zf = zc.copy()
    numfine = 10
    mingrad = gradc[minid]
    while True:
        # Refinement batches are sequential because each new interval depends
        # on the previous batch's best index.
        zf = xp.linspace(zf[minid - 1], zf[minid + 1], numfine)
        gradf = evaluate(zf)
        minid = int(xp.argmin(gradf).item())
        step = zf[1] - zf[0]
        if minid == 0:
            zf -= step
            minid = 1
        elif minid == len(zf) - 1:
            zf += step
            minid -= 1
        if float((xp.abs(mingrad - gradf[minid]) / 100).item()) < fine_acc:
            break

    minid = int(xp.argmin(gradf).item())
    af_dist = zf[minid]
    result = [af_dist]
    if ret_grid:
        result.append((zc, gradc))
    if ret_field:
        result.append(rf.propagate(af_dist))
    return result[0] if len(result) == 1 else result
