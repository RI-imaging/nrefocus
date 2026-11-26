"""CuPy-native, one-parameter Levenberg-Marquardt autofocus minimizer."""

import math

from .._ndarray_backend import xp


def _minimize_lmfit_cupy(rf, metric_func, interval, roi=None,
                         coarse_step=2.0, max_iter=10, ftol=1e-8,
                         xtol=1e-8, fd_step=1e-4,
                         initial_damping=1e-6,
                         ret_grid=False, ret_field=False):
    """Minimize a registered metric with a CuPy-resident LM refinement.

    The search mirrors the project's ``lmfit`` strategy: sample the interval
    every ``coarse_step`` wavelengths, then refine around the best coarse
    candidate inside a local interval four wavelengths wide on either side.
    The coarse scan and finite-difference probes are evaluated as batches.
    LM state and convergence masks remain on the GPU for a bounded number of
    refinement iterations.

    Parameters
    ----------
    rf : Refocus
        Refocusing interface backed by CuPy.
    metric_func : callable
        Registered metric function. Spectrum does not support ROI.
    interval : tuple of float
        Inclusive distance bounds in the units used by ``rf``.
    roi : tuple of slices, optional
        Spatial region used for metric evaluation.
    coarse_step : float, default=2.0
        Spacing of the global scan in wavelengths.
    max_iter : int, default=10
        Maximum number of LM refinement iterations. Converged states remain
        masked until convergence is checked every four iterations.
    ftol : float, default=1e-8
        Relative improvement tolerance on the squared metric residual.
    xtol : float, default=1e-8
        Relative tolerance on the focus-distance update in wavelength units.
    fd_step : float, default=1e-4
        Finite-difference probe spacing in wavelengths.
    initial_damping : float, default=1e-6
        Starting Levenberg-Marquardt damping factor.
    ret_grid : bool, default=False
        Include the coarse distances and metric values in the result.
    ret_field : bool, default=False
        Include the field propagated to the selected distance.

    Returns
    -------
    distance : cupy.ndarray
        Zero-dimensional array containing the selected focus distance.
    tuple, optional
        Requested coarse-grid data and focused field are appended in that
        order.

    Raises
    ------
    RuntimeError
        If the active array backend is not CuPy.
    ValueError
        If the metric or numerical options are unsupported.
    """
    if not xp.is_cupy():
        raise RuntimeError("minimize_lmfit_cupy requires the CuPy backend.")
    if coarse_step <= 0 or fd_step <= 0 or initial_damping <= 0:
        raise ValueError("coarse_step, fd_step, and initial_damping must be "
                         "positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if ftol < 0 or xtol < 0:
        raise ValueError("ftol and xtol must be non-negative.")

    # Transfer only the two bounds, then keep candidate distances and all
    # optimization values on-device. Work in wavelength units like mz_lmfit.
    bounds = xp.asnumpy(xp.asarray(interval)).astype(float, copy=False)
    lower, upper = sorted(bounds / rf.wavelength)
    width = upper - lower

    if width <= coarse_step:
        coarse_host = [(lower + upper) / 2]
    else:
        steps = int(math.floor(width / coarse_step))
        coarse_host = [lower + i * coarse_step for i in range(steps + 1)]
        if coarse_host[-1] < upper:
            coarse_host.append(upper)
    coarse_grid = xp.asarray(coarse_host)

    def evaluate(wavelength_distances):
        """Evaluate one batch of candidate focus distances on the GPU."""
        return metric_func(
            rf, distance=wavelength_distances * rf.wavelength, roi=roi)

    coarse_metric = evaluate(coarse_grid)
    best_index = xp.argmin(coarse_metric)
    coarse_best = coarse_grid[best_index]
    residual = coarse_metric[best_index]

    # Constrain local refinement to the same +/- 4 wavelength neighborhood
    # used by the existing lmfit wrapper, clipped to the requested interval.
    local_lower = xp.maximum(lower, coarse_best - 4.0)
    local_upper = xp.minimum(upper, coarse_best + 4.0)

    # A three-point parabolic estimate gives LM a better starting point than
    # the coarse-grid node while remaining inside its local search interval.
    left_index = xp.maximum(best_index - 1, 0)
    right_index = xp.minimum(best_index + 1, coarse_grid.size - 1)
    y_left = coarse_metric[left_index]
    y_center = coarse_metric[best_index]
    y_right = coarse_metric[right_index]
    curvature = y_left - 2 * y_center + y_right
    safe_curvature = xp.where(
        xp.abs(curvature) > xp.finfo(coarse_grid.dtype).eps, curvature, 1)
    vertex_offset = (0.5 * (y_left - y_right) / safe_curvature *
                     (coarse_grid[right_index] - coarse_grid[left_index]) / 2)
    interior = ((best_index > 0) &
                (best_index < coarse_grid.size - 1) & (curvature > 0))
    z = xp.where(interior, coarse_best + vertex_offset, coarse_best)
    z = xp.clip(z, local_lower, local_upper)
    residual = evaluate(xp.asarray([z]))[0]

    damping = xp.asarray(initial_damping, dtype=coarse_grid.dtype)
    active = xp.asarray(True)

    for iteration in range(max_iter):
        h = xp.asarray(fd_step, dtype=coarse_grid.dtype)
        right = xp.minimum(local_upper, z + h)
        left = xp.maximum(local_lower, z - h)
        probe = xp.where(right > z, right, left)
        denominator = probe - z
        probe_residual = evaluate(xp.asarray([probe]))[0]
        jacobian = ((probe_residual - residual) /
                    xp.where(
                        xp.abs(denominator) > xp.finfo(coarse_grid.dtype).eps,
                        denominator, 1))

        # Scalar LM normal equation: delta = -(J.T J + lambda)^-1 J.T r.
        delta = -(jacobian * residual) / (jacobian ** 2 + damping)
        trial_z = xp.clip(z + delta, local_lower, local_upper)
        trial_residual = evaluate(xp.asarray([trial_z]))[0]

        current_cost = residual ** 2
        trial_cost = trial_residual ** 2
        accepted = active & (trial_cost < current_cost)

        next_z = xp.where(accepted, trial_z, z)
        next_residual = xp.where(accepted, trial_residual, residual)
        next_cost = next_residual ** 2
        step_size = xp.abs(next_z - z)
        cost_change = xp.abs(current_cost - next_cost)

        converged = accepted & (
                (step_size <= xtol * (1 + xp.abs(next_z))) |
                (cost_change <= ftol * xp.maximum(
                    current_cost, xp.finfo(coarse_grid.dtype).eps))
        )
        z = next_z
        residual = next_residual
        damping = xp.where(accepted,
                           xp.maximum(damping / 3, 1e-12),
                           xp.minimum(damping * 10, 1e12))
        active = active & ~converged

        # Converged candidates need no more field propagations. Checking every
        # few iterations avoids both the old fixed-iteration waste and a host
        # synchronization after every individual LM update.
        if (iteration + 1) % 4 == 0 or iteration + 1 == max_iter:
            if not bool(active.item()):
                break

    af_distance = z * rf.wavelength
    result = [af_distance]
    if ret_grid:
        result.append((coarse_grid * rf.wavelength, coarse_metric))
    if ret_field:
        result.append(rf.propagate(af_distance))
    return result[0] if len(result) == 1 else result


def minimize_lmfit_cupy(rf, metric_func, interval, roi=None,
                        coarse_step=2.0, max_iter=10, ftol=1e-8,
                        xtol=1e-8, fd_step=1e-4,
                        initial_damping=1e-6,
                        ret_grid=False, ret_field=False):
    """Run the CuPy LM minimizer with finite-difference Jacobians."""
    return _minimize_lmfit_cupy(
        rf, metric_func, interval, roi=roi, coarse_step=coarse_step,
        max_iter=max_iter, ftol=ftol, xtol=xtol, fd_step=fd_step,
        initial_damping=initial_damping, ret_grid=ret_grid,
        ret_field=ret_field)
