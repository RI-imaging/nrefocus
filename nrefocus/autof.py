import multiprocessing as mp
import numpy as np

from . import iface
from . import metrics
from .minimizers import minimize_legacy
from .propg import refocus_stack


__all__ = [
    "autofocus",
    "autofocus_stack",
]


_cpu_count = mp.cpu_count()


def autofocus(field, nm, res, ival, roi=None,
              metric="average gradient", padding=True,
              ret_d=False, ret_grad=False, num_cpus=1):
    """Numerical autofocusing of a field using the Helmholtz equation.

    Parameters
    ----------
    field: 1d or 2d ndarray
        Electric field is BG-Corrected, i.e. field = EX/BEx
    nm: float
        Refractive index of medium.
    res: float
        Size of wavelength in pixels.
    ival: tuple of floats
        Approximate interval to search for optimal focus in px.
    roi: rectangular region of interest (x1, y1, x2, y2)
        Region of interest of `field` for which the metric will be
        minimized. If not given, the entire `field` will be used.
    metric: str
        - "average gradient" : average gradient metric of amplitude
        - "rms contrast" : RMS contrast of phase data
        - "spectrum" : sum of filtered Fourier coefficients
    padding: bool
        Perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionchanged:: 0.1.4
           improved padding value and padding location
    ret_d: bool
        Return the autofocusing distance in pixels. Defaults to False.
    ret_grad: bool
        Return the computed gradients as a list.
    num_cpus: int
        Not implemented.


    Returns
    -------
    field, [d, [grad]]
    The focused field and optionally, the optimal focusing distance and
    the computed gradients.

    Notes
    -----
    This method uses :class:`nrefocus.RefocusNumpy` for refocusing
    of 2D fields. This is because the :func:`nrefocus.refocus_stack`
    function uses `async` which appears to not work with e.g.
    :mod:`pyfftw`.
    """
    fshape = len(field.shape)
    if fshape == 1:
        # 1D field
        rfcls = iface.RefocusNumpy1D
    elif fshape == 2:
        # 2D field
        rfcls = iface.RefocusNumpy
    else:
        raise AssertionError("Dimension of `field` must be 1 or 2.")

    metric_func = metrics.METRICS[metric]

    # use a made-up pixel size so we can use the new `Refocus` interface
    pixel_size = 1e-6
    rf = rfcls(field=field,
               wavelength=res*pixel_size,
               pixel_size=pixel_size,
               medium_index=nm,
               distance=0,
               kernel="helmholtz",
               padding=padding
               )

    field, d, grad = minimize_legacy(rf=rf,
                                     metric_func=metric_func,
                                     interval=ival,
                                     roi=roi,
                                     padding=padding)

    ret_list = [field]
    if ret_d:
        ret_list += [d]
    if ret_grad:
        ret_list += [grad]

    if len(ret_list) == 1:
        return ret_list[0]
    else:
        return tuple(ret_list)


def autofocus_stack(fieldstack, nm, res, ival, roi=None,
                    metric="average gradient", padding=True,
                    same_dist=False, ret_ds=False, ret_grads=False,
                    num_cpus=_cpu_count, copy=True):
    """Numerical autofocusing of a stack using the Helmholtz equation.

    Parameters
    ----------
    fieldstack: 2d or 3d ndarray
        Electric field is BG-Corrected, i.e. Field = EX/BEx
    nm: float
        Refractive index of medium.
    res: float
        Size of wavelength in pixels.
    ival: tuple of floats
        Approximate interval to search for optimal focus in px.
    roi: rectangular region of interest (x1, y1, x2, y2)
        Region of interest of `field` for which the metric will be
        minimized. If not given, the entire `field` will be used.
    metric: str
        see `autofocus_field`.
    padding: bool
        Perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionchanged:: 0.1.4
           improved padding value and padding location
    same_dist: bool
        Refocus entire sinogram with one distance.
    ret_ds: bool
        Return the autofocusing distances in pixels. Defaults to False.
        If sam_dist is True, still returns autofocusing distances
        of first pass. The used refocusing distance is the
        average.
    ret_grads: bool
        Return the computed gradients as a list.
    num_cpus: int
        Number of CPUs to use
    copy: bool
        If False, overwrites input array.


    Returns
    -------
    The focused field (and the refocussing distance + data if d is None)
    """
    dopt = list()
    grad = list()

    m = fieldstack.shape[0]

    # setup arguments
    stackargs = list()
    for s in range(m):
        stackargs.append([np.array(fieldstack[s], copy=copy), nm, res, ival,
                          roi, metric, padding, True, True, 1])
    # perform first pass
    p = mp.Pool(num_cpus)
    result = p.map_async(_autofocus_wrapper, stackargs).get()
    p.close()
    p.terminate()
    p.join()

    newstack = np.zeros(fieldstack.shape, dtype=fieldstack.dtype)

    for s in range(m):
        field, ds, gs = result[s]
        dopt.append(ds)
        grad.append(gs)
        newstack[s] = field

    # perform second pass if `same_dist` is True
    if same_dist:
        # find average dopt
        davg = np.average(dopt)
        newstack = refocus_stack(fieldstack, davg, nm, res,
                                 num_cpus=num_cpus, copy=copy,
                                 padding=padding)

    ret_list = [newstack]
    if ret_ds:
        ret_list += [dopt]
    if ret_grads:
        ret_list += [grad]

    if len(ret_list) == 1:
        return ret_list[0]
    else:
        return tuple(ret_list)


def _autofocus_wrapper(args):
    """Calls autofocus with *args. Needed for multiprocessing pool.
    """
    return autofocus(*args)
