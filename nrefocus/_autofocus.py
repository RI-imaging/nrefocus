import multiprocessing as mp
import numpy as np

from . import metrics
from . import pad
from ._propagate import fft_propagate, refocus_stack


__all__ = [
    "autofocus",
    "autofocus_stack",
    "minimize_metric",
]


_cpu_count = mp.cpu_count()


def autofocus(field, nm, res, ival, roi=None,
              metric="average gradient", padding=True,
              ret_d=False, ret_grad=False, num_cpus=1):
    """Numerical autofocusing of a field using the Helmholtz equation.


    Parameters
    ----------
    field : 1d or 2d ndarray
        Electric field is BG-Corrected, i.e. field = EX/BEx
    nm : float
        Refractive index of medium.
    res : float
        Size of wavelength in pixels.
    ival : tuple of floats
        Approximate interval to search for optimal focus in px.
    roi : rectangular region of interest (x1, y1, x2, y2)
        Region of interest of `field` for which the metric will be
        minimized. If not given, the entire `field` will be used.
    metric : str
        - "average gradient" : average gradient metric of amplitude
        - "rms contrast" : RMS contrast of phase data
        - "spectrum" : sum of filtered Fourier coefficients
    padding: bool
        Perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionchanged:: 0.1.4
           improved padding value and padding location
    red_d : bool
        Return the autofocusing distance in pixels. Defaults to False.
    red_grad : bool
        Return the computed gradients as a list.
    num_cpus : int
        Not implemented.


    Returns
    -------
    field, [d, [grad]]
    The focused field and optionally, the optimal focusing distance and
    the computed gradients.
    """
    if metric == "average gradient":
        def metric_func(x): return metrics.average_gradient(np.abs(x))
    elif metric == "rms contrast":
        def metric_func(x): return -metrics.contrast_rms(np.angle(x))
    elif metric == "spectrum":
        def metric_func(x): return metrics.spectral(np.abs(x), res)
    else:
        raise ValueError("No such metric: {}".format(metric))

    field, d, grad = minimize_metric(field, metric_func, nm, res, ival,
                                     roi=roi, padding=padding)

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
    fieldstack : 2d or 3d ndarray
        Electric field is BG-Corrected, i.e. Field = EX/BEx
    nm : float
        Refractive index of medium.
    res : float
        Size of wavelength in pixels.
    ival : tuple of floats
        Approximate interval to search for optimal focus in px.
    metric : str
        see `autofocus_field`.
    padding : bool
        Perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionchanged:: 0.1.4
           improved padding value and padding location
    ret_dopt : bool
        Return optimized distance and gradient plotting data.
    same_dist : bool
        Refocus entire sinogram with one distance.
    red_ds : bool
        Return the autofocusing distances in pixels. Defaults to False.
        If sam_dist is True, still returns autofocusing distances
        of first pass. The used refocusing distance is the
        average.
    red_grads : bool
        Return the computed gradients as a list.
    copy : bool
        If False, overwrites input array.


    Returns
    -------
    The focused field (and the refocussing distance + data if d is None)
    """
    dopt = list()
    grad = list()

    M = fieldstack.shape[0]

    # setup arguments
    stackargs = list()
    for s in range(M):
        stackargs.append([fieldstack[s].copy(copy), nm, res, ival,
                          roi, metric, padding, True, True, 1])
    # perform first pass
    p = mp.Pool(num_cpus)
    result = p.map_async(_autofocus_wrapper, stackargs).get()
    p.close()
    p.terminate()
    p.join()
    # result = []
    # for arg in stackargs:
    #    result += _autofocus_wrapper(arg)

    newstack = np.zeros(fieldstack.shape, dtype=fieldstack.dtype)

    for s in range(M):
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


def minimize_metric(field, metric_func, nm, res, ival, roi=None,
                    coarse_acc=1, fine_acc=.005,
                    return_gradient=True, padding=True):
    """Find the focus by minimizing the `metric` of an image

    Parameters
    ----------
    field : 2d array
        electric field
    metric_func : callable
        some metric to be minimized
    ival : tuple of floats
        (minimum, maximum) of interval to search in pixels
    nm : float
        RI of medium
    res : float
        wavelength in pixels
    roi : rectangular region of interest (x1, y1, x2, y2)
        Region of interest of `field` for which the metric will be
        minimized. If not given, the entire `field` will be used.
    coarse_acc : float
        accuracy for determination of global minimum in pixels
    fine_acc : float
        accuracy for fine localization percentage of gradient change
    return_gradient:
        return x and y values of computed gradient
    padding : bool
        perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionchanged:: 0.1.4
           improved padding value and padding location
    """
    if roi is not None:
        assert len(roi) == len(field.shape) * \
            2, "ROI must match field dimension"

    initshape = field.shape
    Fshape = len(initshape)
    propfunc = fft_propagate

    if roi is None:
        if Fshape == 2:
            roi = (0, 0, field.shape[0], field.shape[1])
        else:
            roi = (0, field.shape[0])

    roi = 1*np.array(roi)

    if padding:
        # Pad with correct complex number
        field = pad.pad_add(field)

    if ival[0] > ival[1]:
        ival = (ival[1], ival[0])
    # set coarse interval
    # coarse_acc = int(np.ceil(ival[1]-ival[0]))/100
    N = 100 / coarse_acc
    zc = np.linspace(ival[0], ival[1], N, endpoint=True)

    # compute fft of field
    fftfield = np.fft.fftn(field)

    # fftplan = fftw3.Plan(fftfield.copy(), None, nthreads = _ncores,
    #                     direction="backward", flags=_fftwflags)

    # initiate gradient vector
    gradc = np.zeros(zc.shape)
    for i in range(len(zc)):
        d = zc[i]
        # fsp = propfunc(fftfield, d, nm, res, fftplan=fftplan)
        fsp = propfunc(fftfield, d, nm, res)
        if Fshape == 2:
            gradc[i] = metric_func(fsp[roi[0]:roi[2], roi[1]:roi[3]])
        else:
            gradc[i] = metric_func(fsp[roi[0]:roi[1]])

    minid = np.argmin(gradc)
    if minid == 0:
        zc -= zc[1] - zc[0]
        minid += 1
    if minid == len(zc) - 1:
        zc += zc[1] - zc[0]
        minid -= 1
    zf = 1*zc
    gradf = 1 * gradc

    numfine = 10
    mingrad = gradc[minid]

    while True:
        gradf = np.zeros(numfine)
        ival = (zf[minid - 1], zf[minid + 1])
        zf = np.linspace(ival[0], ival[1], numfine)
        for i in range(len(zf)):
            d = zf[i]
            fsp = propfunc(fftfield, d, nm, res)
            if Fshape == 2:
                gradf[i] = metric_func(fsp[roi[0]:roi[2], roi[1]:roi[3]])
            else:
                gradf[i] = metric_func(fsp[roi[0]:roi[1]])
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
    fsp = propfunc(fftfield, zf[minid], nm, res)

    if padding:
        fsp = pad.pad_rem(fsp)

    if return_gradient:
        return fsp, zf[minid], [(zc, gradc), (zf, gradf)]
    return fsp, zf[minid]


def _autofocus_wrapper(args):
    """Calls autofocus with *args. Needed for multiprocessing pool.
    """
    return autofocus(*args)
