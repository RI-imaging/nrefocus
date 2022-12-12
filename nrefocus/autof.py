import copy
import multiprocessing as mp
import numpy as np

from . import iface
from .propg import refocus_stack


__all__ = [
    "autofocus",
    "autofocus_stack",
]


_cpu_count = mp.cpu_count()


def autofocus(field, nm, res, ival, roi=None,
              metric="average gradient", minimizer="lmfit",
              minimizer_kwargs=None, padding=True, num_cpus=1):
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
        - "std gradient" : standard deviation of gradient metric of amplitude
        - "med gradient" : median gradient metric of amplitude
    minimizer: str
        - "lmfit" : lmfit-based minimizer
        - "legacy" : only use for reproducing old results
    minimizer_kwargs: dict
        Optional keyword arguments to the `minimizer` function
    padding: bool
        Perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionchanged:: 0.1.4
           improved padding value and padding location
    num_cpus: int
        Not implemented.


    Returns
    -------
    d, field [, other]:
        The focusing distance, the field, and optionally any other
        data returned by the minimizer (specify via `minimizer_kwargs`).

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

    if minimizer_kwargs is None:
        minimizer_kwargs = {}
    else:
        minimizer_kwargs = copy.deepcopy(minimizer_kwargs)

    # use a made-up pixel size so we can use the new `Refocus` interface
    pixel_size = 1
    rf = rfcls(field=field,
               wavelength=res*pixel_size,
               pixel_size=pixel_size,
               medium_index=nm,
               distance=0,
               kernel="helmholtz",
               padding=padding
               )

    data = rf.autofocus(metric=metric,
                        minimizer=minimizer,
                        interval=np.array(ival)*rf.pixel_size,
                        roi=roi,
                        minimizer_kwargs=minimizer_kwargs,
                        ret_grid=False,
                        ret_field=True,
                        )

    return data


def autofocus_stack(fieldstack, nm, res, ival, roi=None,
                    metric="average gradient", minimizer="lmfit",
                    minimizer_kwargs=None, padding=True, same_dist=False,
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
    minimizer: str
        - "lmfit" : lmfit-based minimizer
        - "legacy" : only use for reproducing old results
    minimizer_kwargs: dict
        Optional keyword arguments to the `minimizer` function
    padding: bool
        Perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionchanged:: 0.1.4
           improved padding value and padding location
    same_dist: bool
        Refocus entire sinogram with one distance.
    num_cpus: int
        Number of CPUs to use
    copy: bool
        If False, overwrites input array.


    Returns
    -------
    dopt: float or list of float
        The focusing distance(s) (only one value if `same_dist`)
    field_stack: np.ndarray
        The refocused field stack
    """
    dopt = list()

    m = fieldstack.shape[0]

    # setup arguments
    stackargs = list()
    for s in range(m):
        stackargs.append([np.array(fieldstack[s], copy=copy), nm, res, ival,
                          roi, metric, minimizer, minimizer_kwargs,
                          padding, 1])
    # perform first pass
    p = mp.Pool(num_cpus)
    result = p.map_async(_autofocus_wrapper, stackargs).get()
    p.close()
    p.terminate()
    p.join()

    newstack = np.zeros(fieldstack.shape, dtype=fieldstack.dtype)

    for s in range(m):
        if isinstance(result[s], list):
            dopt.append(result[s][0])
            newstack[s] = result[s][1]

    # perform second pass if `same_dist` is True
    if same_dist:
        # find average dopt
        davg = np.average(dopt)
        newstack = refocus_stack(fieldstack, davg, nm, res,
                                 num_cpus=num_cpus, copy=copy,
                                 padding=padding)

        return davg, newstack
    else:
        return dopt, newstack


def _autofocus_wrapper(args):
    """Calls autofocus with *args. Needed for multiprocessing pool.
    """
    return autofocus(*args)
