import multiprocessing as mp
import numpy as np

from . import iface


__all__ = ["refocus", "refocus_stack"]


_cpu_count = mp.cpu_count()


def refocus(field, d, nm, res, method="helmholtz", padding=True):
    """Refocus a 1D or 2D field

    Parameters
    ----------
    field : 1d or 2d array
        1D or 2D background corrected electric field (Ex/BEx)
    d : float
        Distance to be propagated in pixels (negative for backwards)
    nm : float
        Refractive index of medium
    res : float
        Wavelenth in pixels
    method : str
        Defines the method of propagation;
        one of

            - "helmholtz" : the optical transfer function `exp(idkₘ(M-1))`
            - "fresnel"   : paraxial approximation `exp(idk²/kₘ)`
    padding : bool
        perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionadded:: 0.1.4

    Returns
    -------
    Electric field at `d`.

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

    # use a made-up pixel size so we can use the new `Refocus` interface
    pixel_size = 1e-6
    rf = rfcls(field=field,
               wavelength=res*pixel_size,
               pixel_size=pixel_size,
               medium_index=nm,
               distance=0,
               kernel=method,
               padding=padding
               )
    refoc = rf.propagate(distance=d*pixel_size)

    return refoc


def refocus_stack(fieldstack, d, nm, res, method="helmholtz",
                  num_cpus=_cpu_count, copy=True, padding=True):
    """Refocus a stack of 1D or 2D fields


    Parameters
    ----------
    fieldstack : 2d or 3d array
        Stack of 1D or 2D background corrected electric fields (Ex/BEx).
        The first axis iterates through the individual fields.
    d : float
        Distance to be propagated in pixels (negative for backwards)
    nm : float
        Refractive index of medium
    res : float
        Wavelenth in pixels
    method : str
        Defines the method of propagation;
        one of

            - "helmholtz" : the optical transfer function `exp(idkₘ(M-1))`
            - "fresnel"   : paraxial approximation `exp(idk²/kₘ)`

    num_cpus : int
        Defines the number of CPUs to be used for refocusing.
    copy : bool
        If False, overwrites input stack.
    padding : bool
        Perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionadded:: 0.1.4

    Returns
    -------
    Electric field stack at `d`.
    """
    func = refocus
    names = func.__code__.co_varnames[:func.__code__.co_argcount]

    loc = locals()
    vardict = dict()
    for name in names:
        if name in loc.keys():
            vardict[name] = loc[name]
    # default keyword arguments
    func_def = func.__defaults__[::-1]

    vardict["padding"] = padding

    M = fieldstack.shape[0]
    stackargs = list()

    # Create individual arglists for all fields
    for m in range(M):
        kwarg = vardict.copy()
        kwarg["field"] = fieldstack[m]
        # now we turn the kwarg into an arglist
        args = list()
        for i, a in enumerate(names[::-1]):
            # first set default
            if i < len(func_def):
                val = func_def[i]
            if a in kwarg:
                val = kwarg[a]
            args.append(val)
        stackargs.append(args[::-1])

    p = mp.Pool(num_cpus)
    result = p.map_async(_refocus_wrapper, stackargs).get()
    p.close()
    p.terminate()
    p.join()

    if copy:
        data = np.zeros(fieldstack.shape, dtype=result[0].dtype)
    else:
        data = fieldstack

    for m in range(M):
        data[m] = result[m]

    return data


def _refocus_wrapper(args):
    """Just calls autofocus with *args. Needed for multiprocessing pool.
    """
    return refocus(*args)
