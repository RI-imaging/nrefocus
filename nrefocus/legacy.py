"""Deprecated methods"""
import warnings

import numpy as np


def fft_propagate(fftfield, d, nm, res, method="helmholtz", ret_fft=False):
    """Propagates a 1D or 2D Fourier transformed field


    Parameters
    ----------
    fftfield : 1-dimensional or 2-dimensional ndarray
        Fourier transform of 1D Electric field component
    d : float
        Distance to be propagated in pixels (negative for backwards)
    nm : float
        Refractive index of medium
    res : float
        Wavelength in pixels
    method : str
        Defines the method of propagation;
        one of

            - "helmholtz" : the optical transfer function `exp(idkₘ(M-1))`
            - "fresnel"   : paraxial approximation `exp(-idk²/(2kₘ))`

    ret_fft : bool
        Do not perform an inverse Fourier transform and return the field
        in Fourier space.


    Returns
    -------
    Electric field at `d`. If `ret_fft` is True, then the
    Fourier transform of the electric field will be returned (faster).
    """
    warnings.warn("Please don't use the `fft_propagate` method. You will "
                  "find the `nrefocus.Refocus*` classes much more useful "
                  "and more convenient to work with!", DeprecationWarning)
    fshape = len(fftfield.shape)
    assert fshape in [1, 2], "Dimension of `fftfield` must be 1 or 2."

    if fshape == 1:
        func = fft_propagate_2d
    else:
        func = fft_propagate_3d

    names = func.__code__.co_varnames[:func.__code__.co_argcount]

    loc = locals()
    vardict = dict()
    for name in names:
        vardict[name] = loc[name]

    return func(**vardict)


def fft_propagate_2d(fftfield, d, nm, res, method="helmholtz",
                     ret_fft=False):
    """Propagate a 1D  Fourier transformed field in 2D


    Parameters
    ----------
    fftfield : 1d array
        Fourier transform of 1D Electric field component
    d : float
        Distance to be propagated in pixels (negative for backwards)
    nm : float
        Refractive index of medium
    res : float
        Wavelength in pixels
    method : str
        Defines the method of propagation;
        one of

            - "helmholtz" : the optical transfer function `exp(idkₘ(M-1))`
            - "fresnel"   : paraxial approximation `exp(-idk²/(2kₘ))`

    ret_fft : bool
        Do not perform an inverse Fourier transform and return the field
        in Fourier space.


    Returns
    -------
    Electric field at `d`. If `ret_fft` is True, then the
    Fourier transform of the electric field will be returned (faster).
    """
    assert len(fftfield.shape) == 1, "Dimension of `fftfield` must be 1."
    km = (2 * np.pi * nm) / res
    kx = np.fft.fftfreq(len(fftfield)) * 2 * np.pi

    print(nm, res, d)
    # free space propagator is
    if method == "helmholtz":
        # exp(i*sqrt(km²-kx²)*d)
        # Also subtract incoming plane wave. We are only considering
        # the scattered field here.
        root_km = km**2 - kx**2
        rt0 = (root_km > 0)
        # multiply by rt0 (filter in Fourier space)
        fstemp = np.exp(1j * (np.sqrt(root_km * rt0) - km) * d) * rt0
    elif method == "fresnel":
        # exp(i*d*(km-kx²/(2*km))
        # fstemp = np.exp(-1j * d * (kx**2/(2*km)))
        fstemp = np.exp(-1j * d * (kx**2/(2*km)))
    else:
        raise ValueError("Unknown method: {}".format(method))

    if ret_fft:
        return fftfield * fstemp
    else:
        return np.fft.ifft(fftfield * fstemp)


def fft_propagate_3d(fftfield, d, nm, res, method="helmholtz",
                     ret_fft=False):
    """Propagate a 2D  Fourier transformed field in 3D


    Parameters
    ----------
    fftfield : 2d array
        Fourier transform of 2D Electric field component
    d : float
        Distance to be propagated in pixels (negative for backwards)
    nm : float
        Refractive index of medium
    res : float
        Wavelength in pixels
    method : str
        Defines the method of propagation;
        one of

            - "helmholtz" : the optical transfer function `exp(idkₘ(M-1))`
            - "fresnel"   : paraxial approximation `exp(-idk²/(2kₘ))`

    ret_fft : bool
        Do not perform an inverse Fourier transform and return the field
        in Fourier space.


    Returns
    -------
    Electric field at `d`. If `ret_fft` is True, then the
    Fourier transform of the electric field will be returned (faster).
    """
    assert len(fftfield.shape) == 2, "Dimension of `fftfield` must be 2."
    # if fftfield.shape[0] != fftfield.shape[1]:
    #    raise NotImplementedError("Field must be square shaped.")
    # free space propagator is
    # exp(i*sqrt(km**2-kx**2-ky**2)*d)
    km = (2 * np.pi * nm) / res
    kx = (np.fft.fftfreq(fftfield.shape[0]) * 2 * np.pi).reshape(-1, 1)
    ky = (np.fft.fftfreq(fftfield.shape[1]) * 2 * np.pi).reshape(1, -1)
    if method == "helmholtz":
        # exp(i*sqrt(km²-kx²-ky²)*d)
        root_km = km**2 - kx**2 - ky**2
        rt0 = (root_km > 0)
        # multiply by rt0 (filter in Fourier space)
        fstemp = np.exp(1j * (np.sqrt(root_km * rt0) - km) * d) * rt0
    elif method == "fresnel":
        # exp(i*d*(km-(kx²+ky²)/(2*km))
        fstemp = np.exp(-1j * d * (kx**2 + ky**2)/(2*km))
    else:
        raise ValueError("Unknown method: {}".format(method))
    # fstemp[np.where(np.isnan(fstemp))] = 0
    # Also subtract incoming plane wave. We are only considering
    # the scattered field here.
    if ret_fft:
        return fftfield * fstemp
    else:
        return np.fft.ifft2(fftfield * fstemp)
