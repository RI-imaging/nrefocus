#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

__all__ = ["refocus", "fft_propagate"]



def refocus(field, distance, nm, res, method="helmholtz"):
    """ Propagate a 1D or 2D field a certain distance in pixels
    
    Parameters
    ----------
    fftfield : 1d array
        Fourier transform of 1D Electric field component
    distance : float
        distance to be propagated in pixels (negative for backwards)
    nm : float
        refractive index of medium
    res : float
        wavelenth in pixels
    method : str
        defines the method of propagation;
        one of {
                "helmholtz" : the optical transfer function `exp(ikd)`,
                "fresnel"   : paraxial approximation `exp(ik²λd)`
               }
        
    Returns
    -------
    Electric field at that distance
    """
    # FFT of field
    fshape = len(field.shape)
    assert fshape in [1,2], "Dimension of `field` must be 1 or 2."

    func = fft_propagate
    names = func.__code__.co_varnames[:func.__code__.co_argcount]
    
    loc = locals()
    vardict = dict()

    for name in names:
        if loc.has_key(name):
            vardict[name] = loc[name]

    vardict["fftfield"] = np.fft.fft(field)
    
    return func(**vardict)



def fft_propagate(fftfield, distance, nm, res, method="helmholtz",
                  ret_fft=False):
    """ Propagates a 1D or 2D Fourier transformed field
    
    Parameters
    ----------
    fftfield : 1-dimensional or 2-dimensional ndarray
        Fourier transform of 1D Electric field component
    distance : float
        Distance to be propagated in pixels (negative for backwards)
    nm : float
        Refractive index of medium
    res : float
        Wavelength in pixels
    method : str
        Defines the method of propagation;
        one of {
                "helmholtz" : the optical transfer function `exp(ikd)`,
                "fresnel"   : paraxial approximation `exp(ik²λd)`
               }
    ret_fft : bool
        Do not perform an inverse Fourier transform and return the field
        in Fourier space.
    
                
    Returns
    -------
    Electric field at that distance. If `ret_fft` is True, then the
    Fourier transform of the electric field will be returned (faster).
    """
    fshape = len(fftfield.shape)
    assert fshape in [1,2], "Dimension of `fftfield` must be 1 or 2."
    
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



def fft_propagate_2d(fftfield, distance, nm, res, method="helmholtz",
                     ret_fft=False):
    """ Propagate a 1D  Fourier transformed field in 2D
    
    Parameters
    ----------
    fftfield : 1d array
        Fourier transform of 1D Electric field component
    distance : float
        Distance to be propagated in pixels (negative for backwards)
    nm : float
        Refractive index of medium
    res : float
        Wavelength in pixels
    method : str
        Defines the method of propagation;
        one of {
                "helmholtz" : the optical transfer function `exp(ikd)`,
                "fresnel"   : paraxial approximation `exp(ik²λd)`
               }
    ret_fft : bool
        Do not perform an inverse Fourier transform and return the field
        in Fourier space.
    
                
    Returns
    -------
    Electric field at that distance. If `ret_fft` is True, then the
    Fourier transform of the electric field will be returned (faster).
    """
    assert len(fftfield.shape)==1, "Dimension of `fftfield` must be 1."
    l0 = distance
    km = (2*np.pi*nm)/res
    kx = np.fft.fftfreq(len(fftfield))*2*np.pi
    # free space propagator is
    if method == "helmholtz":
        # exp(i*sqrt(km²-kx²)*l0)
        # Also subtract incoming plane wave. We are only considering
        # the scattered field here.
        root_km = km**2-kx**2
        rt0 = (root_km > 0)
        # multiply by rt0 (filter in Fourier space)
        fstemp = np.exp(1j * (np.sqrt(root_km*rt0)-km) * l0)*rt0
    elif method == "fresnel":
        # exp(i*lambda*kx²*PI*l0)
        #fstemp = np.exp(1j * res * kx**2 * np.pi * l0)
        fstemp = np.exp(-1j * res/nm * kx**2 * np.pi * l0 /(2*np.pi)**2)
    else:
        raise ValueError("Unknown method: {}".format(method))

    if ret_fft:
        return fftfield*fstemp
    else:
        return np.fft.ifft(fftfield*fstemp)



def fft_propagate_3d(fftfield, distance, nm, res, method="helmholtz",
                     ret_fft=False):
    """ Propagate a 2D  Fourier transformed field in 3D
    
    Parameters
    ----------
    fftfield : 2d array
        Fourier transform of 2D Electric field component
    distance : float
        Distance to be propagated in pixels (negative for backwards)
    nm : float
        Refractive index of medium
    res : float
        Wavelength in pixels
    method : str
        Defines the method of propagation;
        one of {
                "helmholtz" : the optical transfer function `exp(ikd)`,
                "fresnel"   : paraxial approximation `exp(ik²λd)`
               }
    ret_fft : bool
        Do not perform an inverse Fourier transform and return the field
        in Fourier space.
        
               
    Returns
    -------
    Electric field at that distance. If `ret_fft` is True, then the
    Fourier transform of the electric field will be returned (faster).
    """
    assert len(fftfield.shape)==1, "Dimension of `fftfield` must be 1."
    #if fftfield.shape[0] != fftfield.shape[1]:
    #    raise NotImplementedError("Field must be square shaped.")
    # free space propagator is
    # exp(i*sqrt(km**2-kx**2-ky**2)*l0)
    l0 = distance
    km = (2*np.pi*nm)/res
    kx = (np.fft.fftfreq(fftfield.shape[0])*2*np.pi).reshape(-1,1)
    ky = (np.fft.fftfreq(fftfield.shape[1])*2*np.pi).reshape(1,-1)
    if method == "helmholtz":
        # exp(i*sqrt(km²-kx²-ky²)*l0)
        root_km = km**2-kx**2-ky**2
        rt0 = (root_km > 0)
        # multiply by rt0 (filter in Fourier space)
        fstemp = np.exp(1j * (np.sqrt(root_km*rt0)-km) * l0)*rt0
    elif method == "fresnel":
        # exp(i*lambda*(kx²+ky²)*PI*l0)
        #fstemp = np.exp(1j * res * (kx**2+ky**2) * np.pi * l0)
        fstemp = np.exp(-1j * res/nm * (kx**2+ky**2) * np.pi * l0 /(2*np.pi)**2)
    else:
        raise ValueError("Unknown method: {}".format(method))
    #fstemp[np.where(np.isnan(fstemp))] = 0
    # Also subtract incoming plane wave. We are only considering
    # the scattered field here.
    if ret_fft:
        return fftfield*fstemp
    else:
        return np.fft.ifft(fftfield*fstemp)
