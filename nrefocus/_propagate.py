#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import multiprocessing as mp
import numpy as np


__all__ = ["fft_propagate", "refocus", "refocus_stack"]


_cpu_count = mp.cpu_count()



def refocus(field, d, nm, res, method="helmholtz", num_cpus=1):
    """ Refocus a 1D or 2D field
    
    
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
        one of {
                "helmholtz" : the optical transfer function `exp(ikd)`,
                "fresnel"   : paraxial approximation `exp(ik²λd)`
               }
    num_cpus : int
        Not implemented. Only one CPU is used.
    
        
    Returns
    -------
    Electric field at `d`.
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

    vardict["fftfield"] = np.fft.fftn(field)
    
    return func(**vardict)


def refocus_stack(fieldstack, d, nm, res, method="helmholtz",
                  num_cpus=_cpu_count, copy=True):
    """ Refocus a stack of 1D or 2D fields
    
    
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
        one of {
                "helmholtz" : the optical transfer function `exp(ikd)`,
                "fresnel"   : paraxial approximation `exp(ik²λd)`
               }
    num_cpus : str
        Defines the number of CPUs to be used for refocusing.
    copy : bool
        If False, overwrites input stack.
        
        
    Returns
    -------
    Electric field stack at `d`.
    """
    
    func = refocus
    names = func.__code__.co_varnames[:func.__code__.co_argcount]
    
    loc = locals()
    vardict = dict()
    for name in names:
        if loc.has_key(name):
            vardict[name] = loc[name]
    # default keyword arguments
    func_def = func.__defaults__[::-1]
    
    
    # child processes should only use one cpu
    vardict["num_cpus"] = 1
    
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
        data = np.zeros([M]+list(result[0].shape), dtype=result[0].dtype)
    else:
        data = fieldstack
        
    for m in range(M):
        data[m] = result[m]

    return data
    

def fft_propagate(fftfield, d, nm, res, method="helmholtz",
                  ret_fft=False):
    """ Propagates a 1D or 2D Fourier transformed field
    
    
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
        one of {
                "helmholtz" : the optical transfer function `exp(ikd)`,
                "fresnel"   : paraxial approximation `exp(ik²λd)`
               }
    ret_fft : bool
        Do not perform an inverse Fourier transform and return the field
        in Fourier space.
    
                
    Returns
    -------
    Electric field at `d`. If `ret_fft` is True, then the
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



def fft_propagate_2d(fftfield, d, nm, res, method="helmholtz",
                     ret_fft=False):
    """ Propagate a 1D  Fourier transformed field in 2D
    
    
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
        one of {
                "helmholtz" : the optical transfer function `exp(ikd)`,
                "fresnel"   : paraxial approximation `exp(ik²λd)`
               }
    ret_fft : bool
        Do not perform an inverse Fourier transform and return the field
        in Fourier space.
    
                
    Returns
    -------
    Electric field at `d`. If `ret_fft` is True, then the
    Fourier transform of the electric field will be returned (faster).
    """
    assert len(fftfield.shape)==1, "Dimension of `fftfield` must be 1."
    l0 = d
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



def fft_propagate_3d(fftfield, d, nm, res, method="helmholtz",
                     ret_fft=False):
    """ Propagate a 2D  Fourier transformed field in 3D
    
    
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
        one of {
                "helmholtz" : the optical transfer function `exp(ikd)`,
                "fresnel"   : paraxial approximation `exp(ik²λd)`
               }
    ret_fft : bool
        Do not perform an inverse Fourier transform and return the field
        in Fourier space.
        
               
    Returns
    -------
    Electric field at `d`. If `ret_fft` is True, then the
    Fourier transform of the electric field will be returned (faster).
    """
    assert len(fftfield.shape)==2, "Dimension of `fftfield` must be 2."
    #if fftfield.shape[0] != fftfield.shape[1]:
    #    raise NotImplementedError("Field must be square shaped.")
    # free space propagator is
    # exp(i*sqrt(km**2-kx**2-ky**2)*l0)
    l0 = d
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
        return np.fft.ifft2(fftfield*fstemp)


def _refocus_wrapper(args):
    """Just calls autofocus with *args. Needed for multiprocessing pool.
    """
    return refocus(*args)
