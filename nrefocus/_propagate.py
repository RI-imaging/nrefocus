from __future__ import division, print_function
import numpy as np

def free_space_propagate_1d(fftfield, distance, nm, res,
                            method="helmholtz"):
    """ Propagate a 1D field a certain distance in pixels
    
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
    #fstemp[np.where(np.isnan(fstemp))] = 0
    
    return np.fft.ifft(fftfield*fstemp)
    # prepare inverse Fourier transform
    # Set up fast fourier transform
    #if fftplan is None:
    #    fftplan = fftw3.Plan(fftfield.copy(), None, nthreads = _ncores,
    #                         direction="backward", flags=_fftwflags)
    #field = np.zeros(fftfield.shape, dtype=np.complex)
    #fftplan.guru_execute_dft(fftfield*fstemp, field)
    #fftw3.destroy_plan(fftplan)
    #
    #return field/np.prod(field.shape)



def free_space_propagate_2d(fftfield, distance, nm, res,
                            method="helmholtz"):
    """ Propagate a 2D field a certain distance in pixels
    
    Parameters
    ----------
    fftfield : 2d array
        Fourier transform of 2D Electric field component
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
    return np.fft.ifft2(fftfield*fstemp)
