import numpy as np


def average_gradient(data, *kwargs):
    """ Compute average gradient norm of an image
    """
    return np.average(np.array(np.gradient(data))**2)


def contrast_rms(data, *kwargs):
    """ Compute RMS contrast norm of an image
    """
    av = np.average(data, *kwargs)
    mal = 1 / (data.shape[0] * data.shape[1])
    return np.sqrt(mal * np.sum(np.square(data - av)))


def spectral(data, lambd, *kwargs):
    """ Compute spectral contrast of image

    Performs bandpass filtering in Fourier space according to optical
    limit of detection system, approximated by twice the wavelength.


    Parameters
    ----------
    data : 2d ndarray
        the image to compute the norm from
    lambd : float
        wavelength of the light in pixels

    """
    # Set up fast fourier transform
    # if not data.dtype == np.dtype(np.complex):
    #    data = np.array(data, dtype=np.complex)
    # fftplan = fftw3.Plan(data.copy(), None, nthreads = _ncores,
    #                     direction="forward", flags=_fftwflags)
    # fftdata = np.zeros(data.shape, dtype=np.complex)
    # fftplan.guru_execute_dft(data, fftdata)
    # fftw.destroy_plan(fftplan)
    fftdata = np.fft.fftn(data)

    # Filter Fourier transform
    fftdata[0, 0] = 0
    kx = 2 * np.pi * np.fft.fftfreq(data.shape[0]).reshape(1, -1)
    ky = 2 * np.pi * np.fft.fftfreq(data.shape[1]).reshape(-1, 1)
    kmax = (2 * np.pi) / (2 * lambd)
    fftdata[np.where(kx**2 + ky**2 > kmax**2)] = 0

    spec = np.sum(np.log(1 + np.abs(fftdata))) / np.sqrt(np.prod(data.shape))

    return spec
