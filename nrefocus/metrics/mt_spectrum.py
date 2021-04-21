import numpy as np


def metric_spectrum(rfi, distance, roi=None, **kwargs):
    """Compute spectral contrast

    Performs bandpass filtering in Fourier space according to optical
    limit of detection system, approximated by twice the wavelength.
    """
    if roi is not None:
        raise ValueError("Spectral method does not support ROIs!")

    wavelength_px = rfi.wavelength / rfi.pixel_size
    kernel = rfi.get_kernel(distance)
    fftdata = rfi.fft_origin * kernel

    # Filter Fourier transform
    fftdata[0, 0] = 0
    kx = 2 * np.pi * np.fft.fftfreq(fftdata.shape[0]).reshape(1, -1)
    ky = 2 * np.pi * np.fft.fftfreq(fftdata.shape[1]).reshape(-1, 1)
    kmax = (2 * np.pi) / (2 * wavelength_px)
    fftdata[np.where(kx**2 + ky**2 > kmax**2)] = 0

    spec = np.sum(np.log(1 + np.abs(fftdata))) / np.sqrt(np.prod(rfi.shape))

    return spec
