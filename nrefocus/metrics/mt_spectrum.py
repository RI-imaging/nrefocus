import cupy as cp


class MetricSpectrumValueError(ValueError):
    pass


def metric_spectrum(rfi, distance, roi=None, **kwargs):
    """Compute spectral contrast

    Performs bandpass filtering in Fourier space according to optical
    limit of detection system, approximated by twice the wavelength.
    """
    if roi is not None:
        raise MetricSpectrumValueError(
            "Spectral method does not support ROIs!")

    wavelength_px = rfi.wavelength / rfi.pixel_size
    kernel = rfi.get_kernel(distance)
    fftdata = rfi.fft_origin * kernel

    # Filter Fourier transform
    fftdata[0, 0] = 0
    kx = 2 * cp.pi * cp.fft.fftfreq(fftdata.shape[0]).reshape(1, -1)
    ky = 2 * cp.pi * cp.fft.fftfreq(fftdata.shape[1]).reshape(-1, 1)
    kmax = (2 * cp.pi) / (2 * wavelength_px)
    fftdata[cp.where(kx ** 2 + ky ** 2 > kmax ** 2)] = 0

    spec = cp.sum(cp.log(1 + cp.abs(fftdata))) / cp.sqrt(
        cp.prod(cp.array(rfi.shape))
    )

    return spec
