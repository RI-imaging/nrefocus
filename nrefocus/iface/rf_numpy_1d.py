import numpy as np

from .. import pad


class RefocusNumpy1D:
    def __init__(self, field, wavelength, pixel_size, medium_index=1.3333,
                 distance=0, padding=True):
        """Refocus a 1D field with numpy

        Parameters
        ----------
        field: 1d complex-valued ndarray
            Input 1D field to be refocused
        wavelength: float
            Wavelength of the used light [m]
        pixel_size: float
            Pixel size of the input image [m]
        medium_index: float
            Refractive index of the medium, defaults to water
            (1.3333 at 21.5°C)
        distance: float
            Initial focusing distance [m]
        padding: bool
            Whether or not to perform zero-padding
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.medium_index = medium_index
        self.distance = distance
        self.padding = padding
        if padding:
            field = pad.pad_add(field)
        self.fft_field0 = np.fft.fft(field)

    def get_kernel(self, distance, kernel="helmholtz"):
        """Return the kernel for a 1D propagation"""
        nm = self.medium_index
        res = self.wavelength / self.pixel_size
        d = (distance - self.distance) / self.pixel_size
        twopi = 2 * np.pi

        km = twopi * nm / res
        kx = np.fft.fftfreq(len(self.fft_field0)) * 2 * np.pi

        # free space propagator is
        if kernel == "helmholtz":
            # exp(i*sqrt(km²-kx²)*d)
            # Also subtract incoming plane wave. We are only considering
            # the scattered field here.
            root_km = km ** 2 - kx ** 2
            rt0 = (root_km > 0)
            # multiply by rt0 (filter in Fourier space)
            fstemp = np.exp(1j * (np.sqrt(root_km * rt0) - km) * d) * rt0
        elif kernel == "fresnel":
            # exp(i*d*(km-kx²/(2*km))
            # fstemp = np.exp(-1j * d * (kx**2/(2*km)))
            fstemp = np.exp(-1j * d * (kx ** 2 / (2 * km)))
        else:
            raise KeyError(f"Unknown propagation kernel: '{kernel}'")
        return fstemp

    def propagate(self, distance, kernel):
        """Propagate the initial field to a certain distance

        Parameters
        ----------
        distance: float
            Absolute focusing distance [m]
        kernel: str
            Propagation kernel, one of

            - "helmholtz": the optical transfer function `exp(idkₘ(M-1))`
            - "fresnel": paraxial approximation `exp(idk²/kₘ)`

        Returns
        -------
        refocused_field: 1d ndarray
            Initial 1D field refocused at `distance`
        """
        fft_kernel = self.get_kernel(distance=distance, kernel=kernel)
        refoc = np.fft.ifft(self.fft_field0 * fft_kernel)
        if self.padding:
            refoc = pad.pad_rem(refoc)
        return refoc
