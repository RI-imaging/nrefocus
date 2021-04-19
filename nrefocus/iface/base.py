from abc import ABC, abstractmethod

import numpy as np


class Refocus(ABC):
    def __init__(self, field, wavelength, pixel_size, medium_index=1.3333,
                 distance=0, padding=True):
        """Base class for refocusing

        Parameters
        ----------
        field: 2d complex-valued ndarray
            Input field to be refocused
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
        super(Refocus, self).__init__()
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.medium_index = medium_index
        self.distance = distance
        self.padding = padding
        self.fft_field0 = self._init_fft(field, padding)

    @abstractmethod
    def _init_fft(self, field, padding):
        """Initialize Fourier transform for propagation

        This is where you would compute the initial Fourier transform.
        E.g. for FFTW, you would do planning here.

        Parameters
        ----------
        field: 2d complex-valued ndarray
            Input field to be refocused
        padding: bool
            Whether or not to perform zero-padding

        Returns
        -------
        fft_field0: 2d complex-valued ndarray
            Fourier transform the the initial field

        Notes
        -----
        Any subclass should perform padding with
        :func:`nrefocus.pad.padd_add` during initialization.
        """

    def get_kernel(self, distance, kernel="helmholtz"):
        nm = self.medium_index
        res = self.wavelength / self.pixel_size
        d = (distance - self.distance) / self.pixel_size
        twopi = 2 * np.pi

        km = twopi * nm / res
        kx = (np.fft.fftfreq(self.fft_field0.shape[0]) * twopi).reshape(-1, 1)
        ky = (np.fft.fftfreq(self.fft_field0.shape[1]) * twopi).reshape(1, -1)
        if kernel == "helmholtz":
            # exp(i*sqrt(km²-kx²-ky²)*d)
            root_km = km ** 2 - kx ** 2 - ky ** 2
            rt0 = (root_km > 0)
            # multiply by rt0 (filter in Fourier space)
            fstemp = np.exp(1j * (np.sqrt(root_km * rt0) - km) * d) * rt0
        elif kernel == "fresnel":
            # exp(i*d*(km-(kx²+ky²)/(2*km))
            fstemp = np.exp(-1j * d * (kx ** 2 + ky ** 2) / (2 * km))
        else:
            raise KeyError(f"Unknown propagation kernel: '{kernel}'")
        return fstemp

    @abstractmethod
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
        refocused_field: 2d ndarray
            Initial field refocused at `distance`

        Notes
        -----
        Any subclass should perform padding with
        :func:`nrefocus.pad.pad_rem` during initialization.
        """
