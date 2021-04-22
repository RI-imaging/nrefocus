import numpy as np

from .. import pad

from .base import Refocus


class RefocusNumpy1D(Refocus):
    def __init__(self, field, wavelength, pixel_size, medium_index=1.3333,
                 distance=0, kernel="helmholtz", padding=True):
        r"""Refocus a 1D field with numpy

        .. versionadded:: 0.3.0

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
        kernel: str
            Propagation kernel, one of

            - "helmholtz": the optical transfer function
              :math:`\exp\left(id\left(\sqrt{k_\mathrm{m}^2 - k_\mathrm{x}^2}
              - k_\mathrm{m}\right)\right)`
            - "fresnel": paraxial approximation
              :math:`\exp(-idk_\mathrm{x}^2/2k_\mathrm{m})`
        padding: bool
            Whether or not to perform zero-padding
        """
        super(RefocusNumpy1D, self).__init__(
            field=field,
            wavelength=wavelength,
            pixel_size=pixel_size,
            medium_index=medium_index,
            distance=distance,
            kernel=kernel,
            padding=padding,
        )

    def _init_fft(self, field, padding):
        """Perform initial Fourier transform of the input field

        Parameters
        ----------
        field: 1d complex-valued ndarray
            Input field to be refocused
        padding: bool
            Whether or not to perform zero-padding

        Returns
        -------
        fft_field0: 1d complex-valued ndarray
            Fourier transform the the initial field
        """
        if padding:
            field = pad.pad_add(field)
        return np.fft.fft(field)

    def get_kernel(self, distance):
        """Return the kernel for a 1D propagation"""
        nm = self.medium_index
        res = self.wavelength / self.pixel_size
        d = (distance - self.distance) / self.pixel_size
        twopi = 2 * np.pi

        km = twopi * nm / res
        kx = np.fft.fftfreq(len(self.fft_origin)) * 2 * np.pi

        # free space propagator is
        if self.kernel == "helmholtz":
            # unnormalized: exp(i*sqrt(km²-kx²)*d)
            # Also subtract incoming plane wave. We are only considering
            # the scattered field here.
            root_km = km ** 2 - kx ** 2
            rt0 = (root_km > 0)
            # multiply by rt0 (filter in Fourier space)
            fstemp = np.exp(1j * (np.sqrt(root_km * rt0) - km) * d) * rt0
        elif self.kernel == "fresnel":
            # unnormalized: exp(i*d*(km-kx²/(2*km))
            fstemp = np.exp(-1j * d * kx ** 2 / (2 * km))
        else:
            raise KeyError(f"Unknown propagation kernel: '{self.kernel}'")
        return fstemp

    def propagate(self, distance):
        """Propagate the initial field to a certain distance

        Parameters
        ----------
        distance: float
            Absolute focusing distance [m]

        Returns
        -------
        refocused_field: 1d ndarray
            Initial 1D field refocused at `distance`
        """
        fft_kernel = self.get_kernel(distance=distance)
        refoc = np.fft.ifft(self.fft_origin * fft_kernel)
        if self.padding:
            refoc = pad.pad_rem(refoc)
        return refoc
