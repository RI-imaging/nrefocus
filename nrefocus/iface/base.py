from abc import ABC, abstractmethod

import numexpr as ne
import numpy as np

from .. import metrics
from .. import minimizers


class Refocus(ABC):
    def __init__(self, field, wavelength, pixel_size, medium_index=1.3333,
                 distance=0, kernel="helmholtz", padding=True):
        r"""
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
        kernel: str
            Propagation kernel, one of

            - "helmholtz": the optical transfer function
              :math:`\exp\left(id\left(\sqrt{k_\mathrm{m}^2 - k_\mathrm{x}^2
              - k_\mathrm{y}^2} - k_\mathrm{m}\right)\right)`
            - "fresnel": paraxial approximation
              :math:`\exp(-id(k_\mathrm{x}^2+k_\mathrm{y}^2)/2k_\mathrm{m})`
        padding: bool
            Whether or not to perform zero-padding
        """
        super(Refocus, self).__init__()
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.medium_index = medium_index
        self.distance = distance
        self.kernel = kernel
        self.padding = padding
        self.origin = field
        self.fft_origin = self._init_fft(field, padding)

    @property
    def shape(self):
        """Shape of the padded input field or Fourier transform"""
        return self.fft_origin.shape

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

    def get_kernel(self, distance):
        """Return the current kernel

        Ther kernel type `self.kernel` is used
        (see :func:`Refocus.__init__`)
        """
        nm = self.medium_index
        res = self.wavelength / self.pixel_size
        d = (distance - self.distance) / self.pixel_size
        twopi = 2 * np.pi

        km = twopi * nm / res
        kx = (np.fft.fftfreq(self.fft_origin.shape[0]) * twopi).reshape(-1, 1)
        ky = (np.fft.fftfreq(self.fft_origin.shape[1]) * twopi).reshape(1, -1)
        if self.kernel == "helmholtz":
            # unnormalized: exp(i*d*sqrt(km²-kx²-ky²))
            root_km = ne.evaluate("km ** 2 - kx**2 - ky**2",
                                  local_dict={"kx": kx,
                                              "ky": ky,
                                              "km": km})
            rt0 = ne.evaluate("root_km > 0")
            # multiply by rt0 (filter in Fourier space)
            fstemp = ne.evaluate(
                "exp(1j * d * (sqrt(root_km * rt0) - km)) * rt0",
                local_dict={"root_km": root_km,
                            "rt0": rt0,
                            "km": km,
                            "d": d}
            )
        elif self.kernel == "fresnel":
            # unnormalized: exp(i*d*(km-(kx²+ky²)/(2*km))
            fstemp = ne.evaluate("exp(-1j * d * (kx**2 + ky**2) / (2 * km))",
                                 local_dict={"kx": kx,
                                             "ky": ky,
                                             "km": km,
                                             "d": d})
        else:
            raise KeyError(f"Unknown propagation kernel: '{self.kernel}'")
        return fstemp

    def autofocus(self, metric="average gradient", minimizer="legacy",
                  interval=(None, None), roi=None, minimizer_kwargs=None):
        """Autofocus the initial field

        Parameters
        ----------
        metric: str
            - "average gradient" : average gradient metric of amplitude
            - "rms contrast" : RMS contrast of phase data
            - "spectrum" : sum of filtered Fourier coefficients
        minimizer: str
            - "legacy": custom nrefocus minimizer
        interval: tuple of floats
            Approximate interval to search for optimal focus in px.
        roi: rectangular region of interest (x1, y1, x2, y2)
            Region of interest of `field` for which the metric will be
            minimized. If not given, the entire `field` will be used.

        Returns
        -------
        af_field: 2d ndarray
            Autofocused field
        af_distance: float
            Autofocusing distance
        """
        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        metric_func = metrics.METRICS[metric]
        minimize_func = minimizers.mz_legacy
        af_field, af_distance = minimize_func(rfi=self,
                                              metric_func=metric_func,
                                              interval=interval,
                                              roi=roi,
                                              **minimizer_kwargs)
        return af_field, af_distance

    @abstractmethod
    def propagate(self, distance):
        """Propagate the initial field to a certain distance

        Parameters
        ----------
        distance: float
            Absolute focusing distance [m]

        Returns
        -------
        refocused_field: 2d ndarray
            Initial field refocused at `distance`

        Notes
        -----
        Any subclass should perform padding with
        :func:`nrefocus.pad.pad_rem` during initialization.
        """
