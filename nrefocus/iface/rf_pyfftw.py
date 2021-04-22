import multiprocessing as mp

import numpy as np
import pyfftw

from .. import pad

from .base import Refocus


class RefocusPyFFTW(Refocus):
    """Refocusing with FFTW

    .. versionadded:: 0.4.0
    """
    def _init_fft(self, field, padding):
        """Perform initial Fourier transform of the input field

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
        The number of threads in PyFFTW is currently set to the
        number of CPUs via `multiprocessing.cpu_count()`.
        """
        if padding:
            field = pad.pad_add(field)
        # compute the input Fourier transform
        origin = pyfftw.empty_aligned(field.shape, dtype='complex128')
        fft_origin = pyfftw.empty_aligned(field.shape, dtype='complex128')
        fft_obj = pyfftw.FFTW(origin, fft_origin, axes=(0, 1))
        origin[:] = field
        fft_obj()

        # now setup the backward transform
        inv_input = pyfftw.empty_aligned(field.shape, dtype='complex128')
        inv_output = pyfftw.empty_aligned(field.shape, dtype='complex128')
        self._ifft_obj = pyfftw.FFTW(inv_input, inv_output, axes=(0, 1),
                                     direction="FFTW_BACKWARD",
                                     flags=["FFTW_DESTROY_INPUT"],
                                     threads=mp.cpu_count())
        return fft_origin

    def propagate(self, distance):
        fft_kernel = self.get_kernel(distance=distance)
        np.multiply(self.fft_origin, fft_kernel,
                    out=self._ifft_obj.input_array)
        refoc = self._ifft_obj()
        if self.padding:
            refoc = pad.pad_rem(refoc)
        return refoc
