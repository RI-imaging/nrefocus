import numpy as np

from .. import pad

from .base import Refocus


class RefocusNumpy(Refocus):
    """Refocusing with numpy-based Fourier transform

    .. versionadded:: 0.3.0
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
        """
        if padding:
            field = pad.pad_add(field)
        return np.fft.fft2(field)

    def propagate(self, distance):
        fft_kernel = self.get_kernel(distance=distance)
        refoc = np.fft.ifft2(self.fft_origin * fft_kernel)
        if self.padding:
            refoc = pad.pad_rem(refoc)
        return refoc
