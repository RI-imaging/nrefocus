"""Convenience functions for padding

.. versionadded:: 0.1.4
"""
from __future__ import division, print_function

import numpy as np


def _get_pad_left_right(small, large):
    """Compute left and right padding values.

    Here we use the convention that if the padding
    size is odd, we pad the odd part to the right
    and the even part to the left.

    Parameters
    ----------
    small: int
        Old size of original 1D array
    large: int
        New size off padded 1D array

    Returns
    -------
    (padleft, padright) : tuple
        The proposed padding sizes.
    """
    assert small < large, "Can only pad when new size larger than old size"

    padsize = large - small
    if padsize % 2 != 0:
        leftpad = (padsize - 1)/2
    else:
        leftpad = padsize/2
    rightpad = padsize-leftpad

    return int(leftpad), int(rightpad)


def pad_add(av, size=None, stlen=10):
    """ Perform linear padding for complex array

    The input array `av` is padded with a linear ramp starting at the
    edges and going outwards to an average value computed from a band
    of thickness `stlen` at the outer boundary of the array.

    Pads will only be appended, not prepended to the array.

    If the input array is complex, pads will be complex numbers
    The average is computed for phase and amplitude separately.

    Parameters
    ----------
    av: complex 1D or 2D ndarray
        The array that will be padded.
    size: int or tuple of length 1 (1D) or tuple of length 2 (2D), optional
        The final size of the padded array. Defaults to double the size
        of the input array.
    stlen: int, optional
        The thickness of the frame within `av` that will be used to
        compute an average value for padding.


    Returns
    -------
    pv: complex 1D or 2D ndarray
        Padded array `av` with pads appended to right and bottom.
    """
    if size is None:
        size = list()
        for s in av.shape:
            size.append(int(2*s))
    elif not hasattr(size, "__len__"):
        size = [size]

    assert len(av.shape) in [1, 2], "Only 1D and 2D arrays!"
    assert len(av.shape) == len(
        size), "`size` must have same length as `av.shape`!"

    if len(av.shape) == 2:
        return _pad_add_2d(av, size, stlen)
    else:
        return _pad_add_1d(av, size, stlen)


def _pad_add_1d(av, size, stlen):
    """1D component of `pad_add`"""
    assert len(size) == 1

    padx = _get_pad_left_right(av.shape[0], size[0])

    mask = np.zeros(av.shape, dtype=bool)
    mask[stlen:-stlen] = True
    border = av[~mask]
    if av.dtype.name.count("complex"):
        padval = np.average(np.abs(border)) * \
            np.exp(1j*np.average(np.angle(border)))
    else:
        padval = np.average(border)
    if np.__version__[:3] in ["1.7", "1.8", "1.9"]:
        end_values = ((padval, padval),)
    else:
        end_values = (padval,)
    bv = np.pad(av,
                padx,
                mode="linear_ramp",
                end_values=end_values)
    # roll the array so that the padding values are on the right
    bv = np.roll(bv, -padx[0], 0)
    return bv


def _pad_add_2d(av, size, stlen):
    """2D component of `pad_add`"""
    assert len(size) == 2

    padx = _get_pad_left_right(av.shape[0], size[0])
    pady = _get_pad_left_right(av.shape[1], size[1])

    mask = np.zeros(av.shape, dtype=bool)
    mask[stlen:-stlen, stlen:-stlen] = True
    border = av[~mask]
    if av.dtype.name.count("complex"):
        padval = np.average(np.abs(border)) * \
            np.exp(1j*np.average(np.angle(border)))
    else:
        padval = np.average(border)
    if np.__version__[:3] in ["1.7", "1.8", "1.9"]:
        end_values = ((padval, padval), (padval, padval))
    else:
        end_values = (padval,)
    bv = np.pad(av,
                (padx, pady),
                mode="linear_ramp",
                end_values=end_values)
    # roll the array so that the padding values are on the right
    bv = np.roll(bv, -padx[0], 0)
    bv = np.roll(bv, -pady[0], 1)
    return bv


def pad_rem(pv, size=None):
    """Removes linear padding from array

    This is a convenience function that does the opposite
    of `pad_add`.

    Parameters
    ----------
    pv: 1D or 2D ndarray
        The array from which the padding will be removed.
    size: tuple of length 1 (1D) or 2 (2D), optional
        The final size of the un-padded array. Defaults to half the size
        of the input array.


    Returns
    -------
    pv: 1D or 2D ndarray
        Padded array `av` with pads appended to right and bottom.
    """
    if size is None:
        size = list()
        for s in pv.shape:
            assert s % 2 == 0, "Uneven size; specify correct size of output!"
            size.append(int(s/2))
    elif not hasattr(size, "__len__"):
        size = [size]

    assert len(pv.shape) in [1, 2], "Only 1D and 2D arrays!"
    assert len(pv.shape) == len(
        size), "`size` must have same length as `av.shape`!"

    if len(pv.shape) == 2:
        return pv[:size[0], :size[1]]
    else:
        return pv[:size[0]]
