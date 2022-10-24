import numbers


class ROIValueError(ValueError):
    pass


def parse_roi(roi):
    """Handle the roi information.

    Parameters
    ----------
    roi: list or tuple or ndarray or list of slices
        Region of interest for which the metric will be minimized. The below
        axes below are numpy axes. Options are:
        list or tuple or numpy indexing array (old behaviour):
            [axis_1_start, axis_0_start, axis_1_end, axis_0_end]
            None can be used if no slicing is desired eg:
            [None, None, axis_1_end, axis_0_end]
        list of slices (will be given as is for slicing):
            (slice(axis_0_start, axis_0_end),
             slice(axis_1_start, axis_1_end))
        None
            the entire field will be used.

    Notes
    -----
    The old `roi` parameter list order was given as:
    [x1, y1, x2, y2] which is consistent with the new order:
    [axis_1_start, axis_0_start, axis_1_end, axis_0_end] and is therefore
    not a breaking change.
    For the `roi` param, numpy boolean (mask) array are not yet supported.
    For 1d slices use a list, not a tuple. This is
    because python takes a tuple of a single slice as a slice.

    Returns
    -------
    roi : slices

    """
    err_descr = f"Unexpected value for `roi`: '{roi}'"

    if roi is None:
        # Use all the data
        pass
    elif all(isinstance(s, slice) for s in roi):
        # will be directly used
        pass
    elif (isinstance(roi, (list, tuple))
            and all(isinstance(r, (numbers.Number, type(None))) for r in roi)):
        # We have a list of numbers or None or mix
        if len(roi) == 2:
            # 1d slicing [axis_0_start, axis_0_end]
            roi = slice(roi[0], roi[1])
        elif len(roi) == 4:
            # 2d slicing
            roi = (slice(roi[0], roi[2]), slice(roi[1], roi[3]))
        else:
            raise ROIValueError(err_descr)
    else:
        raise ROIValueError(err_descr)

    return roi
