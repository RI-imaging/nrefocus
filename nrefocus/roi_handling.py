import numbers


class ROIValueError(ValueError):
    pass


def parse_roi(roi):
    """Handle the roi information.

    Parameters
    ----------
    roi: list or tuple or slice or ndarray
        Region of interest for which the metric will be minimized.
        The axes below use the numpy indexing order.
        Options are:
        list or tuple or numpy indexing array (old behaviour):
            [axis_0_start, axis_1_start, axis_0_end, axis_1_end]
            None can be used if no slicing is desired eg.:
            [None, None, axis_0_end, axis_1_end]
        list or tuple of slices (will be passed directly as is):
            (slice(axis_0_start, axis_0_end),
             slice(axis_1_start, axis_1_end))
        None
            The entire field will be used.

    Returns
    -------
    roi : slices

    """
    err_descr = f"Unexpected value for `roi`: '{roi}'"

    if roi is None:
        # Use all the data
        pass
    elif isinstance(roi, slice):
        # will be directly used
        pass
    elif all(isinstance(s, slice) for s in roi):
        # will be directly used
        roi = tuple(roi)
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
