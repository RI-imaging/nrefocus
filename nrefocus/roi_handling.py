import numbers


class ROIValueError(ValueError):
    pass


def parse_roi(roi):
    """Handle the roi information.

    Parameters
    ----------
    roi : list or tuple
        roi should be in the numpy indexing order. Options:
        list or tuple:
            [axis_0_start, axis_0_end, axis_1_start, axis_1_end]
        list of lists or tuple of tuples:
            [[axis_0_start, axis_0_end],
             [axis_1_start, axis_1_end]]
        tuple of slices:
            (slice(axis_0_start, axis_0_end),
             slice(axis_1_start, axis_1_end))
            numpy boolean array (not yet supported)

    Returns
    -------
    roi : slice or None
        If roi is None, then None is returned. If roi is one of the above
        allowed types, a slice is returned.

    """

    err_descr = (f"The roi provided was not correct, "
                 f"expected either a list or tuple of numbers, "
                 f"or a list of lists, "
                 f"or a tuple of slices, "
                 # "or a numpy boolean array. "
                 f"Got {type(roi)=} instead")

    if roi is not None:
        if isinstance(roi, (list, tuple)):
            if all(isinstance(s, (list, tuple)) for s in roi):
                # assume we have a list of lists
                roi = (slice(roi[0][0], roi[0][1]),
                       slice(roi[1][0], roi[1][1]))
            elif all(isinstance(s, numbers.Number) for s in roi):
                # assume we have a list or tuple of numbers
                roi = (slice(roi[0], roi[1]), slice(roi[2], roi[3]))
            elif all(isinstance(s, slice) for s in roi):
                # can be directly used
                pass
            # allow for boolean array here
            else:
                raise ROIValueError(err_descr)
        else:
            raise ROIValueError(err_descr)
    return roi
