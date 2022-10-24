class ROIValueError(ValueError):
    pass


def parse_roi(roi):
    """Handle the roi information. Only accepts tuple of slices.

    Parameters
    ----------
    roi : tuple of slices
        roi should be in the numpy indexing order:
            (slice(axis_0_start, axis_0_end),
             slice(axis_1_start, axis_1_end))
        Use None to indicate no slicing:
            (slice(None, None),
             slice(axis_1_start, axis_1_end))

    Returns
    -------
    roi : slice or None

    Notes
    -----
    You can test the roi directly on your data by verifying that data[roi]
    works.

    """
    err_descr = (f"The roi provided was not correct, "
                 f"expected a tuple of slices, "
                 f"Got {roi=} instead of {type(roi)=}")

    if roi is not None:
        if isinstance(roi, tuple) and all(isinstance(s, slice) for s in roi):
            return roi
        else:
            raise ROIValueError(err_descr)
    return roi
