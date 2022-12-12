import pytest

from nrefocus.roi_handling import parse_roi, ROIValueError


def test_parse_roi_single_list():
    """Check if a single list works"""
    # [axis_0_start, axis_1_start, axis_0_end, axis_1_end]
    roi = [5, 2, 6, 4]
    # will result in xy
    roi_expected = (slice(5, 6, None),
                    slice(2, 4, None))

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_single_list_nones():
    """Check if a single list works with None"""

    roi = [None, 2, 6, None]
    roi_expected = (slice(None, 6, None),
                    slice(2, None, None))

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_list_1d():
    """Check if a single list works for 1d roi"""
    # [axis_0_start, axis_0_end]
    roi = [2, 6]
    roi_expected = slice(2, 6)

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_list_of_slices():
    """Check if a single tuple works"""

    roi = [slice(2, 4), slice(5, 6)]
    roi_expected = (slice(2, 4, None),
                    slice(5, 6, None))

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_list_of_slices_nones():
    """Check if a list of slices works with None"""

    roi = [slice(None, 4), slice(None, 6)]
    roi_expected = (slice(None, 4, None),
                    slice(None, 6, None))

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_slice_only():
    """Check if a single slice works"""

    roi = slice(2, 4)
    roi_expected = slice(2, 4)

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_list_of_slice_1d_error():
    """Check if a single list of tuple works for 1d roi"""

    roi = [slice(2, 6)]
    roi_expected = (slice(2, 6, None),)

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_error_list():
    """Check if a list of lists works (it doesn't)"""

    roi = [[2, 4], [5, 6]]

    with pytest.raises(ROIValueError):
        _ = parse_roi(roi)
