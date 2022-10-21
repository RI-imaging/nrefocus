import numpy as np
import pytest

from nrefocus.roi_handling import parse_roi, ROIValueError


def test_parse_roi_single_list():
    """Check if a single list works"""

    roi = [2, 4, 5, 6]
    roi_expected = (slice(2, 4, None),
                    slice(5, 6, None))

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_single_tuple():
    """Check if a single tuple works"""

    roi = [2, 4, 5, 6]
    roi_expected = (slice(2, 4, None),
                    slice(5, 6, None))

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_list_of_lists():
    """Check if a single tuple works"""

    roi = [[2, 4], [5, 6]]
    roi_expected = (slice(2, 4, None),
                    slice(5, 6, None))

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_tuple_of_tuples():
    """Check if a single tuple works"""

    roi = ((2, 4), (5, 6))
    roi_expected = (slice(2, 4, None),
                    slice(5, 6, None))

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_list_of_tuples():
    """Check if a single tuple works"""

    roi = [(2, 4), (5, 6)]
    roi_expected = (slice(2, 4, None),
                    slice(5, 6, None))

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_tuple_of_slices():
    """Check if a single tuple works"""

    roi = (slice(2, 4), slice(5, 6))
    roi_expected = (slice(2, 4, None),
                    slice(5, 6, None))

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_list_of_slices():
    """Check if a single tuple works"""

    roi = [slice(2, 4), slice(5, 6)]
    roi_expected = [slice(2, 4, None),
                    slice(5, 6, None)]

    roi_actual = parse_roi(roi)

    assert roi_actual == roi_expected


def test_parse_roi_error_as_nonlist():
    """Check if a single tuple works"""

    roi = slice(2, 4)

    with pytest.raises(ROIValueError):
        _ = parse_roi(roi)


def test_parse_roi_error_as_list():
    """Check if a single tuple works"""

    roi = [np.array([2, 4]), np.array([5, 6])]

    with pytest.raises(ROIValueError):
        _ = parse_roi(roi)
