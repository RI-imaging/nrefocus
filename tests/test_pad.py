"""Test padding"""
import numpy as np

from nrefocus import pad


def test_pad_2d():
    sizes = [(50, 100),
             (51, 100),
             (50, 101),
             (51, 101),
             ]
    for sx, sy in sizes:
        a = np.arange(sx*sy).reshape((sx, sy))
        b = pad.pad_add(a, size=None, stlen=10)
        assert np.sum(a - b[:sx, :sy]) == 0
        assert np.sum(a - pad.pad_rem(b)) == 0


def test_pad_1d():
    sizes = [50, 51, 100, 101]
    for sx in sizes:
        a = np.arange(sx)
        b = pad.pad_add(a, size=None, stlen=10)
        assert np.sum(a - b[:sx]) == 0
        assert np.sum(a - pad.pad_rem(b)) == 0


def test_pad_1d_cmplx():
    sizes = [50, 51, 100, 101]
    for sx in sizes:
        a = 1.1*np.exp(1j*np.abs(np.linspace(-0.5, .7, sx))**(1/3))
        b = pad.pad_add(a, size=None, stlen=10)
        assert np.sum(a - b[:sx]) == 0
        assert np.sum(a - pad.pad_rem(b)) == 0


def test_pad_2d_cmplx():
    sizes = [(50, 100),
             (51, 100),
             (50, 101),
             (51, 101),
             ]
    for sx, sy in sizes:
        x = np.linspace(0, 1, sx).reshape(-1, 1)
        y = np.linspace(0, .5, sy).reshape(1, -1)
        a = 1.1*np.exp(1j*x*y)
        b = pad.pad_add(a, size=None, stlen=10)
        assert np.sum(a - b[:sx, :sy]) == 0
        assert np.sum(a - pad.pad_rem(b)) == 0


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
