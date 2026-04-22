"""
Module that controls and exposes the active ndarray backend (NumPy or CuPy).

.. versionadded:: 0.6.0
"""

import importlib

_default_backend = "numpy"
_xp = importlib.import_module(_default_backend)


class NDArrayBackend:
    """Proxy object exposing the current ndarray backend."""

    def __init__(self):
        self._xp = _xp

    def get(self):
        """Return the currently active backend module."""
        return self._xp

    def set(self, backend_name: str = "numpy"):
        """Switch the backend between 'numpy' and 'cupy'."""
        global _xp
        try:
            # run the backend swap regardless
            self._xp = importlib.import_module(backend_name)
            _xp = self._xp  # keep global in sync
        except ModuleNotFoundError as err:
            raise ImportError(f"The backend '{backend_name}' is not "
                              f"installed. Either install it or use the "
                              f"default backend: 'numpy'.") from err

    # --- Convenience passthroughs ---
    def __getattr__(self, name):
        """Delegate unknown attributes to the backend module."""
        return getattr(self._xp, name)

    def backend_name(self):
        return self._xp.__name__

    def is_numpy(self):
        return self._xp.__name__.startswith("numpy")

    def is_cupy(self):
        return self._xp.__name__.startswith("cupy")

    def assert_numpy(self):
        assert self.is_numpy(), (
            "ndarray_backend is not 'numpy'. "
            "To use RefocusNumpy, run `set('numpy')`."
        )

    def assert_cupy(self):
        assert self.is_cupy(), (
            "ndarray_backend is not 'cupy'. "
            "To use RefocusCupy, run `set('cupy')`."
        )


class NDArrayBackendWarning(UserWarning):
    def __init__(self, message):
        self.message = message


# Export a single global proxy instance
xp = NDArrayBackend()
# This is what is imported by the user
get_ndarray_backend = xp.get
set_ndarray_backend = xp.set
