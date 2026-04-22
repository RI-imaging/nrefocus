import importlib

import pytest


def skip_if_missing(name: str) -> pytest.mark.skip:
    """Ignore tests which do not have the package installed.

    Notes
    -----
    This might be usable as a pytest plugin in the future, if someone builds
    it. See here: https://github.com/pytest-dev/pytest/discussions/13140
    Pytest's indirect parametrization could also be used, but adds a layer of
    complexity for new developers.

    """
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        return pytest.mark.skip(f"{name} not installed")
    return lambda func: func
