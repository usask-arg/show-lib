from __future__ import annotations

import importlib
import pkgutil

import pytest

import showlib

MODULES = sorted(
    m.name for m in pkgutil.walk_packages(showlib.__path__, prefix="showlib.")
)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
