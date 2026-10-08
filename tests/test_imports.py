from __future__ import annotations

import importlib
import pkgutil

import pytest

import showlib

# Modules that are known to be broken against the current dependency set. These are strict xfails,
# so fixing one of them will fail this test until it is removed from the list.
KNOWN_BROKEN = {
    "showlib.flights.er2_2023.l1bdata": "imports L1bImageBase, which no longer exists",
}

MODULES = sorted(
    m.name for m in pkgutil.walk_packages(showlib.__path__, prefix="showlib.")
)


PARAMS = [
    (
        pytest.param(
            name, marks=pytest.mark.xfail(reason=KNOWN_BROKEN[name], strict=True)
        )
        if name in KNOWN_BROKEN
        else name
    )
    for name in MODULES
]


@pytest.mark.parametrize("module", PARAMS)
def test_module_imports(module):
    importlib.import_module(module)


def test_known_broken_list_is_current():
    assert set(KNOWN_BROKEN) <= set(MODULES)
