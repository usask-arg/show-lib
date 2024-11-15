from __future__ import annotations

import sasktran2 as sk


def h2o_optical_property(
    start_wavenumber=7295,
    end_wavenumber=7340,
    wavenumber_resolution=0.01,
    reduction_factor=1,
    backend="hapi",
):
    return sk.database.HITRANDatabase(
        molecule="H2O",
        start_wavenumber=start_wavenumber,
        end_wavenumber=end_wavenumber,
        wavenumber_resolution=wavenumber_resolution,
        reduction_factor=reduction_factor,
        backend=backend,
    )
