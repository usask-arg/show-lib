from __future__ import annotations

from pathlib import Path

import xarray as xr

from showlib.flights.er2_2023.convert_l1b import convert_full_l1b_to_sds

(Path(snakemake.input.raw_l1b).parent.parent / "l1b").mkdir(exist_ok=True)

convert_full_l1b_to_sds(
    Path(snakemake.input.raw_l1b),
    Path(snakemake.input.raw_l1b).parent.parent / "l1b",
    xr.open_dataset(snakemake.input.filter),
    granularity_minutes=1,
)
