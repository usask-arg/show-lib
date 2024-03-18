from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import xarray as xr
from skretrieval.time.mjd import mjd_to_datetime

from showlib.l0.data import L0FileWriter, L0Image


def convert_raw_l1a_to_l0_granules(file: Path, out_folder: Path, granularity_minutes=5):
    ds = xr.open_dataset(file)

    times = ds["mjd"].to_numpy()[:, 0]

    start_time = mjd_to_datetime(times[0])
    granularity_delta = timedelta(minutes=granularity_minutes)

    l0s = []
    for i in range(0, len(times), 5):
        time = mjd_to_datetime(times[i])
        if time - start_time > granularity_delta:
            writer = L0FileWriter(l0s)
            writer.save(
                out_folder.joinpath(
                    f"HAWC_H2OL_Uncalibrated_L0_{start_time.strftime('%Y-%m-%dT%H-%M-%SZ')}.v0_0_1.STD.nc"
                )
            )

            l0s = []
            start_time = time
            break

        selected = ds.isel(sample=i)

        l0s.append(L0Image(selected))


if __name__ == "__main__":
    convert_raw_l1a_to_l0_granules(
        Path(
            "/Users/dannyz/Downloads/20231128_science_flight1_l1b_cm_1_north_slice3.nc"
        ),
        granularity_minutes=1,
    )
