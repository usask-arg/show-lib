from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from skretrieval.util import configure_log

from showlib.flights.er2_2023.Specifications.shower2_specs import SHOW_specs as specs
from showlib.l1b.data import L1bFileWriter, L1bImage
from showlib.l1b.l1b_processing import level1B_processing as l1b


def process_l1a_to_l1b(SHOW_l1a_file: Path, output_folder: Path):
    SHOW_l1a_data = xr.open_dataset(SHOW_l1a_file)
    output_file = output_folder.joinpath(
        output_folder, SHOW_l1a_file.stem.replace("L1A", "L1B") + ".nc"
    )

    SHOW_l1b_entries = []
    for time in SHOW_l1a_data.time.data:
        # select the entry to process in the L1A file
        l1a_ds = SHOW_l1a_data.sel(time=time)

        # Set the Level 1B processing options
        processing_steps = {
            "DC_Filter": False,
            "apply_phase_correction": True,
            "apply_apodization": True,
            "apply_finite_pixel_correction": True,
            "apply_filter_correction": True,
            "apply_abscal": True,
        }

        # SHS configuration
        shs_config = specs()

        # set up the L1B processor
        L1B_process = l1b(processing_steps=processing_steps, specs=shs_config)

        # Level 1A to Level 1B
        l1b_data = L1B_process.process_signal(l1a_ds)

        # Merge the geometry information with the L0 file
        SHOW_l1b_entries.append(
            L1bImage(
                radiance=l1b_data["radiance"].data,
                radiance_noise=l1b_data["radiance_noise"].data,
                wavenumber_spacing=shs_config.wavenumber_spacing
                * np.ones_like(l1a_ds["tangent_altitude"].data),
                left_wavenumber=shs_config.wav_num[0]
                * np.ones_like(l1a_ds["tangent_altitude"].data),
                tangent_altitude=l1a_ds["tangent_altitude"].data,
                tangent_latitude=l1a_ds["tangent_latitude"].data,
                tangent_longitude=l1a_ds["tangent_longitude"].data,
                observer_altitude=l1a_ds["spacecraft_altitude"].data,
                observer_latitude=l1a_ds["spacecraft_latitude"].data,
                observer_longitude=l1a_ds["spacecraft_longitude"].data,
                sza=l1a_ds["solar_zenith_angle"].data,
                saa=l1a_ds["relative_solar_azimuth_angle"].data,
                los_azimuth_angle=l1a_ds["los_azimuth_angle"].data,
                time=l1a_ds["time"].data,
            )
        )

    writer = L1bFileWriter(SHOW_l1b_entries)
    writer.save(output_file)


if __name__ == "__main__":
    configure_log()
    in_folder = Path(
        r"C:\Users\t383r\SHOW ER-2 Algorithm Dev\ER2_2023\data\SDSPipelineTest\L1A"
    )
    output_folder = Path(
        r"C:\Users\t383r\SHOW ER-2 Algorithm Dev\ER2_2023\data\SDSPipelineTest\L1B"
    )

    # proc_l1a_to_l1b()
    for file in in_folder.glob("HAWC*"):
        if file.suffix == ".nc":
            process_l1a_to_l1b(file, output_folder)
