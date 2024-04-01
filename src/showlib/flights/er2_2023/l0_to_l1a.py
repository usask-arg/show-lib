from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from skretrieval.util import configure_log

from showlib.flights.er2_2023.Platform.ER2Platform import ER2Platform as er2
from showlib.flights.er2_2023.Specifications.shower2_specs import SHOW_specs as specs
from showlib.flights.er2_2023.l1a.data import L1AFileWriter, L1AImage
from showlib.flights.er2_2023.l1a.l1a_processing import level1A_processing as l1a


def er2_attitude(show_utc, filename):
    er2data = xr.load_dataset(filename)
    slice = er2data.sel(time=show_utc, method="nearest")

    er2altitude = slice.altitude.data
    pitch = slice.pitch.data
    roll = slice.variables["roll"].data
    heading = slice.heading.data
    latitude = slice.latitude.data
    longitude = slice.longitude.data

    observer = [latitude, longitude, er2altitude]
    aircraft_orientation = [heading, pitch, roll]

    return observer, aircraft_orientation


def geometry_setup(shs_config, show_utc_time, observer=None, aircraft_orientation=None):
    #######################################################Get the sampling information###########################################
    if aircraft_orientation is None:
        aircraft_orientation = [45, 0, 0]
    if observer is None:
        observer = [0, 135, 21000]
    ER2geometry = er2(shs_config)
    ER2geometry.make_ER2_geometry(
        utc=show_utc_time, observer=observer, aircraft_orientation=aircraft_orientation
    )
    return ER2geometry


def set_measurement_config(
    utc=None,
    iWGnc_filename=None,
):
    # SHS configuration
    shs_config = specs()

    measurement_config_utc = utc
    er2 = er2_attitude(measurement_config_utc, iWGnc_filename)

    # Generate a measurement set
    measurement_geometry = geometry_setup(
        shs_config, measurement_config_utc, observer=er2[0], aircraft_orientation=er2[1]
    )

    return measurement_geometry, shs_config


def process_l0_to_l1a(
    SHOW_l0_file: Path, iWG_file: Path, Cal_file: Path, output_folder: Path
):
    # Open the data file
    SHOW_data = xr.open_dataset(SHOW_l0_file)
    output_file = output_folder.joinpath(
        output_folder,
        SHOW_l0_file.stem.replace("L0", "L1A").replace("Uncalibrated", "Calibrated")
        + ".nc",
    )

    SHOW_l1a_entries = []
    for time in SHOW_data.time.data:
        # select the l0 entry to process
        ds = SHOW_data.sel(time=time)

        # define the measurement configuration
        measurement_geometry, shs_config = set_measurement_config(
            utc=time, iWGnc_filename=iWG_file
        )

        # Define the l1a processor
        l1a_processor = l1a(shs_config)

        # process the entry
        l1a_data = l1a_processor.process_signal(ds)

        # flip the images to put the ground at the bottom of the image
        l1a_image = np.flipud(l1a_data["image"].data)
        C2 = np.flipud(l1a_data["C2"].data)
        noise = np.flipud(l1a_data["error"].data)

        # Calibration data
        calibration_data = xr.open_dataset(Cal_file)
        filter_shape = np.interp(
            shs_config.wav_num[0:247],
            calibration_data.wavenumbers.data,
            calibration_data.filter_shape.data,
        )
        abs_cal = 0.475 * calibration_data.abscal.data
        pixel_response = shs_config.pixel_response[0:247]

        # Merge the geometry information with the L0 file
        SHOW_l1a_entries.append(
            L1AImage(
                image=l1a_image,
                C2=C2,
                noise=noise,
                tangent_locations=measurement_geometry.tangent_locations,
                tangent_latitudes=measurement_geometry.tangent_latitudes,
                tangent_longitudes=measurement_geometry.tangent_longitudes,
                spacecraft_altitude=measurement_geometry.observer_altitude,
                spacecraft_latitude=measurement_geometry.observer_latitude,
                spacecraft_longitude=measurement_geometry.observer_longitude,
                solar_zenith_angle=measurement_geometry.sza,
                relative_solar_azimuth_angle=measurement_geometry.saa,
                los_azimuth_angle=measurement_geometry.los_azimuth,
                time=time,
                filter_shape=filter_shape,
                abs_cal=abs_cal,
                pixel_response=pixel_response,
            )
        )

    writer = L1AFileWriter(SHOW_l1a_entries)
    writer.save(output_file)


if __name__ == "__main__":
    configure_log()
    in_folder = Path(
        r"C:\Users\t383r\SHOW ER-2 Algorithm Dev\ER2_2023\data\SDSPipelineTest\UncalibratedL0"
    )
    output_folder = Path(
        r"C:\Users\t383r\SHOW ER-2 Algorithm Dev\ER2_2023\data\SDSPipelineTest\L1A"
    )

    # corresponding iWG file
    iWG_file = Path(
        r"C:\Users\t383r\SHOW ER-2 Algorithm Dev\ER2_2023\data\iwg1-50hz-27Nov2023-2214.nc"
    )

    # calibration file
    cal_file = Path(
        r"C:\Users\t383r\SHOW ER-2 Algorithm Dev\ER2_2023\show-lib\CalibrationData\SHOW_ER2_abscal_2024_02_08_v0.2.nc"
    )

    # proc_l0_to_l1a()
    for file in in_folder.glob("HAWC*"):
        if file.suffix == ".nc":
            process_l0_to_l1a(file, iWG_file, cal_file, output_folder)
