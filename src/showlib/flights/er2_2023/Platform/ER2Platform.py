from __future__ import annotations

import numpy as np
import pandas as pd
import sasktran as sk
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from sasktran import LineOfSight
from skretrieval.core import OpticalGeometry

# from skplatform import Platform
from skretrieval.platforms import Platform

# from arg_time import ut_to_datetime64
from skretrieval.time import ut_to_datetime64
from skretrieval.util import rotation_matrix


class ER2Platform:
    def __init__(self, shs_config):
        self.__verbose__ = True
        self.specs = shs_config
        self.vertical_sampling = self.specs.DetNumPixY

    # def load_saved_data(self, csv_filename = r'C:\Users\jeffl\SHOWER2Sim\show-er-simulator\showretrieval\Support_Libraries\Platform\2017_ScienceFlight.csv'):
    #
    #     path = csv_filename
    #
    #     data = pd.read_csv(path,
    #                        index_col=0,
    #                        usecols=[1, 2, 3, 4, 8, 13, 14, 16, 17],
    #                        names=['time', 'latitude', 'longitude', 'altitude', 'speed', 'heading', 'track',
    #                               'pitch_angle', 'roll_angle'],
    #                        parse_dates=True, skiprows=700).to_xarray().sortby('time')
    #     data = data.where(np.isfinite(data.latitude) & np.isfinite(data.roll_angle), drop=True)
    #     return data

    def make_ER2_geometry(
        self,
        utc=None,
        observer=None,
        aircraft_orientation=None,
    ):
        """Make a single limb observation from the ER2 platform, at some time, where the observer location is defined in the geodetic location given by latitude, longitude and height"""
        if aircraft_orientation is None:
            aircraft_orientation = [0, 0, 0]
        if observer is None:
            observer = [80, 235, 600000]
        if utc is None:
            utc = ["2020-07-15T15:00:00.0000000"]
        self.observer_latitude = observer[0]
        self.observer_longitude = observer[1]
        self.observer_altitude = observer[2]
        utc = utc
        observer = observer
        self.hfov = self.specs.hfov
        self.vfov = self.specs.vfov

        # convert utc start time to a datetime object
        ut = ut_to_datetime64(utc)

        aircraft = [
            aircraft_orientation[0],
            aircraft_orientation[1]
            + self.specs.boresight_pitchangle
            + 50 * self.specs.height_degrees_per_pixel
            - 0.07,
            aircraft_orientation[2],
        ]

        # make a platform
        platform = Platform()

        # add the measurement set
        platform.add_measurement_set(
            ut, ("llh", observer), ("yaw_pitch_roll", "standard", aircraft)
        )

        # add the lines of sight
        self.optical_axis = platform.make_optical_geometry()[0]

        # Get a vertical image of the lines of sight for each measurement
        self.lines_of_sight = self.generate_SHOW_vertical_image(
            self.optical_axis, self.vertical_sampling
        )

        # Get the corresponding tangent locations
        self.tangent_locations = self.calc_tangents(self.lines_of_sight)
        self.tangent_latitudes = self.calc_latitudes(self.lines_of_sight)
        self.tangent_longitudes = self.calc_longitudes(self.lines_of_sight)
        self.los_azimuth = self.calc_line_of_sight_azimuth(self.lines_of_sight)

        # Calculate the reference tangent
        self.reference_tangent = (
            self.measurement_geometry(self.optical_axis)[0].tangent_location().altitude
        )

        # Get the distance to the satellite
        self.optical_axis_LOS = self.measurement_geometry(self.optical_axis)[0]
        self.distance = (
            np.linalg.norm(
                self.optical_axis_LOS.tangent_location().location
                - self.optical_axis_LOS.observer
            )
            / 1000
        )

        # solar angles
        self.sza, self.saa = self.calculate_sza_saa()

    def calc_reference_tangent(
        self,
        utc=None,
        observer=None,
        aircraft_orientation=None,
    ):
        """Make a single limb observation from the ER2 platform, at some time, where the observer location is defined in the geodetic location given by latitude, longitude and height"""

        if aircraft_orientation is None:
            aircraft_orientation = [0, 0, 0]
        if observer is None:
            observer = [80, 235, 600000]
        if utc is None:
            utc = ["2020-07-15T15:00:00.0000000"]
        utc = utc
        observer = observer
        self.hfov = self.specs.hfov
        self.vfov = self.specs.vfov

        # convert utc start time to a datetime object
        ut = ut_to_datetime64(utc)

        aircraft_orientation[1] = aircraft_orientation[1] + self.specs.bore_sight_pitch
        # make a platform
        platform = Platform()

        # add the measurement set
        platform.add_measurement_set(
            ut, ("llh", observer), ("yaw_pitch_roll", "standard", aircraft_orientation)
        )

        # add the lines of sight
        self.optical_axis = platform.make_optical_geometry()[0]

        # Calculate the reference tangent
        return (
            self.measurement_geometry(self.optical_axis)[0].tangent_location().altitude
        )


    def generate_SHOW_vertical_image(self, optical_axis, vertical_sampling):
        """Takes in the optical axis Optical Geometry, vertical sampling and the vertical field of view and generates
        the instrument lines of sight in the vertical dimension"""

        geo = sk.Geometry()

        # Get the sensor FOV
        vert_fov = self.vfov * np.pi / 180

        # Set the vertical sampling across the vertical FOV based off the specified high res sampling
        vert_sampling = vertical_sampling

        # define the local up for the central tangent
        local_up = optical_axis.local_up
        local_up = local_up / np.linalg.norm(local_up)

        # define the central line of sight
        central_los = optical_axis.look_vector
        central_los = central_los / np.linalg.norm(central_los)

        vert_fov = vert_fov
        model_angles = np.linspace(-vert_fov / 2, vert_fov / 2, vertical_sampling)
        vert_angular_spacing = np.mean(np.diff(model_angles))

        # For each model angle rotate the central los to obtain the corresponding los in the vertical image
        rot_axis = np.cross(central_los, local_up)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        los_optical_geometry = []

        for model_angle in model_angles:
            rot_matrix = rotation_matrix(rot_axis, model_angle)

            # Rotate the look vector in the vertical dimension
            look_vector = rot_matrix @ central_los

            geo.lines_of_sight.append(
                sk.LineOfSight(optical_axis.mjd, optical_axis.observer, look_vector)
            )
            los_optical_geometry.append(
                OpticalGeometry(
                    look_vector=look_vector,
                    observer=optical_axis.observer,
                    mjd=optical_axis.mjd,
                    local_up=local_up,
                )
            )

        self.angles = model_angles

        return los_optical_geometry

    def calc_tangents(self, los_optical_geometry: OpticalGeometry):
        tangent_locations = []
        observer_distance = []
        for meas in los_optical_geometry:
            line_of_sight = self.measurement_geometry(meas)
            try:
                tangent_locations.append(line_of_sight[0].tangent_location().altitude)
            except:
                tangent_locations.append(0)

        return np.array(tangent_locations)

    def calc_latitudes(self, los_optical_geometry: OpticalGeometry):
        lats = []
        for meas in los_optical_geometry:
            line_of_sight = self.measurement_geometry(meas)
            try:
                lats.append(line_of_sight[0].tangent_location().latitude)
            except:
                lats.append(0)
        return np.array(lats)

    def calc_longitudes(self, los_optical_geometry: OpticalGeometry):
        lons = []
        for meas in los_optical_geometry:
            line_of_sight = self.measurement_geometry(meas)
            try:
                lons.append(line_of_sight[0].tangent_location().latitude)
            except:
                lons.append(0)
        return np.array(lons)

    def calc_line_of_sight_azimuth(self, los_optical_geometry: OpticalGeometry):
        los_azimuth = []
        for meas in los_optical_geometry:
            line_of_sight = self.measurement_geometry(meas)
            try:
                north = -line_of_sight[0].tangent_location().local_south
                los_azimuth.append(
                    np.arccos(np.dot(line_of_sight[0].look_vector, north))
                )
            except:
                los_azimuth.append(0)
        return np.array(los_azimuth)

    def calc_distance_to_tangent(self, los_optical_geometry: OpticalGeometry):
        observer_distance = []
        for meas in los_optical_geometry:
            line_of_sight = self.measurement_geometry(meas)
            try:
                observer_distance.append(
                    np.linalg.norm(
                        line_of_sight[0].tangent_location().location
                        - line_of_sight[0].observer
                    )
                )
            except:
                observer_distance.append(0)

        return np.array(observer_distance)

    def measurement_geometry(self, optical_geometry: OpticalGeometry):
        return [
            LineOfSight(
                optical_geometry.mjd,
                optical_geometry.observer,
                optical_geometry.look_vector,
            )
        ]

    def calculate_sza_saa(self):
        mjd_time = self.optical_axis.mjd
        sun_dir = get_sun(Time(mjd_time, format="mjd"))

        location = EarthLocation(
            lat=self.tangent_latitudes, lon=self.tangent_longitudes
        )
        sza = (
            90
            - sun_dir.transform_to(
                AltAz(obstime=Time(mjd_time, format="mjd"), location=location)
            ).alt.value
        )
        saa = (
            sun_dir.transform_to(
                AltAz(obstime=Time(mjd_time, format="mjd"), location=location)
            ).az.value
            - 180
        )

        return sza, saa
