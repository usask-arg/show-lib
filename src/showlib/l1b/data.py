from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sasktran2 as sk
import xarray as xr
from sasktran2.geodetic import WGS84
from sasktran2.viewinggeo.ecef import ecef_to_sasktran2_ray
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.retrieval.observation import Observation
from skretrieval.util import rotation_matrix

from showlib.schemas import apply_metadata, check
from showlib.util import wrap_degrees


class L1bImage(Observation):
    """One L1B image, used as the observation in the L2 retrieval; see l1b.yaml."""

    @classmethod
    def from_np_arrays(
        cls,
        *,
        radiance: np.ndarray,
        radiance_noise: np.ndarray,
        wavenumber: np.ndarray,
        tangent_altitude: np.ndarray,
        tangent_latitude: np.ndarray,
        tangent_longitude: np.ndarray,
        time: datetime | np.datetime64,
        spacecraft_latitude: float,
        spacecraft_longitude: float,
        spacecraft_altitude: float,
        solar_zenith_angle: np.ndarray,
        relative_solar_azimuth_angle: np.ndarray,
        los_azimuth_angle: np.ndarray,
        processing: dict,
        saturated_fraction: np.ndarray | None = None,
        detector_rows: np.ndarray | None = None,
    ):
        """Build an image from arrays.

        radiance, radiance_noise, and wavenumber have shape (los, spectral_sample);
        a one-dimensional wavenumber is shared by every line of sight. processing
        holds the L1B processing attributes (apodization_window, ...).
        detector_rows defaults to 0, 1, 2, ...
        """
        spectral_dims = ["los", "spectral_sample"]
        radiance = np.asarray(radiance, dtype=float)
        if saturated_fraction is None:
            saturated_fraction = np.zeros(radiance.shape[0])
        if detector_rows is None:
            detector_rows = np.arange(radiance.shape[0])

        ds = xr.Dataset(
            {
                "radiance": (spectral_dims, radiance),
                "radiance_noise": (spectral_dims, radiance_noise),
                "tangent_altitude": ("los", tangent_altitude),
                "tangent_latitude": ("los", tangent_latitude),
                "tangent_longitude": ("los", tangent_longitude),
                "spacecraft_latitude": ((), float(spacecraft_latitude)),
                "spacecraft_longitude": ((), float(spacecraft_longitude)),
                "spacecraft_altitude": ((), float(spacecraft_altitude)),
                "solar_zenith_angle": ("los", solar_zenith_angle),
                "relative_solar_azimuth_angle": ("los", relative_solar_azimuth_angle),
                "los_azimuth_angle": ("los", los_azimuth_angle),
                "saturated_fraction": ("los", saturated_fraction),
            },
            coords={
                "time": np.datetime64(time, "ns"),
                "los": np.asarray(detector_rows, dtype=np.int64),
                "wavenumber": (
                    spectral_dims,
                    np.broadcast_to(wavenumber, radiance.shape).copy(),
                ),
            },
            attrs=dict(processing),
        )
        return cls(ds)

    def __init__(
        self,
        ds: xr.Dataset,
        low_alt=0,
        high_alt=100000,
        low_wvnumber_filter=0,
        high_wvnumber_filter=1e10,
        override_los=False,
        row_groups: list[tuple[int, ...]] | None = None,
    ):
        """``row_groups`` gives, for each line of sight of ``ds``, the detector rows
        averaged into it (see average_rows); by default each line of sight is its
        own ``los`` row."""
        if row_groups is not None and len(row_groups) != ds.sizes["los"]:
            msg = (
                f"Expected {ds.sizes['los']} row groups, one per line of sight, "
                f"got {len(row_groups)}"
            )
            raise ValueError(msg)
        self._row_groups = row_groups
        self._ds = ds
        self._low_alt = low_alt
        self._high_alt = high_alt
        self._low_wvnumber_filter = low_wvnumber_filter
        self._high_wvnumber_filter = high_wvnumber_filter
        self._override_los = override_los

    @property
    def ds(self):
        return self._ds

    def _retrieval_los_mask(self) -> np.ndarray:
        altitude = self._ds["tangent_altitude"].to_numpy()
        mask = (altitude > self._low_alt) & (altitude < self._high_alt)
        if "saturated_fraction" in self._ds:
            mask &= ~(self._ds["saturated_fraction"].to_numpy() > 0)
        if not np.any(mask):
            msg = "No unsaturated L1B rows remain within the retrieval altitude range"
            raise ValueError(msg)
        return mask

    def retrieval_rows(self) -> list:
        """Detector row of each line of sight used in the retrieval, or for
        averaged rows, the tuple of detector rows averaged into it."""
        mask = self._retrieval_los_mask()
        if self._row_groups is None:
            return list(self._ds["los"].to_numpy()[mask])
        return [
            group for group, keep in zip(self._row_groups, mask, strict=True) if keep
        ]

    def sk2_geometry(self) -> dict[sk.ViewingGeometry]:
        viewing_geo = sk.ViewingGeometry()

        good_alt = self._retrieval_los_mask()

        for i in range(np.count_nonzero(good_alt)):
            if self._override_los:
                obs_geo = WGS84()
                obs_geo.from_lat_lon_alt(
                    self._ds["spacecraft_latitude"],
                    self._ds["spacecraft_longitude"],
                    self._ds["spacecraft_altitude"],
                )

                tangent_geo = WGS84()
                tangent_geo.from_lat_lon_alt(
                    self._ds["tangent_latitude"].to_numpy()[good_alt][i],
                    self._ds["tangent_longitude"].to_numpy()[good_alt][i],
                    self._ds["tangent_altitude"].to_numpy()[good_alt][i],
                )

                look_v = tangent_geo.location - obs_geo.location
                look_v = look_v / np.linalg.norm(look_v)

                rv = np.cross(
                    obs_geo.local_up,
                    look_v,
                )
                rv /= np.linalg.norm(rv)
                rm = rotation_matrix(rv, np.deg2rad(0.21))

                look_v = rm @ look_v

                viewing_geo.add_ray(
                    ecef_to_sasktran2_ray(
                        obs_geo.location,
                        look_v,
                        pd.to_datetime(self._ds["time"].to_numpy()),
                        solar_handler=sk.solar.SolarGeometryHandlerAstropy(),
                    )
                )
            else:
                viewing_geo.add_ray(
                    sk.TangentAltitudeSolar(
                        self._ds["tangent_altitude"].to_numpy()[good_alt][i],
                        np.deg2rad(
                            self._ds["relative_solar_azimuth_angle"].to_numpy()[
                                good_alt
                            ][i]
                        ),
                        float(self._ds["spacecraft_altitude"]),
                        np.cos(
                            np.deg2rad(
                                self._ds["solar_zenith_angle"].to_numpy()[good_alt][i]
                            )
                        ),
                    )
                )

        return {"meas": viewing_geo}

    def skretrieval_l1(self, *args, **kwargs):
        good_alt = self._retrieval_los_mask()

        wavenumber = self._ds["wavenumber"].to_numpy()[good_alt]
        if np.any(wavenumber != wavenumber[0]):
            msg = "The retrieval needs the same wavenumber grid on every line of sight"
            raise NotImplementedError(msg)
        wvnum = wavenumber[0]
        good = (wvnum > self._low_wvnumber_filter) & (
            wvnum < self._high_wvnumber_filter
        )

        ds = xr.Dataset()
        for name in ["radiance", "radiance_noise"]:
            ds[name] = xr.DataArray(
                self._ds[name].to_numpy()[np.ix_(good_alt, good)].T,
                dims=["wavenumber", "los"],
            )
        ds["tangent_altitude"] = (
            ["los"],
            self._ds["tangent_altitude"].to_numpy()[good_alt],
        )
        ds.coords["wavenumber"] = wvnum[good]

        return {"meas": RadianceGridded(ds)}

    def sample_wavelengths(self) -> dict[np.array]:
        """
        The sample wavelengths for the observation in [nm]

        Returns
        -------
        dict[np.array]
        """
        l1 = self.skretrieval_l1()

        return {"meas": 1e7 / l1["meas"].data.wavenumber}

    def reference_cos_sza(self) -> dict[float]:
        """
        The reference cosine of the solar zenith angle for the observation

        Returns
        -------
        dict[float]
        """
        good_alt = self._retrieval_los_mask()
        mean_sza_deg = float(self._ds["solar_zenith_angle"][good_alt].mean())
        return {"meas": float(np.cos(np.deg2rad(mean_sza_deg)))}

    def reference_latitude(self) -> dict[float]:
        """
        The reference latitude for the observation

        Returns
        -------
        dict[float]
        """
        good_alt = self._retrieval_los_mask()
        return {"meas": float(self._ds["tangent_latitude"][good_alt].mean())}

    def reference_longitude(self) -> dict[float]:
        """
        The reference longitude for the observation

        Returns
        -------
        dict[float]
        """
        good_alt = self._retrieval_los_mask()
        return {"meas": float(self._ds["tangent_longitude"][good_alt].mean())}

    def append_information_to_l1(self, l1: dict[RadianceGridded], **kwargs) -> None:
        """
        A method that allows for the observation to append information to the L1 data
        simulated by the forward model. Useful for adding things that are in the real L1 data
        to the simulations that may be useful inside the measurement vector.

        Parameters
        ----------
        l1 : dict[RadianceGridded]
        """
        good_alt = self._retrieval_los_mask()
        l1["meas"].data["tangent_altitude"] = (
            ["los"],
            self._ds["tangent_altitude"].to_numpy()[good_alt],
        )


class L1bDataSet:
    """An L1B granule: consecutive images along time; see l1b.yaml."""

    def __init__(self, ds: xr.Dataset, name: str = "", parent: Path | None = None):
        check(ds, "l1b")
        self._ds = ds
        self._name = name
        self._parent = parent

    @classmethod
    def from_images(
        cls, images: list[L1bImage], name: str = "", parent: Path | None = None
    ) -> L1bDataSet:
        datasets = [image.ds for image in images]
        wavenumber = datasets[0]["wavenumber"].variable
        if any(not ds["wavenumber"].variable.equals(wavenumber) for ds in datasets):
            msg = "All images in a granule must share one wavenumber grid"
            raise ValueError(msg)
        if any(ds.attrs != datasets[0].attrs for ds in datasets[1:]):
            msg = "All images in a granule must have the same processing settings"
            raise ValueError(msg)

        ds = xr.concat(
            [ds.drop_vars("wavenumber") for ds in datasets],
            dim="time",
            data_vars="all",
            coords="minimal",
            join="exact",
            combine_attrs="override",
        ).assign_coords(wavenumber=wavenumber)
        return cls(apply_metadata(ds, "l1b"), name, parent)

    @classmethod
    def from_file(cls, file_path: Path) -> L1bDataSet:
        return cls(xr.load_dataset(file_path), file_path.stem, file_path.parent.parent)

    @property
    def ds(self):
        return self._ds

    def save(self, out_file: Path) -> None:
        self._ds.to_netcdf(out_file)

    def image(
        self,
        index: int,
        row_reduction: int = 1,
        low_alt=0,
        high_alt=100000,
        row_averaging: int = 1,
        **kwargs,
    ):
        """One image as a retrieval observation.

        ``row_reduction`` keeps every n-th row; ``row_averaging`` then averages
        consecutive groups of that many of the kept rows into one line of sight
        (see average_rows). Altitude limits apply to the resulting lines of sight.
        """
        ds = self._ds.isel(time=index).isel(los=slice(None, None, row_reduction))
        row_groups = None
        if row_averaging != 1:
            ds, row_groups = average_rows(ds, row_averaging)
        return L1bImage(
            ds, low_alt=low_alt, high_alt=high_alt, row_groups=row_groups, **kwargs
        )

    @property
    def l2_por_path(self):
        return self._parent.joinpath("por").joinpath(
            self._name.replace("L1B", "L2_POR").replace("Radiances", "Wvapor") + ".nc"
        )

    @property
    def l2_path(self):
        return self._parent.joinpath("l2").joinpath(
            self._name.replace("L1B", "L2").replace("Radiances", "Wvapor") + ".nc"
        )


# Per-line-of-sight angles averaged on the circle, since they wrap (the relative
# solar azimuth is near ±180 for backscatter)
_CIRCULAR_GEOMETRY = (
    "tangent_longitude",
    "relative_solar_azimuth_angle",
    "los_azimuth_angle",
)
_LINEAR_GEOMETRY = ("tangent_altitude", "tangent_latitude", "solar_zenith_angle")


def average_rows(
    image: xr.Dataset, num_rows: int
) -> tuple[xr.Dataset, list[tuple[int, ...]]]:
    """Average consecutive groups of ``num_rows`` lines of sight of one L1B image.

    Each group becomes one line of sight: the mean radiance, its noise
    sqrt(sum(radiance_noise**2)) / count (rows are independent), and the mean
    geometry, with the azimuths and longitude averaged on the circle. A group is
    saturated if any of its rows is (saturated_fraction is the largest). The last
    group may have fewer rows. Every row in a group must have the same wavenumber
    grid. ``los`` becomes the first detector row of each group, and the groups'
    detector rows are returned alongside, for the line shapes.

    The retrieval models a group as one line of sight at its mean geometry with
    the mean of its rows' line shapes, which is accurate while the radiance is
    close to linear in tangent altitude across the group.
    """
    if isinstance(num_rows, bool) or not isinstance(num_rows, int | np.integer):
        msg = f"row_averaging must be a positive integer, got {num_rows!r}"
        raise TypeError(msg)
    if num_rows < 1:
        msg = f"row_averaging must be a positive integer, got {num_rows}"
        raise ValueError(msg)

    rows = image["los"].to_numpy()
    starts = np.arange(0, len(rows), num_rows)
    groups = [
        tuple(int(row) for row in rows[start : start + num_rows]) for start in starts
    ]
    group = xr.DataArray(np.arange(len(rows)) // num_rows, dims="los")

    wavenumber = image["wavenumber"]
    first = wavenumber.isel(los=starts)
    if not np.array_equal(wavenumber.to_numpy(), first.to_numpy()[group.to_numpy()]):
        msg = "Averaged rows must share one wavenumber grid"
        raise ValueError(msg)

    def mean(name):
        return image[name].groupby(group).mean().rename(group="los")

    def circular_mean(name):
        angle = np.deg2rad(image[name])
        sin = np.sin(angle).groupby(group).mean()
        cos = np.cos(angle).groupby(group).mean()
        return np.rad2deg(np.arctan2(sin, cos)).rename(group="los")

    averaged = image.drop_dims("los").assign(
        radiance=mean("radiance"),
        radiance_noise=np.sqrt(
            (image["radiance_noise"] ** 2).groupby(group).sum()
        ).rename(group="los")
        / xr.DataArray([len(g) for g in groups], dims="los"),
        **{name: mean(name) for name in _LINEAR_GEOMETRY},
        **{name: circular_mean(name) for name in _CIRCULAR_GEOMETRY},
    )
    if "saturated_fraction" in image:
        averaged["saturated_fraction"] = (
            image["saturated_fraction"].groupby(group).max().rename(group="los")
        )
    averaged = averaged.assign_coords(
        los=("los", rows[starts]), wavenumber=first.variable
    )
    for name, lower in (
        ("relative_solar_azimuth_angle", -180.0),
        ("los_azimuth_angle", 0.0),
    ):
        averaged[name] = averaged[name].copy(data=wrap_degrees(averaged[name], lower))
    for name in averaged.variables:
        if name in image.variables:
            averaged[name].attrs = image[name].attrs
    averaged.attrs = image.attrs
    return (
        averaged.transpose(*[dim for dim in image.dims if dim in averaged.dims]),
        groups,
    )
