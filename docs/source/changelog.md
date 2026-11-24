(_changelog)=
# Changelog

## Unreleased

This release redefines the SHOW data products and the L1A → L1B → L2 processing.
Files and code written for v2026.10.0 need updating; see the breaking changes below.

### Breaking changes

#### Data formats
- L1A, L1B and calibration-database files are now defined by versioned YAML schemas
  (`showlib.schemas`: L1A 0.3.0, L1B 0.3.0, CAL_DB 0.4.0, L2 0.1.0) and are validated when
  they are loaded or written. Files written by v2026.10.0 cannot be read, and the ER-2
  legacy instrument-line-shape calibration layout is no longer supported.
- The calibration database now holds a parametric, pixel-integrated instrument model
  (OPD, fringe phase, Littrow wavenumber, etendue, visibility, transmission, detector
  parameters) instead of a stored line shape. `dark_offset` is a required field.
- L2 files have a schema. Renamed L2 fields:
  - `h2o_vmr_1sigma` → `h2o_vmr_noise_error` (it was always the noise-only error)
  - `averaging_kernel` → `h2o_vmr_averaging_kernel`

  New L2 fields: `h2o_vmr_solution_error`, an optional aerosol group
  (`aerosol_extinction` with its prior, errors and averaging kernel), diagnostics (`cost`,
  `optimizer_status`, `converged`, `num_function_evaluations`, altitude limits, `num_los`)
  and placeholder quality flags. The surface albedo is retrieved but not stored.
- The L2 `longitude` is the circular mean of the tangent longitudes, in [-180, 180).
  It was a plain mean, which put profiles that straddle 180° (or 0°, for longitudes in
  [0, 360)) on the far side of the Earth.

#### APIs
- Removed: `process_l1a_to_l1b_dataset`, `L1AFileWriter`, `L1bFileWriter`, `L2FileWriter`,
  and the old `showlib.l1b.l1b` and `showlib.l1b.l1b_processing` pipeline.
- `process_l1a_to_l1b(l1a, cal_db, apodization=10.0, noise_fraction=0.0, dark=static_dark)`
  processes an in-memory `L1ADataSet` with a `CalibrationDatabase` and returns an
  `L1bDataSet`; it no longer takes a file and an output folder.
- `L2DataSet` (with `from_profiles`, `from_file` and `save`) replaces `L2FileWriter`.
- `process_l1b_to_l2`, `process_l1b_to_l2_file` and `l2_preprocessor` take a `config`
  from `showlib.l2.config.load_retrieval_config` instead of keyword arguments. All retrieval
  settings now live in `showlib/l2/retrieval_config.yaml`. The new `retrieve_l2` runs the
  retrieval on an in-memory granule.
- `h2o_ret_optical_property` and `stratospheric_aerosol_optical_property` moved to
  `showlib.l2.retrieval`.

#### Retrieval behaviour
- Both stages use SASKTRAN2's `TwoStream` multiple-scatter source (was
  `DiscreteOrdinates` with 2 streams), set by `engine.multiple_scatter_source`.
- Aerosol is retrieved in the main retrieval, in log space, with its prior floored at
  1e-3 of its peak.
- The L1B noise correlation from apodization is used by whitening each line of sight
  (`retrieval.noise_correlation`, on by default).
- New defaults:
  - `minimizer.ftol` 1e-3 (was 1e-6) and `minimizer.max_nfev` 50 (was 20). At the old
    settings the H2O retrieval did not converge within 20 evaluations.
  - `retrieval.state.surface.lambertian_albedo.prior_influence` 1e4 (was 0), which holds
    the albedo at its initial value (0.3) to about 0.01. With 0 its level was
    unconstrained and the H2O noise error was erratic.
  - `preprocessor.wavenumber_cminv` is `auto` (the in-band sample with the least H2O
    absorption), so `load_retrieval_config()` works without overrides.

### New features
- Instrument model: dark signal (`dark_offset`, `dark_current`, and a `dark` callable in
  `process_l1a_to_l1b`), a grating-tilt option in `ideal_calibration_database`, and
  `showlib.cal_db.ideal.ideal_calibration_database` for simulations.
- `simulate_l1a` builds one row's forward operator at a time (13 GiB to 0.7 GiB peak memory
  for a 200-row image).
- L2 row averaging (`observation.row_averaging`).
- The pre-fit runs over its sample's line-shape window (`preprocessor.line_shape_tolerance`),
  and its lowest usable altitude accounts for the measurement noise
  (`preprocessor.low_altitude.noise_sigma`, `min_good_rows`).
- `SHOWLineShapes.save` and `SHOWLineShapes.load` keep line shapes for a calibration
  database on disk, and `retrieve_l2` and `process_l1b_to_l2_file` take them through
  `line_shapes`.
- Data-format pages are built from the schemas in the documentation.

### Dependencies
- Requires `skretrieval>=2026.11.0`. With 2026.10.0, showlib and aliprocessing share one
  state-element registry in a process, so an ALI retrieval run after importing showlib
  uses SHOW's 1300–1400 nm aerosol database.
- Requires `xarray>=2024.10` (for `xarray.DataTree`).

### Known issues
- The pre-fit holds H2O at its climatology. Where the true H2O differs, as near the
  tropopause, it leaves a hole in the aerosol prior that the main retrieval does not
  recover from.
