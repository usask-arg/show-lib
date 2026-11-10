"""Configuration for the SHOW L1B -> L2 retrieval; defaults in retrieval_config.yaml."""

from __future__ import annotations

import copy
import os
from importlib import resources
from pathlib import Path

import numpy as np
import sasktran2 as sk
import yaml

# Sections whose contents are passed to skretrieval and may hold any key
_FREE_FORM = "state"


def load_retrieval_config(overrides: dict | str | os.PathLike | None = None) -> dict:
    """The default retrieval settings with ``overrides`` merged in.

    ``overrides`` is a dict or a YAML file with the same layout as
    retrieval_config.yaml. Raises ValueError for unknown keys outside ``state``
    sections, and when a required setting is missing.
    """
    text = (
        resources.files(__package__)
        .joinpath("retrieval_config.yaml")
        .read_text("utf-8")
    )
    config = yaml.safe_load(text)
    if isinstance(overrides, str | os.PathLike):
        overrides = yaml.safe_load(Path(overrides).read_text("utf-8"))
    _merge(config, copy.deepcopy(overrides or {}), path="", free_form=False)

    if config["preprocessor"]["wavenumber_cminv"] is None:
        msg = (
            "preprocessor.wavenumber_cminv must be set to a weakly H2O-absorbing "
            "sample wavenumber in the instrument's band"
        )
        raise ValueError(msg)
    return config


def altitude_grid(config: dict) -> np.ndarray:
    """Retrieval altitude grid [m]: the union of the configured ranges."""
    return np.unique(
        np.concatenate(
            [
                np.arange(segment["start"], segment["stop"], segment["step"])
                for segment in config["altitude_grid_m"]
            ]
        ).astype(float)
    )


def engine_kwargs(config: dict, stage: str) -> dict:
    """Engine settings for ``stage`` ("preprocessor" or "retrieval")."""
    kwargs = {**config["engine"], **config[stage].get("engine", {})}
    kwargs["multiple_scatter_source"] = getattr(
        sk.MultipleScatterSource, kwargs["multiple_scatter_source"]
    )
    return kwargs


def _merge(base: dict, overrides: dict, path: str, free_form: bool) -> None:
    for key, value in overrides.items():
        location = f"{path}{key}"
        if key not in base and not free_form:
            msg = f"Unknown retrieval setting {location}"
            raise ValueError(msg)
        if isinstance(base.get(key), dict) and not isinstance(value, dict):
            msg = f"Retrieval setting {location} must be a mapping, got {value!r}"
            raise ValueError(msg)
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(
                base[key],
                value,
                f"{location}.",
                free_form=free_form or key == _FREE_FORM,
            )
        else:
            base[key] = value
