from __future__ import annotations

from pathlib import Path

import sasktran2 as sk


def solar_toon_file() -> Path:
    db_root = sk.appconfig.database_root()
    return db_root.joinpath("solar/toon/solar_merged_20160127_600_26316_100.out")


def solar_kurucz_folder() -> Path:
    db_root = sk.appconfig.database_root()
    return db_root.joinpath("solar/kurucz")
