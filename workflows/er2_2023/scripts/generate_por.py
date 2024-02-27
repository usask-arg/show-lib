from __future__ import annotations

from pathlib import Path

from showlib.por.l2.process import l2_por_merra

l2_por_merra(Path(snakemake.input[0]), Path(snakemake.output[0]).parent)
