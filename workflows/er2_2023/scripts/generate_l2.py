from __future__ import annotations

from pathlib import Path

from showlib.processing.l1b_to_l2 import process_l1b_to_l2

process_l1b_to_l2(
    Path(snakemake.input[0]), Path(snakemake.output[0]).parent, Path(snakemake.input[2])
)
