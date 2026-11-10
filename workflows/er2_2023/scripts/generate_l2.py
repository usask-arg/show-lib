from __future__ import annotations

from pathlib import Path

from showlib.l2.config import load_retrieval_config
from showlib.processing.l1b_to_l2 import process_l1b_to_l2_file

process_l1b_to_l2_file(
    Path(snakemake.input[0]),
    Path(snakemake.output[0]).parent,
    Path(snakemake.input[2]),
    load_retrieval_config({"preprocessor": {"wavenumber_cminv": 7317.8}}),
)
