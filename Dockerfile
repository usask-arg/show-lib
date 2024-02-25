FROM quay.io/condaforge/mambaforge

ADD . /src/

RUN mamba env create -f /src/env.yml -n show_lib

SHELL ["mamba", "run", "-n", "show_lib", "/bin/bash", "-c"]

RUN pip install /src/
