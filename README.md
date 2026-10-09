# showlib

[![Documentation Status](https://readthedocs.org/projects/showlib/badge/?version=latest)](https://showlib.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/usask-arg/showlib/main.svg)](https://results.pre-commit.ci/latest/github/usask-arg/showlib/main)

Research and development libraries developed at the University of Saskatchewan for the SHOW instrument

## Installation
The package can be installed through

`pip install showlib`

## Usage
Documentation can be found at  https://showlib.readthedocs.io/

## Development
The development environment is managed with [uv](https://docs.astral.sh/uv/)

```
uv sync                       # create .venv with showlib and the dev tools
uv run pytest                 # run the tests
uv run pre-commit run -a      # lint and format
uv run --group docs sphinx-build -b html docs/source docs/build   # build the docs
```

## License
This project is licensed under the MIT license
