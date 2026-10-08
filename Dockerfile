FROM python:3.12-trixie AS build

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# copy source code, including .git so setuptools-scm can determine the version
WORKDIR /app
COPY . .
# install showlib and its runtime dependencies into /app/.venv
RUN uv sync --no-dev --no-editable

FROM python:3.12-slim-trixie AS production
WORKDIR /app
# only copy the environment into the production container. The path needs to stay the same as in the
# build container since the venv scripts reference it
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
