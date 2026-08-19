# syntax=docker/dockerfile:1

# --- builder ---------------------------------------------------------------
# LightFM ships Cython extensions that must be compiled from source on most
# platforms/architectures, so the build stage needs a toolchain. Kept out of
# the final image to keep it small and reduce attack surface.
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin pip/setuptools/wheel so the LightFM build is reproducible.
RUN pip install --no-cache-dir --upgrade pip==24.2 setuptools==74.1.2 wheel==0.44.0

COPY requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# --- frontend ----------------------------------------------------------------
# Compiles the dashboard's Tailwind CSS. This is the only place Node/npm
# exists anywhere in this project -- it's a Docker build stage, not a host
# requirement (everything ships/builds via `docker compose build`), and it
# never makes it into the runtime image below, which only gets the compiled
# static/tailwind.css. package-lock.json pins the exact dependency tree so
# this stage is reproducible, matching requirements.txt's exact-pin style.
# Pin Node minor/patch so 26.x floating tags cannot change the frontend build
# without an explicit bump in this file.
FROM node:26.7.0-slim AS frontend

WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci

COPY src/cicerone/templates/ ./src/cicerone/templates/
COPY src/cicerone/static/dashboard.js ./src/cicerone/static/dashboard.js
COPY src/cicerone/static/tailwind.input.css ./tailwind.input.css
RUN npx tailwindcss -i ./tailwind.input.css -o ./tailwind.css --minify

# --- package ---------------------------------------------------------------
# sdist + wheel for PyPI and the runtime image. Reuses the frontend CSS
# artifact; does not install the ML stack.
FROM python:3.11-slim-bookworm AS package

WORKDIR /app
RUN pip install --no-cache-dir build==1.5.0 twine==7.0.0

COPY pyproject.toml MANIFEST.in LICENSE README.md CHANGELOG.md ./
COPY requirements.txt requirements-redis.txt requirements-sequential.txt ./
COPY src/ ./src/
COPY --from=frontend /build/tailwind.css ./src/cicerone/static/tailwind.css

RUN python -m build \
    && python -m twine check dist/* \
    && PYTHONPATH=src python -m cicerone.packaging

# --- runtime -----------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# libgomp1: OpenMP runtime required by the compiled LightFM extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — never run the service as root.
RUN groupadd --gid 1000 cicerone && \
    useradd --uid 1000 --gid cicerone --shell /bin/bash --create-home cicerone

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY --from=package /app/dist /dist
COPY requirements.txt ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && pip install --no-cache-dir --no-index --find-links=/dist cicerone-recommender \
    && rm -rf /wheels /dist \
    && cicerone --help >/dev/null

COPY config/ ./config/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /tmp/cicerone && chown -R cicerone:cicerone /app /tmp/cicerone
USER cicerone

# Default: `cicerone start` (batch job + scheduler, or serve from job.mode).
# Override command per compose service, e.g. `--config … dashboard`.
# Config is `/app/config/cicerone.toml` unless `--config` / CICERONE_CONFIG_PATH.
EXPOSE 8000 8080 8090
ENTRYPOINT ["cicerone"]
CMD ["start"]

# --- test ---------------------------------------------------------------
# Never shipped/deployed: only built and run in CI (see .github/workflows/ci.yml
# and docker-compose.ci.yml). Adds pytest/coverage/moto on top of the runtime
# image so tests exercise the exact same dependency versions as production.
FROM runtime AS test
USER root
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY pyproject.toml MANIFEST.in ./
COPY requirements-redis.txt requirements-sequential.txt ./
COPY src/ ./src/
COPY --from=frontend /build/tailwind.css ./src/cicerone/static/tailwind.css
COPY tests/ ./tests/
# Checked-in serve OpenAPI schema — asserted against create_app().openapi() in tests.
COPY docs/openapi/ ./docs/openapi/
# Canonical Postgres defaults — parsed by tests/support/postgres_defaults.py at runtime.
COPY docker/postgres/defaults.env ./docker/postgres/defaults.env
ENV PYTHONPATH=/app/src
RUN chown -R cicerone:cicerone /app
USER cicerone
ENTRYPOINT []
CMD ["pytest", "--cov=cicerone", "--cov-report=term-missing", "--cov-report=xml"]
