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
FROM node:26.5.1-slim AS frontend

WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci

COPY src/cicerone/templates/ ./src/cicerone/templates/
COPY src/cicerone/static/dashboard.js ./src/cicerone/static/dashboard.js
COPY src/cicerone/static/tailwind.input.css ./tailwind.input.css
RUN npx tailwindcss -i ./tailwind.input.css -o ./tailwind.css --minify

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
COPY requirements.txt ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY src/ ./src/
COPY --from=frontend /build/tailwind.css ./src/cicerone/static/tailwind.css
COPY config/ ./config/
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN chmod +x ./docker/entrypoint.sh

ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /tmp/cicerone && chown -R cicerone:cicerone /app /tmp/cicerone
USER cicerone

# Batch mode (default): no HTTP surface, just a scheduled job (R2 in ->
# train -> R2 out), plus an optional event-driven trigger listener on 8080
# (see [job.trigger] in cicerone.toml). Serve mode (job.mode = "serve")
# instead exposes the read API on 8000. entrypoint.sh picks the right one
# based on the loaded config. The dashboard (cicerone.dashboard, port 8090)
# is always its own separate entrypoint/container -- see the "dashboard"
# service in docker-compose.yml, which overrides ENTRYPOINT directly rather
# than going through entrypoint.sh's batch/serve dispatch.
EXPOSE 8000 8080 8090
ENTRYPOINT ["./docker/entrypoint.sh"]

# --- test ---------------------------------------------------------------
# Never shipped/deployed: only built and run in CI (see .github/workflows/ci.yml
# and docker-compose.ci.yml). Adds pytest/coverage/moto on top of the runtime
# image so tests exercise the exact same dependency versions as production.
FROM runtime AS test
USER root
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY pyproject.toml ./
COPY tests/ ./tests/
# Canonical Postgres defaults — parsed by tests/postgres_defaults.py at runtime.
COPY docker/postgres/defaults.env ./docker/postgres/defaults.env
RUN chown -R cicerone:cicerone /app
USER cicerone
ENTRYPOINT []
CMD ["pytest", "--cov=cicerone", "--cov-report=term-missing", "--cov-report=xml"]
