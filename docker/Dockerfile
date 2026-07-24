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
# based on the loaded config.
EXPOSE 8000 8080
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
RUN chown -R cicerone:cicerone /app
USER cicerone
ENTRYPOINT []
CMD ["pytest", "--cov=cicerone", "--cov-report=term-missing", "--cov-report=xml"]
