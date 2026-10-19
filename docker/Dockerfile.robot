# syntax=docker/dockerfile:1

# Local-only Robot Framework black-box/system tests (docker-compose.robot.yml,
# system_tests/) against the real recommender/serve/dashboard containers over
# HTTP. Standalone from docker/Dockerfile so it never affects that image's
# stages or its default `docker build` target.
FROM python:3.11-slim-bookworm

WORKDIR /app
COPY requirements-robot.txt ./
RUN pip install --no-cache-dir -r requirements-robot.txt
COPY system_tests/ ./system_tests/
ENTRYPOINT ["robot"]
CMD ["--outputdir", "system_tests/results", "system_tests"]
