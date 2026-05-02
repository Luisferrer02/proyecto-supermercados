# syntax=docker/dockerfile:1.7

# ============================================================================
# Stage 1 — TypeScript build (Express server)
# ============================================================================
FROM node:24-bookworm-slim AS ts-build
WORKDIR /build

# Install only deps needed to compile
COPY web/backend/package.json web/backend/package-lock.json* ./
RUN npm ci

COPY web/backend/tsconfig.json ./
COPY web/backend/src ./src
RUN npx tsc

# Re-install only production deps for the runtime stage
RUN rm -rf node_modules && npm ci --omit=dev


# ============================================================================
# Stage 2 — Runtime: Python 3.12 + Node 20 + mlops/ + Express dist
# ============================================================================
# Express spawns Python child processes, so the runtime image needs both
# runtimes. Base on python:3.12-slim and add Node manually — smaller than
# starting from node:20 and adding Python.
FROM python:3.12-slim-bookworm AS runtime

ARG NODE_MAJOR=24
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_ENV=production \
    PORT=3001 \
    MLOPS_DIR=/app/mlops \
    RESULTS_DIR=/app/mlops/results \
    PYTHON_PATH=python3 \
    VENV_DIR=/opt/venv

# System deps:
#   - build-essential / gcc: occasionally required by chromadb / sqlite extensions
#   - curl, ca-certificates, gnupg: for the NodeSource setup script
#   - libgomp1: PyTorch needs libgomp at runtime
#   - tini: PID-1 signal handling so SIGTERM kills child Python procs cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg tini libgomp1 \
        build-essential gcc \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Python venv with mlops/ deps (named /opt/venv, picked up by VENV_DIR)
RUN python -m venv ${VENV_DIR}
ENV PATH="${VENV_DIR}/bin:${PATH}"

COPY mlops/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

WORKDIR /app

COPY --from=ts-build /build/dist            /app/web/backend/dist
COPY --from=ts-build /build/node_modules    /app/web/backend/node_modules
COPY --from=ts-build /build/package.json    /app/web/backend/package.json

COPY mlops/ /app/mlops/

# Install mlops as a package so utils/ and models/ are importable
RUN pip install -e /app/mlops/

# Don't bundle the local broken venvs — runtime venv lives at /opt/venv
RUN rm -rf /app/mlops/venv /app/mlops/venv-new /app/mlops/__pycache__ \
    && find /app/mlops -name '__pycache__' -type d -exec rm -rf {} +

# Non-root runtime user
RUN groupadd --system --gid 1001 app \
    && useradd  --system --uid 1001 --gid app --home /app --shell /sbin/nologin app \
    && mkdir -p /app/mlops/data/monthly /app/mlops/results /app/chromadb \
    && chown -R app:app /app /opt/venv

USER app
WORKDIR /app/web/backend

EXPOSE 3001
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["node", "dist/index.js"]
