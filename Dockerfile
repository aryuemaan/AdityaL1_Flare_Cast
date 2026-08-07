# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Aditya-FlareCast container: core pipeline + FastAPI + dashboard.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps kept minimal; add build-essential only if a wheel needs compiling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve,dashboard,boost]"

# Copy the rest of the project.
COPY configs ./configs
COPY scripts ./scripts

# Non-root user.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Default: run the API. Override CMD to run the pipeline or dashboard.
CMD ["uvicorn", "aditya_flarecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
