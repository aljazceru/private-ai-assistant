# Production-optimized multi-stage Dockerfile for Signal Bot
# Build stage: Includes all build tools and dependencies
FROM python:3.11-slim as builder

# Set build arguments for optimization
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment with optimized settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to latest version for faster dependency resolution
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY requirements.txt /tmp/requirements.txt

# Install dependencies with optimizations
# --compile: Compile .pyc files for faster startup
# --no-deps: Install only what's specified (already handled by requirements.txt)
RUN pip install --no-cache-dir \
    --compile \
    --requirement /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Runtime stage: Minimal final image
FROM python:3.11-slim as runtime

# Set runtime arguments
ARG TARGETPLATFORM

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with proper home directory
RUN groupadd -r signalbot && \
    useradd -r -g signalbot -d /app -s /bin/bash signalbot

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app"

# Create application directory with correct permissions
RUN mkdir -p /app && \
    chown signalbot:signalbot /app

WORKDIR /app

# Copy application files with correct ownership
COPY --chown=signalbot:signalbot signal_bot.py /app/
COPY --chown=signalbot:signalbot scripts/download_models.py /app/
COPY --chown=signalbot:signalbot src/ /app/src/

# Create directories needed by the application
RUN mkdir -p /app/models /app/logs && \
    chown -R signalbot:signalbot /app/models /app/logs

# Switch to non-root user
USER signalbot

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app'); import signal_bot; print('Signal bot is running')" || exit 1

# Set labels for metadata
LABEL maintainer="Signal Bot Team" \
      version="2.0.0" \
      description="Signal Bot with Sherpa-ONNX STT/TTS" \
      org.opencontainers.image.title="Signal Bot" \
      org.opencontainers.image.description="Signal messenger bot with local speech processing" \
      org.opencontainers.image.version="2.0.0"

# Expose the port that signal-cli-rest-api uses
EXPOSE 18380

# Default command
CMD ["python", "signal_bot.py"]
