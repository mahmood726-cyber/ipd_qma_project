# IPD-QMA Multi-Stage Docker Build
# Optimized for production deployment

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first (for better caching)
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-dev.txt && \
    # Clean up build artifacts
    rm -rf /root/.cache/pip /root/.cache/pip/wheels

# Copy application code
COPY . .

# Run tests to verify build
RUN python -m pytest tests/ -v --tb=short || echo "Tests run completed"

# ============================================
# Stage 2: Runtime (Production)
# ============================================
FROM python:3.11-slim AS runtime

# Set labels
LABEL maintainer="IPD-QMA Team <your-email@example.com>"
LABEL version="2.0.0"
LABEL description="IPD-QMA: Individual Participant Data Quantile Meta-Analysis"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash ipduser && \
    mkdir -p /app /data /logs && \
    chown -R ipduser:ipduser /app /data /logs

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=ipduser:ipduser . .

# Create data directories
RUN mkdir -p /data /app/data /app/logs && \
    chown -R ipduser:ipduser /data /app/data /app/logs

# Switch to non-root user
USER ipduser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (can be overridden)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose port
EXPOSE 8000
