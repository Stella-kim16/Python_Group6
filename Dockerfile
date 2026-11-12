# ==============================================================================
# Stage 1: Builder — Install dependencies into an isolated prefix for caching
# ==============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Copy only requirements.txt first to leverage Docker layer caching.
# Changes to application code in subsystem1/2/3 will NOT invalidate this layer.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ==============================================================================
# Stage 2: Runtime — Minimal production image with non-root user
# ==============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Copy pre-built Python packages from the builder stage
COPY --from=builder /install /usr/local

# Create a non-root user and group for container security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --create-home --shell /bin/bash appuser

# Copy application source code and raw dataset
COPY subsystem1/ ./subsystem1/
COPY subsystem2/ ./subsystem2/
COPY subsystem3/ ./subsystem3/
COPY data/ ./data/
COPY main.py app.py ./

# Ensure the non-root user owns the working directory and data volumes
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8501

# Container-level healthcheck using Python stdlib (no extra curl/wget binary needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Default command launches the Streamlit web interface
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
