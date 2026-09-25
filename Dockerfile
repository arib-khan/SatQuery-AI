# AeroLens AI — Production FastAPI Backend Dockerfile (CPU-optimized)
FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    PYTHONPATH=/app:/app/src \
    # Force CPU mode — no GPU in this deployment
    SATQUERY_FORCE_CPU=1 \
    # Gradio is NOT loaded when running via FastAPI server
    SATQUERY_LOAD_GRADIO=0

WORKDIR /app

# Install system dependencies for OpenCV, PIL, RasterIO, and geospatial processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    # rasterio requires GDAL
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for Docker layer caching
COPY requirements-production.txt .

# Step 1: Install CPU-only PyTorch FIRST (prevents pip from downloading 2.5GB CUDA wheel)
# CPU torch wheel: ~220MB download vs ~2.5GB for CUDA build
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining production dependencies
# (torch is already installed, so it won't be re-downloaded)
RUN pip install --no-cache-dir -r requirements-production.txt

# Copy application source code and benchmark samples
COPY . .

# Expose FastAPI backend port
EXPOSE 8000

# Healthcheck — use /api/health (fast, no model check) during startup warmup
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl --fail http://localhost:8000/api/health || exit 1

# Launch AeroLens AI FastAPI server
CMD ["python", "-m", "uvicorn", "src.satquery.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
