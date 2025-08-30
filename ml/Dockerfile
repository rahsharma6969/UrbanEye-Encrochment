FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs for rasterio/fiona/gdal, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libjpeg-dev libpng-dev libpq-dev \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Make GDAL discoverable at build-time for wheels that look for it
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Install Python deps (except torch)
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# CPU torch by default; override in GPU image/override file
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy code
COPY . /app

# Cache dir for STAC/PC
ENV MPLCONFIGDIR=/tmp/mpl \
    NUMBA_CACHE_DIR=/tmp/numba

# Default command = just sleep (we’ll exec commands from Make/Compose)
CMD ["bash", "-lc", "sleep infinity"]

