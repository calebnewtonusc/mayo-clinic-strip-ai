# Dockerfile for Mayo Clinic STRIP AI inference API

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask flask-cors gunicorn

# Copy application code
COPY src/ ./src/
COPY deploy/ ./deploy/
COPY config/ ./config/
COPY setup.py .

# Install package in editable mode for proper imports
RUN pip install --no-cache-dir -e .

# Create directory for models
RUN mkdir -p /app/models

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=deploy/api_with_metrics.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run API (uses Flask dev server, or override with docker-compose)
# Note: Model must be loaded via main(), so direct Flask app import doesn't work with gunicorn
CMD ["python3", "deploy/api_with_metrics.py"]
