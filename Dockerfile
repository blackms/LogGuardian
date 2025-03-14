FROM python:3.8-slim

LABEL maintainer="LogGuardian Team <support@logguardian.org>"
LABEL description="LogGuardian - LLM-based Log Anomaly Detection System"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash logguardian

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY setup.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir -e . 

# Create directories for data, models and logs
RUN mkdir -p /app/data /app/models /app/logs
RUN chown -R logguardian:logguardian /app

# Copy application code
COPY logguardian/ /app/logguardian/

# Set user
USER logguardian

# Set environment variable for model caching
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# Expose the API port
EXPOSE 8000

# Create entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command - can be overridden in docker-compose.yml
CMD ["api"]