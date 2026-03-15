# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user before setting workdir
RUN groupadd --system appgroup && useradd --system --gid appgroup --no-create-home appuser

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data and results directories and hand ownership to the app user
RUN mkdir -p data results && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose Streamlit dashboard port
EXPOSE 8501

# Explicitly set the path for the data directory
ENV DATA_DIR=/app/data

# Default command
CMD ["python", "collector/ultimate_collector.py", "--help"]
