FROM python:3.11-slim

# System deps for PyMuPDF and aiohttp
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create data / results dirs
RUN mkdir -p data results models

# Default: run the collector CLI
ENTRYPOINT ["python", "-m", "collector.ultimate_collector"]
CMD ["--help"]
