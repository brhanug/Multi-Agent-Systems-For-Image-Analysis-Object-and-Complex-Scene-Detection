# Dockerfile for Visual Historian MAS
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (OpenCV, etc)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy core pipeline scripts
COPY scripts/core_pipeline/ /app/scripts/core_pipeline/
COPY config.yaml /app/config.yaml

# Entrypoint for specific agents
# Usage: docker run visual-historian --agent 0
ENTRYPOINT ["python3", "scripts/core_pipeline/run_all_experiments_from_scratch.sh"]
