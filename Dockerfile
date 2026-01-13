# Lightweight Dockerfile for cloud deployment
# For full ML features, use docker-compose locally with all dependencies

FROM python:3.11-slim

WORKDIR /app

# Install minimal API dependencies only (keeps image under 500MB)
COPY requirements-minimal.txt ./
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY start.py ./

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=8000

# Expose the API port
EXPOSE 8000

# Run the application via Python script (handles PORT env var)
CMD ["python", "start.py"]
