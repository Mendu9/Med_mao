# MAO API — multi-stage build
FROM python:3.11-slim AS base

# System deps for spaCy, psycopg2, whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source
COPY . .

# Install project as package
RUN pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["uvicorn", "mao.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
