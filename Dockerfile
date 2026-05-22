# MAO — HF Spaces compatible Dockerfile
# Runs FastAPI (port 8080) + Streamlit (port 7860) in one container.
FROM python:3.11-slim

# HF Spaces requires non-root user
RUN useradd -m -u 1000 appuser

# System deps: spaCy, psycopg2, audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy model
RUN python -m spacy download en_core_web_sm

# Model cache inside /app — always writable by appuser, no /data permission issues
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache
ENV DATA_DIR=/app/mao/data

# Create cache dir as root before switching user
RUN mkdir -p /app/cache

# Copy source and set ownership
COPY --chown=appuser:appuser . .

# Install project as editable package
RUN pip install --no-cache-dir -e .

# Ensure cache is owned by appuser (belt-and-suspenders after COPY)
RUN chown -R appuser:appuser /app/cache

# HF Spaces requires port 7860 for the public-facing app
EXPOSE 7860 8080

USER appuser

# start.sh launches FastAPI in background then Streamlit in foreground
CMD ["/bin/bash", "/app/start.sh"]
