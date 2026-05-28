# MAO — HF Spaces compatible Dockerfile
# Runs FastAPI (port 8080, internal) + Streamlit (port 7860, public) in one container.
FROM python:3.11-slim

# System deps (runs as root — install before user switch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user exactly as HF Spaces expects (UID 1000)
RUN useradd -m -u 1000 user

# Use /home/user as the writable root — always owned by UID 1000 on HF Spaces
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/huggingface \
    DATA_DIR=/home/user/app/mao/data

WORKDIR /home/user/app

# Install Python deps (runs as user — goes into ~/.local)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source
COPY --chown=user . .

# Install project as editable package
RUN pip install --no-cache-dir --user -e .

# Pre-create all directories the app may write to at runtime
RUN mkdir -p /home/user/.cache/huggingface \
             /home/user/app/mao/data \
             /home/user/app/cache

# HF Spaces public port
EXPOSE 7860 8080

# start.sh launches FastAPI in background then Streamlit in foreground
CMD ["/bin/bash", "/home/user/app/start.sh"]
