FROM python:3.9-slim-bullseye

# Install dependencies needed for Ollama and other tools
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl wget gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Setup working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m -u 1000 coderunner && \
    chown -R coderunner:coderunner /app

RUN chmod 755 /app

# Set appropriate environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=0.0.0.0 \
    OLLAMA_MODELS=/app/models

# Create directory for Ollama models
RUN mkdir -p /app/models && \
    chown -R coderunner:coderunner /app/models

# Create sandbox directory
WORKDIR /app/sandbox
RUN chmod 700 /app/sandbox && \
    chown coderunner:coderunner /app/sandbox

USER coderunner

# Default command now includes starting Ollama server in background
CMD ["sh", "-c", "ollama serve & python -X dev runner.py"]