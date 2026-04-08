FROM python:3.11-slim

# ---- metadata ----
LABEL maintainer="hudsonbunga"
LABEL description="IT Helpdesk OpenEnv — Scaler OpenEnv Hackathon"
LABEL version="1.0.0"

WORKDIR /app

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ---- python deps (cached layer) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- application code ----
COPY . .

# ---- non-root user (HF Spaces best practice) ----
RUN useradd -m -u 1000 appuser
USER appuser

# ---- runtime ----
EXPOSE 7860
ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --silent --fail http://localhost:7860/ || exit 1

CMD ["python", "inference.py"]