# ──────────────────────────────────────────────────────────────
#  APU Predictive Maintenance — Production Dockerfile
#  Port: 8080 (required by GCP Cloud Run)
#  Model / scaler NOT baked in — downloaded from GCS at startup
# ──────────────────────────────────────────────────────────────
FROM python:3.9-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

WORKDIR /app

# Install OS-level dependencies needed by LightGBM / psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Layer-cache friendly: copy requirements first ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──
COPY config/      ./config/
COPY src/         ./src/
COPY entrypoint/  ./entrypoint/
COPY Utils/       ./Utils/
COPY frontend/    ./frontend/

# Do NOT copy:
#   .APU_venv/         — not needed inside container
#   Artifacts/Model/   — model loaded from GCS at runtime
#   Artifacts/Scaler/  — scaler loaded from GCS at runtime
#   NoteBook/          — research code, not production
#   logs/              — generated at runtime
#   testing_datasets/  — local test data only

# Cloud Run requires apps to listen on 8080
EXPOSE 8080

CMD ["python", "frontend/app.py"]
