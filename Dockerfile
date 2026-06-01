# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies required by OpenCV (headless) and libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY app.py .
COPY load_model_compat.py .
COPY guide.png .

# Copy only the model file the app actually uses at runtime
COPY asl_recognition_new_model.keras .

# Streamlit listens on 8501 by default; Cloud Run expects $PORT (default 8080)
# We override both via ENV so Cloud Run can inject $PORT at startup
ENV PORT=8501

EXPOSE 8501

# Run Streamlit — bind to 0.0.0.0 so it's reachable from outside the container
CMD streamlit run app.py \
        --server.port=${PORT} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false
