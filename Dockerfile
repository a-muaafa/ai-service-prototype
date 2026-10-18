# ── Stage 1: Install dependencies ────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy application code
COPY main.py ./

# Copy model artifacts (must be generated BEFORE docker build)
COPY model.joblib model_metadata.json ./

# Fail fast with a clear message if the model is invalid
RUN python -c "\
import sys, joblib; \
try: \
    m = joblib.load('model.joblib'); \
    assert hasattr(m, 'predict'), 'Model has no predict method'; \
    print('✓ Model verified successfully'); \
except Exception as e: \
    print(f'✗ Model verification FAILED: {e}', file=sys.stderr); \
    sys.exit(1) \
"

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

# Health check using Python (no curl/wget in slim image)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
