# Serving image: ONNX Runtime only, no PyTorch (keeps the image small).

FROM python:3.11-slim AS builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.onnx \
    THRESHOLD_PATH=/app/models/threshold.json
COPY models ./models
RUN useradd --create-home appuser
USER appuser
EXPOSE 8080
# Cloud Run injects PORT; default to 8080 locally.
CMD ["sh", "-c", "uvicorn inspection.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
