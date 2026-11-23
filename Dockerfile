# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip && /opt/venv/bin/pip install -r requirements.txt

# build the int8 embedding model; onnx and huggingface_hub are only needed for this step
COPY app ./app
COPY config ./config
RUN /opt/venv/bin/pip install onnx huggingface_hub \
 && /opt/venv/bin/python -m app.rag.embedder prepare /opt/models/minilm-int8 \
 && /opt/venv/bin/pip uninstall -y onnx huggingface_hub \
 && rm -rf /root/.cache/huggingface


FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    PATH="/opt/venv/bin:$PATH" EMBEDDER__MODEL_DIR=/opt/models/minilm-int8 \
    HOME=/home/user PORT=7860 MALLOC_ARENA_MAX=2

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=user /opt/models /opt/models
COPY --chown=user app ./app
COPY --chown=user config ./config
COPY --chown=user pipeline/__init__.py pipeline/schemas.py ./pipeline/
COPY --chown=user evals/report.json ./evals/report.json
# built by pipeline/build.py
COPY --chown=user data/app.db data/*.npy data/manifest.json ./data/

RUN mkdir -p runtime && chown user runtime
USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/healthz', timeout=4).status == 200 else 1)"

# one worker: the rate limiter, circuit breakers and SQLite writer live in-process
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
