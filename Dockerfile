FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user \
    --timeout 100 \
    --retries 5 \
    -r requirements.txt


# Final stage: slim runtime image
FROM python:3.12-slim

RUN groupadd -g 1001 appuser \
    && useradd -u 1001 -g appuser -m -s /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Application code
COPY --chown=appuser:appuser app ./app

RUN mkdir -p /home/appuser/scikit_learn_data \
    && chown -R appuser:appuser /home/appuser/scikit_learn_data

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
