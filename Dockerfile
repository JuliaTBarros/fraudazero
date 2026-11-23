# Imagem compartilhada por: mlflow, api, subscriber, simulator e dashboard
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY subscriber ./subscriber
COPY embedded ./embedded
COPY app ./app
