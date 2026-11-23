FROM apache/airflow:2.10.5-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Treino roda em um venv separado para não conflitar com as dependências do Airflow.
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/train-venv \
    && /opt/train-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && chmod -R a+rX /opt/train-venv

USER airflow
