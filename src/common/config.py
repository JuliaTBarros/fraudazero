"""Configuração compartilhada: tudo vem de variáveis de ambiente, com padrões para o Docker Compose."""

import os

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "fraudazero/transactions")

API_URL = os.getenv("API_URL", "http://api:8000")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://fraudazero:fraudazero@postgres:5432/fraudazero",
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "fraudazero")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

DATA_DIR = os.getenv("DATA_DIR", "data")
