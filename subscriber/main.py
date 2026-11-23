"""Consome transações do MQTT, pontua via API e persiste no Postgres."""

import json
import time

import paho.mqtt.client as mqtt
import requests
from sqlalchemy import create_engine, text

from src.common import config

DDL = """
CREATE TABLE IF NOT EXISTS transactions (
    tx_id TEXT PRIMARY KEY,
    received_at TIMESTAMPTZ DEFAULT now(),
    features JSONB,
    label INT,
    score DOUBLE PRECISION,
    is_fraud BOOLEAN,
    latency_ms DOUBLE PRECISION
)
"""

INSERT = text(
    """
    INSERT INTO transactions (tx_id, features, label, score, is_fraud, latency_ms)
    VALUES (:tx_id, CAST(:features AS JSONB), :label, :score, :is_fraud, :latency_ms)
    ON CONFLICT (tx_id) DO NOTHING
    """
)


def connect_db():
    engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
    while True:
        try:
            with engine.begin() as conn:
                conn.execute(text(DDL))
            return engine
        except Exception as exc:
            print(f"Postgres indisponível ({exc}); tentando de novo em 3s", flush=True)
            time.sleep(3)


engine = connect_db()


def score(features):
    """Chama a API. Retorna (score, is_fraud, latency_ms) ou (None, None, None) se a API falhar."""
    start = time.time()
    try:
        resp = requests.post(
            f"{config.API_URL}/predict", json={"features": features}, timeout=5
        )
        resp.raise_for_status()
        body = resp.json()
        return body["score"], body["is_fraud"], (time.time() - start) * 1000
    except Exception as exc:
        print(f"Falha ao pontuar: {exc}", flush=True)
        return None, None, None


def on_connect(client, userdata, flags, reason_code, properties):
    print(
        f"Conectado ao broker ({reason_code}); assinando {config.MQTT_TOPIC}",
        flush=True,
    )
    client.subscribe(config.MQTT_TOPIC, qos=1)


def on_message(client, userdata, msg):
    payload = json.loads(msg.payload)
    sc, is_fraud, latency = score(payload["features"])
    with engine.begin() as conn:
        conn.execute(
            INSERT,
            {
                "tx_id": payload["tx_id"],
                "features": json.dumps(payload["features"]),
                "label": payload.get("label"),
                "score": sc,
                "is_fraud": is_fraud,
                "latency_ms": latency,
            },
        )


if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    while True:
        try:
            client.connect(config.MQTT_HOST, config.MQTT_PORT)
            break
        except Exception as exc:
            print(f"Broker indisponível ({exc}); tentando de novo em 3s", flush=True)
            time.sleep(3)
    client.loop_forever()
