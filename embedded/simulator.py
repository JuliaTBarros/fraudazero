"""Simulador do sistema embarcado: reproduz o fluxo `data/stream/stream.csv` publicando no MQTT."""

import json
import os
import time
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt

from src.common import config
from src.common.schema import FEATURES, TARGET

STREAM = Path(config.DATA_DIR) / "stream" / "stream.csv"
INTERVAL = float(os.getenv("SIM_INTERVAL_S", "0.5"))

while not STREAM.exists():  # o arquivo é gerado pela DAG (tarefa `prepare`)
    print(f"Aguardando {STREAM} (rode a DAG no Airflow)...", flush=True)
    time.sleep(10)

df = pd.read_csv(STREAM)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
while True:
    try:
        client.connect(config.MQTT_HOST, config.MQTT_PORT)
        break
    except Exception as exc:
        print(f"Broker indisponível ({exc}); tentando de novo em 3s", flush=True)
        time.sleep(3)
client.loop_start()

lap = 0
while True:  # ao acabar o arquivo, recomeça (tx_id inclui o número da volta)
    for i, row in enumerate(df.to_dict("records")):
        payload = {
            "tx_id": f"tx-{lap}-{i}",
            "ts": time.time(),
            "features": {f: float(row[f]) for f in FEATURES},
            "label": int(row[TARGET]),
        }
        client.publish(config.MQTT_TOPIC, json.dumps(payload), qos=1)
        time.sleep(INTERVAL)
    lap += 1
