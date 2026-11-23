"""API de inferência. Carrega o modelo `models:/<MODEL_NAME>@<MODEL_ALIAS>` do MLflow."""

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.common import config
from src.common.schema import FEATURES

app = FastAPI(title="FraudaZero API")
state = {"model": None}


class Transaction(BaseModel):
    features: dict[str, float]


def load_model():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    state["model"] = mlflow.sklearn.load_model(
        f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.post("/reload")
def reload_model():
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Modelo indisponível: {exc}")
    return {"status": "reloaded"}


@app.post("/predict")
def predict(tx: Transaction):
    if state["model"] is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Modelo indisponível: {exc}")

    missing = [f for f in FEATURES if f not in tx.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Features ausentes: {missing}")

    row = pd.DataFrame([[tx.features[f] for f in FEATURES]], columns=FEATURES)
    score = float(state["model"].predict_proba(row)[0, 1])
    return {
        "score": score,
        "is_fraud": score >= config.FRAUD_THRESHOLD,
        "threshold": config.FRAUD_THRESHOLD,
    }
