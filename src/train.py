"""CLI de dados e treino. Uso: python -m src.train {ingest|prepare|baseline|compare|tune}

Chamado pela DAG `fraudazero_pipeline` (venv /opt/train-venv no contêiner do Airflow).
"""

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import config
from src.common.schema import FEATURES, TARGET

DATA = Path(config.DATA_DIR)
RAW = DATA / "raw" / "creditcard.csv"
TRAIN = DATA / "processed" / "train.csv"
STREAM = DATA / "stream" / "stream.csv"
BOOTSTRAP_FRACTION = (
    0.8  # primeiros 80% (por Time) = treino; últimos 20% = fluxo simulado
)


def ingest():
    """Valida a base bruta. TODO: incorporar ao treino as transações rotuladas gravadas no Postgres."""
    if not RAW.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {RAW}. Veja o README (passo 2)."
        )
    df = pd.read_csv(RAW, usecols=[TARGET])
    print(f"ingest: {len(df)} linhas, {int(df[TARGET].sum())} fraudes")


def prepare():
    """Ordena por Time e separa o bootstrap dataset (treino) do fluxo simulado (stream)."""
    df = pd.read_csv(RAW).sort_values("Time").reset_index(drop=True)
    cut = int(len(df) * BOOTSTRAP_FRACTION)
    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    STREAM.parent.mkdir(parents=True, exist_ok=True)
    df.iloc[:cut].to_csv(TRAIN, index=False)
    df.iloc[cut:].to_csv(STREAM, index=False)
    print(f"prepare: treino={cut} linhas, stream={len(df) - cut} linhas")


def _metrics(y_true, scores, threshold=0.5):
    pred = scores >= threshold
    return {
        "pr_auc": average_precision_score(y_true, scores),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred),
    }


def baseline():
    """Baseline: regressão logística com holdout temporal. Registra o modelo e marca o alias `champion`."""
    df = pd.read_csv(TRAIN)  # já ordenado por Time
    cut = int(len(df) * 0.8)
    train, valid = df.iloc[:cut], df.iloc[cut:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    model.fit(train[FEATURES], train[TARGET])
    metrics = _metrics(valid[TARGET], model.predict_proba(valid[FEATURES])[:, 1])

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("baseline")
    with mlflow.start_run(run_name="logreg-baseline"):
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "class_weight": "balanced",
                "split": "holdout-temporal-80-20",
            }
        )
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=config.MODEL_NAME
        )
    MlflowClient().set_registered_model_alias(
        config.MODEL_NAME, config.MODEL_ALIAS, info.registered_model_version
    )
    print("baseline:", {k: round(v, 4) for k, v in metrics.items()})


def compare():
    # TODO (Dados e ML): KNN (subamostra), Árvore, Random Forest, AdaBoost e MLP com stratified k-fold;
    # SMOTE/class_weight dentro do Pipeline; experimento "comparacao" no MLflow.
    raise NotImplementedError("compare: implementar a comparação de modelos")


def tune():
    # TODO (Dados e ML): Optuna ou Random Search no melhor modelo; experimento "otimizacao" no MLflow;
    # registrar o melhor modelo e mover o alias `champion` se superar o atual.
    raise NotImplementedError("tune: implementar a otimização de hiperparâmetros")


COMMANDS = {
    "ingest": ingest,
    "prepare": prepare,
    "baseline": baseline,
    "compare": compare,
    "tune": tune,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    COMMANDS[parser.parse_args().command]()
