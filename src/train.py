"""CLI de dados e treino. Uso: python -m src.train {ingest|prepare|baseline|compare|tune}

Chamado pela DAG `fraudazero_pipeline` (venv /opt/train-venv no contêiner do Airflow).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import validation
from src.common import config
from src.common.schema import TARGET
from src.features import preprocessor

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


def evaluate_and_log(model, run_name, params, experiment, register=False):
    """Avalia um Pipeline NÃO treinado do jeito padrão do projeto e registra tudo no MLflow.

    Métricas registradas:
    - `pr_auc`, `roc_auc`, `recall`, `precision`, `f1`: holdout temporal 80/20 dentro do train.csv (principal)
    - `train_*`: as mesmas no treino do holdout (comparar com as de validação = diagnóstico de overfitting)
    - `cv_valid_*_mean/std` e `cv_train_*_mean/std`: stratified k-fold (k=5)
    Artefatos: matriz de confusão e curva PR (holdout), `cv_folds.csv` e o modelo treinado em todo o train.csv.
    Use o mesmo `experiment` para todos os modelos de uma comparação (ex.: "comparacao").
    """
    X, y = validation.load_train()
    hold = validation.holdout_temporal(model, X, y)
    folds = validation.cv_stratified(model, X, y)
    final = clone(model).fit(X, y)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            params
            | {
                "validation": "holdout-temporal-80-20 + stratified-kfold-5",
                "threshold": 0.5,
                "dedupe": True,
                "n_train_rows": len(X),
            }
        )
        mlflow.log_metrics(hold["valid"])
        mlflow.log_metrics({f"train_{m}": v for m, v in hold["train"].items()})
        mlflow.log_metric("fit_seconds", hold["fit_seconds"])
        mlflow.log_metrics(validation.summarize_cv(folds))
        mlflow.log_text(folds.to_csv(index=False), "cv_folds.csv")
        y_valid, scores = hold["y_valid"], hold["scores_valid"]
        for name, fig in [
            ("confusion_matrix.png", validation.plot_confusion(y_valid, scores)),
            ("pr_curve.png", validation.plot_pr_curve(y_valid, scores, name=run_name)),
        ]:
            mlflow.log_figure(fig, name)
            plt.close(fig)
        info = mlflow.sklearn.log_model(
            final,
            artifact_path="model",
            registered_model_name=config.MODEL_NAME if register else None,
            input_example=X.head(2),
        )
    pr = folds["valid_pr_auc"]
    print(f"{run_name} holdout:", {k: round(v, 4) for k, v in hold["valid"].items()})
    print(f"{run_name} cv-5: pr_auc = {pr.mean():.4f} ± {pr.std():.4f}")
    return {"holdout": hold, "folds": folds, "model": final, "info": info}


def baseline_model():
    """Regressão logística com class_weight='balanced' (desbalanceamento tratado no próprio modelo)."""
    return Pipeline(
        [
            ("prep", preprocessor()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def baseline():
    """Baseline registrado no MLflow. Vira `champion` só se ainda não houver um campeão."""
    result = evaluate_and_log(
        baseline_model(),
        run_name="logreg-baseline",
        params={
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "preprocessing": "hour-of-day sin/cos + log1p(Amount) + StandardScaler",
        },
        experiment="baseline",
        register=True,
    )
    _set_champion_if_missing(result["info"].registered_model_version)


def _set_champion_if_missing(version):
    """O baseline só vira `champion` na primeira vez; depois, nunca rebaixa um campeão já escolhido."""
    client = MlflowClient()
    try:
        current = client.get_model_version_by_alias(
            config.MODEL_NAME, config.MODEL_ALIAS
        )
        print(f"alias '{config.MODEL_ALIAS}' mantido na versão {current.version}")
    except MlflowException:
        client.set_registered_model_alias(
            config.MODEL_NAME, config.MODEL_ALIAS, version
        )
        print(f"alias '{config.MODEL_ALIAS}' -> versão {version}")


def compare():
    # TODO (A6): para cada modelo, montar um Pipeline e chamar
    #   evaluate_and_log(model, run_name=..., params=..., experiment="comparacao")
    # KNN: RandomUnderSampler DENTRO de um imblearn.pipeline.Pipeline (ver docs/CONTEXTO.md, etapa 8).
    raise NotImplementedError("compare: implementar a comparação de modelos")


def tune():
    # TODO (A7): Optuna ou Random Search no melhor modelo; experimento "otimizacao" no MLflow;
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
