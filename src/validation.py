"""Validação dos modelos: holdout temporal, k-fold estratificado e teste final no stream.

Uso típico (A6/A7): passe um Pipeline NÃO treinado; cada função faz `clone` e treina do zero,
então todo pré-processamento e reamostragem (SMOTE, undersampling) fica restrito aos dados de treino.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.common import config
from src.common.schema import FEATURES, TARGET

DATA = Path(config.DATA_DIR)
TRAIN = DATA / "processed" / "train.csv"
STREAM = DATA / "stream" / "stream.csv"
SEED = 42


def load_train(dedupe=True):
    """Bootstrap dataset em ordem temporal. Remove cópias exatas (840 linhas; ver docs/CONTEXTO.md)."""
    df = pd.read_csv(TRAIN)
    if dedupe:
        df = df.drop_duplicates().reset_index(drop=True)
    return df[FEATURES], df[TARGET]


def load_stream():
    """Últimos 20% por Time. Só para o teste final (A8): não use na escolha de modelo nem de hiperparâmetro."""
    df = pd.read_csv(STREAM)
    return df[FEATURES], df[TARGET]


def metrics(y_true, scores, threshold=0.5):
    """PR-AUC é a principal. Recall/precisão/F1 são da classe fraude no limiar dado. ROC-AUC só como referência."""
    pred = scores >= threshold
    return {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def _fit(model, X, y):
    start = time.perf_counter()
    fitted = clone(model).fit(X, y)
    return fitted, time.perf_counter() - start


def _scores(model, X):
    return model.predict_proba(X)[:, 1]


def holdout_temporal(model, X, y, frac=0.8, threshold=0.5):
    """Treina nos primeiros `frac` (por Time) e valida no restante: simula prever o futuro com o passado."""
    cut = int(len(X) * frac)
    fitted, fit_s = _fit(model, X.iloc[:cut], y.iloc[:cut])
    scores = _scores(fitted, X.iloc[cut:])
    return {
        "train": metrics(y.iloc[:cut], _scores(fitted, X.iloc[:cut]), threshold),
        "valid": metrics(y.iloc[cut:], scores, threshold),
        "fit_seconds": fit_s,
        "y_valid": y.iloc[cut:].to_numpy(),
        "scores_valid": scores,
    }


def cv_stratified(model, X, y, k=5, threshold=0.5, seed=SEED):
    """K-fold estratificado. Devolve um DataFrame com uma linha por fold (métricas de treino e validação)."""
    rows = []
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(folds.split(X, y), start=1):
        fitted, fit_s = _fit(model, X.iloc[tr], y.iloc[tr])
        train_m = metrics(y.iloc[tr], _scores(fitted, X.iloc[tr]), threshold)
        valid_m = metrics(y.iloc[va], _scores(fitted, X.iloc[va]), threshold)
        rows.append(
            {"fold": i, "fit_seconds": fit_s}
            | {f"train_{m}": v for m, v in train_m.items()}
            | {f"valid_{m}": v for m, v in valid_m.items()}
        )
    return pd.DataFrame(rows)


def summarize_cv(folds):
    """{coluna_mean, coluna_std} para cada métrica dos folds (formato pronto para mlflow.log_metrics)."""
    out = {}
    for col in folds.columns.drop("fold"):
        out[f"cv_{col}_mean"] = float(folds[col].mean())
        out[f"cv_{col}_std"] = float(folds[col].std())
    return out


def evaluate_on_stream(fitted_model, threshold=0.5):
    """Teste final de generalização no stream.csv (dados nunca vistos, posteriores no tempo).

    Usa os rótulos só aqui, offline. Rode UMA vez, com o campeão já escolhido (A8).
    """
    X, y = load_stream()
    scores = _scores(fitted_model, X)
    return {
        "metrics": metrics(y, scores, threshold),
        "y": y.to_numpy(),
        "scores": scores,
    }


def plot_confusion(y_true, scores, threshold=0.5, title="Matriz de confusão"):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        (scores >= threshold).astype(int),
        display_labels=["Legítima", "Fraude"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"{title} (limiar {threshold})")
    ax.grid(False)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    fig.tight_layout()
    return fig


def plot_pr_curve(y_true, scores, title="Curva precisão × recall", name="Modelo"):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    PrecisionRecallDisplay.from_predictions(
        y_true, scores, ax=ax, name=name, color="#2a78d6", plot_chance_level=True
    )
    ax.set_title(title)
    ax.set_xlabel("Recall (fraude)")
    ax.set_ylabel("Precisão (fraude)")
    fig.tight_layout()
    return fig


def gap(train_metrics, valid_metrics):
    """Diferença treino - validação por métrica (positivo e grande = sinal de overfitting)."""
    return {
        m: float(train_metrics[m] - valid_metrics[m])
        for m in train_metrics
        if m in valid_metrics
    }
