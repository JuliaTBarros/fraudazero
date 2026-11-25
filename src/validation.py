"""Validação dos modelos: protocolo de avaliação em três conjuntos, sem vazamento.

- Desenvolvimento = primeiros 80% (por Time) do train.csv. Stratified 5-fold e TODA otimização de
  hiperparâmetros acontecem só aqui (`split_dev_holdout` + `cv_stratified`).
- Holdout temporal = últimos 20% do train.csv. Compara modelos e escolhe o campeão e o limiar
  (`holdout_temporal`: treina no desenvolvimento, valida no holdout).
- stream.csv = teste final, usado UMA vez, só no campeão (`evaluate_on_stream`).

Uso típico (A6/A7): passe um Pipeline NÃO treinado; cada função faz `clone` e treina do zero,
então todo pré-processamento e reamostragem (SMOTE, undersampling) fica restrito aos dados de treino.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from src.common import config
from src.common.schema import FEATURES, TARGET

DATA = Path(config.DATA_DIR)
TRAIN = DATA / "processed" / "train.csv"
STREAM = DATA / "stream" / "stream.csv"
SEED = 42

# Estilo único das figuras (notebooks, MLflow e reports/figs): fundo claro, grade discreta, sem bordas extras.
PLOT_STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "axes.titlecolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "text.color": "#0b0b0b",
    "axes.grid": True,
    "grid.color": "#e8e7e3",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}
plt.rcParams.update(PLOT_STYLE)
# primeiros 80% do train.csv (por Time) = desenvolvimento; os 20% restantes = holdout temporal
DEV_FRACTION = 0.8


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


def split_dev_holdout(X, y, frac=DEV_FRACTION):
    """Corte temporal: (X_dev, y_dev, X_hold, y_hold). Os dados já vêm em ordem de Time."""
    cut = int(len(X) * frac)
    return X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]


def _fit(model, X, y):
    start = time.perf_counter()
    fitted = clone(model).fit(X, y)
    return fitted, time.perf_counter() - start


def _scores(model, X):
    return model.predict_proba(X)[:, 1]


def holdout_temporal(model, X, y, frac=DEV_FRACTION, threshold=0.5):
    """Treina no desenvolvimento (passado) e valida no holdout (futuro). Recebe o train.csv inteiro."""
    X_dev, y_dev, X_hold, y_hold = split_dev_holdout(X, y, frac)
    fitted, fit_s = _fit(model, X_dev, y_dev)
    start = time.perf_counter()
    scores = _scores(fitted, X_hold)
    predict_s = time.perf_counter() - start
    return {
        "train": metrics(y_dev, _scores(fitted, X_dev), threshold),
        "valid": metrics(y_hold, scores, threshold),
        "fit_seconds": fit_s,
        "predict_seconds": predict_s,
        "y_valid": y_hold.to_numpy(),
        "scores_valid": scores,
        "model": fitted,
    }


def _cv(model, X, y, splits, threshold):
    rows = []
    for i, (tr, va) in enumerate(splits, start=1):
        fitted, fit_s = _fit(model, X.iloc[tr], y.iloc[tr])
        train_m = metrics(y.iloc[tr], _scores(fitted, X.iloc[tr]), threshold)
        valid_m = metrics(y.iloc[va], _scores(fitted, X.iloc[va]), threshold)
        rows.append(
            {"fold": i, "fit_seconds": fit_s}
            | {f"train_{m}": v for m, v in train_m.items()}
            | {f"valid_{m}": v for m, v in valid_m.items()}
        )
    return pd.DataFrame(rows)


def cv_stratified(model, X, y, k=5, threshold=0.5, seed=SEED):
    """K-fold estratificado. Devolve um DataFrame com uma linha por fold (métricas de treino e validação).

    Passe só o conjunto de DESENVOLVIMENTO (`split_dev_holdout`): o holdout não pode entrar nos folds.
    """
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return _cv(model, X, y, folds.split(X, y), threshold)


def cv_temporal(model, X, y, k=5, threshold=0.5):
    """Validação temporal (TimeSeriesSplit): cada split treina no passado e valida no bloco seguinte.

    Mesmo formato do `cv_stratified`. Passe só o DESENVOLVIMENTO, em ordem de Time.
    """
    return _cv(model, X, y, TimeSeriesSplit(n_splits=k).split(X), threshold)


def bootstrap_ci(y_true, scores, n=1000, seed=SEED, alpha=0.05):
    """IC (1 - alpha) da PR-AUC por bootstrap das linhas. Devolve (baixo, alto, amostras)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        if y_true[idx].any():
            stats.append(average_precision_score(y_true[idx], scores[idx]))
    stats = np.array(stats)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), stats


def summarize_cv(folds):
    """{coluna_mean, coluna_std} para cada métrica dos folds (formato pronto para mlflow.log_metrics)."""
    out = {}
    for col in folds.columns.drop("fold"):
        out[f"cv_{col}_mean"] = float(folds[col].mean())
        out[f"cv_{col}_std"] = float(folds[col].std())
    return out


def choose_threshold(y_true, scores):
    """Limiar que maximiza o F1 da classe fraude (critério da A8). Devolve (limiar, precisão, recall, F1)."""
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(t[i]), float(p[i]), float(r[i]), float(f1[i])


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
