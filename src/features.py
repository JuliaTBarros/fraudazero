"""Pré-processamento do modelo. Vai DENTRO do Pipeline, então é ajustado só nos dados de treino de cada fold.

A API recebe as mesmas FEATURES do contrato (incluindo `Time` bruto); a transformação acontece no modelo.
As funções ficam neste módulo (e não em lambdas) para o pickle do modelo encontrá-las na API e no Airflow.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

V_COLS = [f"V{i}" for i in range(1, 29)]
SECONDS_PER_DAY = 86400


def hour_of_day_cyclic(X):
    """`Time` (segundos desde a 1ª transação) -> hora do dia em seno/cosseno (23h fica perto de 0h)."""
    hour = (np.asarray(X, dtype=float) % SECONDS_PER_DAY) / 3600
    angle = 2 * np.pi * hour / 24
    return np.hstack([np.sin(angle), np.cos(angle)])


def _hour_names(transformer, input_features):
    return np.array(["hour_sin", "hour_cos"])


def preprocessor():
    """Time -> hora do dia (seno/cosseno); Amount -> log1p + padronização; V1..V28 -> padronização."""
    return ColumnTransformer(
        [
            (
                "time",
                FunctionTransformer(hour_of_day_cyclic, feature_names_out=_hour_names),
                ["Time"],
            ),
            (
                "amount",
                make_pipeline(
                    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    StandardScaler(),
                ),
                ["Amount"],
            ),
            ("v", StandardScaler(), V_COLS),
        ]
    )
