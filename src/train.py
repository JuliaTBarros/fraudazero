"""CLI de dados e treino. Uso: python -m src.train {ingest|prepare|baseline|compare|tune|final|stream_test}

Chamado pela DAG `fraudazero_pipeline` (venv /opt/train-venv no contêiner do Airflow).
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

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


def evaluate_and_log(
    model, run_name, params, experiment, register=False, threshold=0.5
):
    """Avalia um Pipeline NÃO treinado do jeito padrão do projeto e registra tudo no MLflow.

    Protocolo (ver src/validation.py): desenvolvimento = primeiros 80% do train.csv; holdout = últimos 20%.
    Métricas registradas:
    - `pr_auc`, `roc_auc`, `recall`, `precision`, `f1`: holdout temporal (treina no desenvolvimento)
    - `train_*`: as mesmas no desenvolvimento (comparar com as de validação = diagnóstico de overfitting)
    - `cv_valid_*_mean/std` e `cv_train_*_mean/std`: stratified 5-fold SÓ no desenvolvimento
    - `fit_seconds` (treino no desenvolvimento), `predict_seconds` (pontuar o holdout) e `predict_ms_per_1k`
    Artefatos: matriz de confusão e curva PR (holdout), `cv_folds.csv`, `holdout_scores.csv`
    e o modelo treinado em todo o train.csv (desenvolvimento + holdout).
    Use o mesmo `experiment` para todos os modelos de uma comparação (ex.: "comparacao").
    """
    X, y = validation.load_train()
    X_dev, y_dev, _, _ = validation.split_dev_holdout(X, y)
    hold = validation.holdout_temporal(model, X, y, threshold=threshold)
    folds = validation.cv_stratified(model, X_dev, y_dev, threshold=threshold)
    tfolds = validation.cv_temporal(model, X_dev, y_dev, threshold=threshold)
    final = clone(model).fit(X, y)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            params
            | {
                "validation": "dev-stratified-kfold-5 + holdout-temporal-20",
                "threshold": threshold,
                "dedupe": True,
                "n_train_rows": len(X),
                "n_dev_rows": len(X_dev),
            }
        )
        n_hold = len(hold["y_valid"])
        mlflow.log_metrics(hold["valid"])
        mlflow.log_metrics({f"train_{m}": v for m, v in hold["train"].items()})
        mlflow.log_metric("fit_seconds", hold["fit_seconds"])
        mlflow.log_metric("predict_seconds", hold["predict_seconds"])
        mlflow.log_metric("predict_ms_per_1k", 1e6 * hold["predict_seconds"] / n_hold)
        mlflow.log_metrics(validation.summarize_cv(folds))
        mlflow.log_text(folds.to_csv(index=False), "cv_folds.csv")
        y_valid, scores = hold["y_valid"], hold["scores_valid"]
        _log_diagnostics(tfolds, y_valid, scores)
        mlflow.log_text(
            pd.DataFrame({"y": y_valid, "score": scores}).to_csv(index=False),
            "holdout_scores.csv",
        )
        for name, fig in [
            (
                "confusion_matrix.png",
                validation.plot_confusion(y_valid, scores, threshold),
            ),
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


def _log_diagnostics(tfolds, y_valid, scores):
    """Na run ativa: CV temporal (`cv_temporal_*`, `cv_temporal_folds.csv`) e IC 95% da PR-AUC do holdout."""
    lo, hi, _ = validation.bootstrap_ci(y_valid, scores)
    mlflow.log_metrics({"pr_auc_ci_low": lo, "pr_auc_ci_high": hi})
    mlflow.log_metrics(
        {
            k.replace("cv_", "cv_temporal_", 1): v
            for k, v in validation.summarize_cv(tfolds).items()
        }
    )
    mlflow.log_text(tfolds.to_csv(index=False), "cv_temporal_folds.csv")


def diagnose(*names):
    """Etapa 1 da sessão 2: acrescenta CV temporal e IC 95% do holdout às runs já existentes de `comparacao`.

    Reconstrói cada modelo pelo nome da run (configuração da A6) e grava as métricas na própria run.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    X, y = validation.load_train()
    X_dev, y_dev, _, _ = validation.split_dev_holdout(X, y)
    runs = mlflow.search_runs(
        experiment_names=["comparacao"], order_by=["attributes.start_time DESC"]
    ).drop_duplicates("tags.mlflow.runName")
    models = candidate_models()
    for _, run in runs.iterrows():
        name = run["tags.mlflow.runName"]
        if (names and name not in names) or name not in models:
            continue
        hold = pd.read_csv(
            mlflow.artifacts.download_artifacts(
                run_id=run.run_id, artifact_path="holdout_scores.csv"
            )
        )
        tfolds = validation.cv_temporal(models[name][0], X_dev, y_dev)
        with mlflow.start_run(run_id=run.run_id):
            _log_diagnostics(tfolds, hold["y"], hold["score"])
        print(
            f"{name}: temporal pr_auc = {tfolds.valid_pr_auc.mean():.4f} ± {tfolds.valid_pr_auc.std():.4f}",
            flush=True,
        )


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


SEED = validation.SEED
SMOTE_RATIO = (
    0.1  # SMOTE leva as fraudes a 10% das legítimas (1:10), só no treino de cada fold
)
KNN_UNDER_RATIO = (
    0.02  # KNN: todas as fraudes + legítimas sorteadas na razão 1:50, só no treino
)


def _pipe(clf, *samplers):
    """prep -> [reamostragem] -> clf. Com reamostragem, usa o Pipeline do imblearn (aplica só no fit)."""
    steps = [("prep", preprocessor()), *samplers, ("clf", clf)]
    return ImbPipeline(steps) if samplers else Pipeline(steps)


def _smote():
    return ("smote", SMOTE(sampling_strategy=SMOTE_RATIO, random_state=SEED))


def _under():
    return (
        "under",
        RandomUnderSampler(sampling_strategy=KNN_UNDER_RATIO, random_state=SEED),
    )


def candidate_models():
    """Modelos da A6: nome -> (Pipeline não treinado, parâmetros para o MLflow). Hiperparâmetros padrão razoáveis."""
    tree = {"max_depth": 10, "min_samples_leaf": 20}
    rf = {"n_estimators": 200, "min_samples_leaf": 1, "max_features": "sqrt"}
    ada = {"n_estimators": 200, "learning_rate": 0.5}
    mlp = {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "max_iter": 50}
    knn = {"n_neighbors": 15, "weights": "distance"}
    smote = {"imbalance": f"SMOTE({SMOTE_RATIO})"}
    under = {"imbalance": f"RandomUnderSampler({KNN_UNDER_RATIO})"}
    return {
        "logreg": (
            baseline_model(),
            {"model": "LogisticRegression", "imbalance": "class_weight=balanced"},
        ),
        "decision-tree": (
            _pipe(
                DecisionTreeClassifier(
                    **tree, class_weight="balanced", random_state=SEED
                )
            ),
            {"model": "DecisionTree", "imbalance": "class_weight=balanced"} | tree,
        ),
        "random-forest": (
            _pipe(
                RandomForestClassifier(
                    **rf, class_weight="balanced", n_jobs=-1, random_state=SEED
                )
            ),
            {"model": "RandomForest", "imbalance": "class_weight=balanced"} | rf,
        ),
        "adaboost-balanced": (
            _pipe(
                AdaBoostClassifier(
                    DecisionTreeClassifier(
                        max_depth=1, class_weight="balanced", random_state=SEED
                    ),
                    **ada,
                    random_state=SEED,
                )
            ),
            {"model": "AdaBoost", "imbalance": "stump class_weight=balanced"} | ada,
        ),
        "adaboost-smote": (
            _pipe(AdaBoostClassifier(**ada, random_state=SEED), _smote()),
            {"model": "AdaBoost"} | smote | ada,
        ),
        "mlp": (
            _pipe(MLPClassifier(**mlp, random_state=SEED)),
            {"model": "MLP", "imbalance": "none"} | mlp,
        ),
        "mlp-smote": (
            _pipe(MLPClassifier(**mlp, random_state=SEED), _smote()),
            {"model": "MLP"} | smote | mlp,
        ),
        "knn-under": (
            _pipe(KNeighborsClassifier(**knn, n_jobs=-1), _under()),
            {"model": "KNN"} | under | knn,
        ),
        "knn-under-smote": (
            _pipe(KNeighborsClassifier(**knn, n_jobs=-1), _under(), _smote()),
            {
                "model": "KNN",
                "imbalance": f"{under['imbalance']} + SMOTE({SMOTE_RATIO})",
            }
            | knn,
        ),
    }


def compare(*names):
    """A6: avalia cada modelo no protocolo padrão e registra uma run por modelo no experimento `comparacao`.

    `python -m src.train compare` roda todos; `python -m src.train compare mlp knn-under` roda só esses.
    Não registra versão do modelo nem mexe no `champion` (isso é da A8).
    """
    models = candidate_models()
    rows = []
    for name in names or models:
        model, params = models[name]
        r = evaluate_and_log(
            model, run_name=name, params=params, experiment="comparacao"
        )
        pr = r["folds"]["valid_pr_auc"]
        rows.append(
            {"model": name, "cv_pr_auc": f"{pr.mean():.4f} ± {pr.std():.4f}"}
            | {k: round(v, 4) for k, v in r["holdout"]["valid"].items()}
            | {"fit_s": round(r["holdout"]["fit_seconds"], 1)}
        )
    print(pd.DataFrame(rows).to_string(index=False))


BEST_PARAMS = Path(__file__).parent / "best_params.json"
FIGS = Path(__file__).parent.parent / "reports" / "figs"
TUNE_TRIALS = 30  # por estratégia; o 1º trial é a configuração padrão da A6
TUNE_CV = "temporal"  # objetivo da otimização: CV temporal no desenvolvimento (ver CONTEXTO, S2.4)
TUNE_STORAGE = f"sqlite:///{(DATA / 'optuna.db').as_posix()}"
RANDOM_SEARCH_MODEL = (
    "random-forest"  # também otimizado com Random Search (base de comparação)
)


def _layers(s):
    return tuple(int(u) for u in s.split("-"))


# Espaço de busca de cada modelo: trial -> parâmetros do Pipeline. DEFAULTS = valores da A6 (trial 0).
SPACES = {
    "random-forest": lambda t: {
        "clf__n_estimators": t.suggest_int("n_estimators", 100, 400, step=50),
        "clf__max_depth": t.suggest_categorical("max_depth", [None, 8, 12, 16, 24]),
        "clf__min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20, log=True),
        "clf__max_features": t.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
        "clf__class_weight": t.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", None]
        ),
    },
    "mlp-smote": lambda t: {
        "smote__sampling_strategy": t.suggest_float("smote_ratio", 0.01, 0.5, log=True),
        "clf__hidden_layer_sizes": _layers(
            t.suggest_categorical(
                "hidden_layer_sizes", ["32", "64-32", "128-64", "128-64-32"]
            )
        ),
        "clf__alpha": t.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "clf__learning_rate_init": t.suggest_float(
            "learning_rate_init", 1e-4, 1e-2, log=True
        ),
        "clf__batch_size": t.suggest_categorical("batch_size", [100, 200, 500]),
    },
    "knn-under": lambda t: {
        "clf__n_neighbors": t.suggest_int("n_neighbors", 5, 75, log=True),
        "clf__weights": t.suggest_categorical("weights", ["uniform", "distance"]),
        "under__sampling_strategy": t.suggest_float(
            "under_ratio", 0.005, 0.1, log=True
        ),
    },
}
DEFAULTS = {
    "random-forest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    "mlp-smote": {
        "smote_ratio": 0.1,
        "hidden_layer_sizes": "64-32",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "batch_size": 200,
    },
    "knn-under": {"n_neighbors": 15, "weights": "distance", "under_ratio": 0.02},
}


def build_model(name, params=None):
    """Pipeline do modelo `name` da A6 com os parâmetros do Pipeline sobrescritos (ex.: clf__max_depth)."""
    model = clone(candidate_models()[name][0])
    params = dict(params or {})
    if "clf__hidden_layer_sizes" in params:
        params["clf__hidden_layer_sizes"] = tuple(params["clf__hidden_layer_sizes"])
    return model.set_params(**params)


def _study(name, strategy, sampler, X_dev, y_dev, n_trials):
    """Uma otimização Optuna: maximiza a PR-AUC média da CV (TUNE_CV) no desenvolvimento.

    Os trials ficam em TUNE_STORAGE (SQLite em data/): se o processo cair, rodar de novo retoma o estudo.
    """
    import optuna

    cv = {"stratified": validation.cv_stratified, "temporal": validation.cv_temporal}

    def objective(trial):
        folds = cv[TUNE_CV](build_model(name, SPACES[name](trial)), X_dev, y_dev)
        trial.set_user_attr("cv_pr_auc_std", float(folds["valid_pr_auc"].std()))
        trial.set_user_attr("cv_train_pr_auc", float(folds["train_pr_auc"].mean()))
        trial.set_user_attr("fit_seconds", float(folds["fit_seconds"].mean()))
        return float(folds["valid_pr_auc"].mean())

    def progress(study, trial):
        print(
            f"  {name} {strategy} trial {trial.number}: {trial.value:.4f}"
            f" (melhor {study.best_value:.4f})",
            flush=True,
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=f"{name}-{strategy}-{TUNE_CV}",
        storage=TUNE_STORAGE,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )
    if not _completed(study):
        study.enqueue_trial(
            DEFAULTS[name]
        )  # 1º trial = configuração da A6 (referência)
    remaining = n_trials - len(_completed(study))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[progress])
    return study


def _completed(study):
    return [t for t in study.trials if t.value is not None]


def _default_value(name, study):
    """PR-AUC da configuração padrão da A6 (o trial enfileirado) dentro do estudo."""
    return next(t.value for t in _completed(study) if t.params == DEFAULTS[name])


def _plot_history(name, studies):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = {"optuna-tpe": "#2a78d6", "random-search": "#eb6834"}
    for strategy, study in studies.items():
        values = [t.value for t in _completed(study)]
        n = range(len(values))
        ax.scatter(n, values, s=14, color=colors[strategy], alpha=0.45, linewidths=0)
        ax.plot(
            n,
            pd.Series(values).cummax(),
            color=colors[strategy],
            lw=2,
            label=f"{strategy} (melhor até o trial)",
        )
    default = _default_value(name, study)
    ax.axhline(default, color="#52514e", ls="--", lw=1)
    ax.annotate(
        f"padrão da A6 = {default:.4f}",
        (len(values) - 1, default),
        ha="right",
        va="top",
        fontsize=9,
        color="#52514e",
    )
    ax.set_xlabel("Trial")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel(f"PR-AUC média (CV {TUNE_CV}, desenvolvimento)")
    ax.set_title(f"Otimização de hiperparâmetros: {name}")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def _plot_importance(name, study):
    import optuna

    imp = optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED)
    )
    s = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(s) + 1.2))
    ax.barh(s.index, s.values, color="#2a78d6", height=0.6)
    for i, v in enumerate(s.values):
        ax.annotate(
            f"{v:.2f}",
            (v, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Importância (fANOVA, soma = 1)")
    ax.set_title(f"Importância dos hiperparâmetros: {name} (Optuna TPE)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig, imp


def tune(*names, n_trials=TUNE_TRIALS):
    """A7: Optuna (TPE) nos modelos indicados; Random Search também em RANDOM_SEARCH_MODEL.

    Busca só no desenvolvimento, com a CV TUNE_CV (temporal: o k-fold estratificado premia memorização). A melhor configuração (pela CV) é
    avaliada no protocolo padrão e registrada no experimento `otimizacao`, e os parâmetros vão para
    src/best_params.json. Não mexe no `champion` (isso é da A8).
    """
    import optuna

    X, y = validation.load_train()
    X_dev, y_dev, _, _ = validation.split_dev_holdout(X, y)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("otimizacao")
    FIGS.mkdir(parents=True, exist_ok=True)
    for name in names or SPACES:
        samplers = {"optuna-tpe": optuna.samplers.TPESampler(seed=SEED)}
        if name == RANDOM_SEARCH_MODEL:
            samplers["random-search"] = optuna.samplers.RandomSampler(seed=SEED)
        studies = {}
        for strategy, sampler in samplers.items():
            start = time.perf_counter()
            study = _study(name, strategy, sampler, X_dev, y_dev, n_trials)
            elapsed = time.perf_counter() - start
            studies[strategy] = study
            default = _default_value(name, study)
            with mlflow.start_run(run_name=f"{name}-{strategy}"):
                mlflow.log_params(
                    {
                        "model": name,
                        "strategy": strategy,
                        "n_trials": len(_completed(study)),
                        "tune_cv": TUNE_CV,
                    }
                    | {f"best_{k}": v for k, v in study.best_params.items()}
                )
                mlflow.log_metrics(
                    {
                        "best_cv_pr_auc": study.best_value,
                        "default_cv_pr_auc": default,
                        "gain_cv_pr_auc": study.best_value - default,
                        "best_trial": study.best_trial.number,
                        "search_seconds": elapsed,
                    }
                )
                mlflow.log_text(
                    study.trials_dataframe().to_csv(index=False), "trials.csv"
                )
                for t in _completed(study):
                    with mlflow.start_run(run_name=f"trial-{t.number}", nested=True):
                        mlflow.log_params(t.params)
                        mlflow.log_metrics({"cv_pr_auc": t.value} | t.user_attrs)
            print(
                f"{name} {strategy}: padrão {default:.4f} -> melhor {study.best_value:.4f}"
                f" (trial {study.best_trial.number}) em {elapsed / 60:.1f} min; {study.best_params}"
            )

        fig = _plot_history(name, studies)
        fig.savefig(FIGS / f"opt_{name}_historico.png", dpi=150)
        plt.close(fig)
        fig, imp = _plot_importance(name, studies["optuna-tpe"])
        fig.savefig(FIGS / f"opt_{name}_importancia.png", dpi=150)
        plt.close(fig)
        print(f"{name} importância:", {k: round(v, 3) for k, v in imp.items()})

        strategy, best = max(studies.items(), key=lambda kv: kv[1].best_value)
        params = SPACES[name](optuna.trial.FixedTrial(best.best_params))
        evaluate_and_log(
            build_model(name, params),
            run_name=f"{name}-tuned",
            params={"model": name, "strategy": strategy}
            | {k: str(v) for k, v in params.items()},
            experiment="otimizacao",
        )
        saved = json.loads(BEST_PARAMS.read_text()) if BEST_PARAMS.exists() else {}
        saved.setdefault("models", {})[name] = {
            "params": params,
            "strategy": strategy,
            "cv_pr_auc": best.best_value,
            "default_cv_pr_auc": _default_value(name, best),
            "tune_cv": TUNE_CV,
        }
        BEST_PARAMS.write_text(json.dumps(saved, indent=2) + "\n")


def final():
    """A8/B6: retreina o campeão de src/best_params.json em todo o train.csv, registra e move o `champion`.

    O limiar escolhido na A8 vai como parâmetro da run e como tag `threshold` da versão do modelo.
    Determinístico (random_state=42): rodar de novo reproduz o mesmo modelo e as mesmas métricas.
    """
    cfg = json.loads(BEST_PARAMS.read_text())
    name, threshold = cfg["champion"], cfg["threshold"]
    params = cfg["models"].get(name, {}).get("params", {})
    result = evaluate_and_log(
        build_model(name, params),
        run_name=f"{name}-champion",
        params={"model": name} | {k: str(v) for k, v in params.items()},
        experiment="campeao",
        register=True,
        threshold=threshold,
    )
    version = result["info"].registered_model_version
    client = MlflowClient()
    client.set_model_version_tag(
        config.MODEL_NAME, version, "threshold", str(threshold)
    )
    client.set_registered_model_alias(config.MODEL_NAME, config.MODEL_ALIAS, version)
    print(
        f"alias '{config.MODEL_ALIAS}' -> versão {version} ({name}, limiar {threshold})"
    )


def stream_test():
    """A8: teste final no stream.csv, UMA única vez, com o `champion` já escolhido e o seu limiar.

    Registra as métricas `stream_*` (no limiar escolhido e em 0,5), a matriz de confusão, a curva PR
    e os scores na própria run do campeão.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
    threshold = float(mv.tags["threshold"])
    model = mlflow.sklearn.load_model(
        f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
    )
    res = validation.evaluate_on_stream(model, threshold=threshold)
    y, scores = res["y"], res["scores"]
    at_05 = validation.metrics(y, scores, 0.5)
    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_metrics({f"stream_{k}": v for k, v in res["metrics"].items()})
        mlflow.log_metrics({f"stream_{k}_at_0.5": v for k, v in at_05.items()})
        mlflow.log_text(
            pd.DataFrame({"y": y, "score": scores}).to_csv(index=False),
            "stream_scores.csv",
        )
        for fname, fig in [
            (
                "stream_confusion_matrix.png",
                validation.plot_confusion(y, scores, threshold, "Stream (teste final)"),
            ),
            (
                "stream_pr_curve.png",
                validation.plot_pr_curve(
                    y, scores, "Curva PR no stream", f"campeão v{mv.version}"
                ),
            ),
        ]:
            mlflow.log_figure(fig, fname)
            fig.savefig(FIGS / f"a8_{fname}", dpi=150)  # cópia para o relatório
            plt.close(fig)
    print(f"stream (versão {mv.version}, limiar {threshold}):", res["metrics"])
    print("stream (limiar 0,5):", at_05)


COMMANDS = {
    "ingest": ingest,
    "prepare": prepare,
    "baseline": baseline,
    "compare": compare,
    "diagnose": diagnose,
    "tune": tune,
    "final": final,
    "stream_test": stream_test,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("names", nargs="*", help="compare/tune: só estes modelos")
    args = parser.parse_args()
    COMMANDS[args.command](*args.names)
