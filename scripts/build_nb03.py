"""Monta notebooks/03_comparacao.ipynb (código + markdown).

Uso: python scripts/build_nb03.py . [interpretacoes.json]
O JSON opcional traz o texto das células de interpretação (chave -> markdown).
"""

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
MD = (
    json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    if len(sys.argv) > 2
    else {}
)

cells = []


def md(key, default="_(interpretação a preencher)_"):
    cells.append(("markdown", MD.get(key, default)))


def code(src):
    cells.append(("code", src.strip("\n")))


md(
    "intro",
    """# FraudaZero — 03. Comparação de modelos (A6)

Compara regressão logística (baseline), Árvore de Decisão, Random Forest, AdaBoost, MLP e KNN no protocolo de avaliação do projeto (`notebooks/02_experimentos.ipynb`, seção 6):

| Conjunto | Papel |
|---|---|
| Desenvolvimento (primeiros 80% do `train.csv`) | Stratified 5-fold (os mesmos folds para todos) e 5-fold **temporal** (`TimeSeriesSplit`) |
| Holdout temporal (últimos 20% do `train.csv`) | Comparação dos modelos treinados no desenvolvimento, com IC 95% por bootstrap (1.000 reamostragens) |
| `stream.csv` | **Não é usado aqui** (teste final do campeão, A8) |

**Como reproduzir:** com a stack de pé, `python -m src.train compare` (registra uma run por modelo no experimento `comparacao` do MLflow). Este notebook só **lê** essas runs (métricas, `cv_folds.csv`, `cv_temporal_folds.csv` e `holdout_scores.csv`), então todo número abaixo vem do MLflow.""",
)
code(
    """
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATA_DIR", str(ROOT / "data"))
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
from src.validation import PLOT_STYLE

plt.rcParams.update(PLOT_STYLE)

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
FIGS = ROOT / "reports" / "figs"

# Uma cor por família de modelo, na ordem fixa da paleta categórica; variantes compartilham a cor.
FAMILY = {
    "logreg": "Regressão logística",
    "decision-tree": "Árvore de Decisão",
    "random-forest": "Random Forest",
    "adaboost": "AdaBoost",
    "mlp": "MLP",
    "knn": "KNN",
}
COLOR = dict(zip(FAMILY, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
ORDER = [
    "logreg", "decision-tree", "random-forest", "adaboost-balanced", "adaboost-smote",
    "mlp", "mlp-smote", "knn-under", "knn-under-smote",
]


def family(run):
    return next(f for f in FAMILY if run.startswith(f))
"""
)
md("load_md", "## 1. Resultados registrados no MLflow")
code(
    """
exp = mlflow.get_experiment_by_name("comparacao")
runs = mlflow.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"])
runs = runs.drop_duplicates("tags.mlflow.runName").set_index("tags.mlflow.runName")
runs = runs.loc[[r for r in ORDER if r in runs.index]]

folds, tfolds, scores = {}, {}, {}
for name, run_id in runs["run_id"].items():
    folds[name] = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="cv_folds.csv"))
    tfolds[name] = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="cv_temporal_folds.csv"))
    scores[name] = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="holdout_scores.csv"))

def m(c):
    return runs[f"metrics.{c}"]


table = pd.DataFrame(
    {
        "tratamento": runs["params.imbalance"],
        "estratificado": m("cv_valid_pr_auc_mean"),
        "estrat. desvio": m("cv_valid_pr_auc_std"),
        "temporal": m("cv_temporal_valid_pr_auc_mean"),
        "temp. desvio": m("cv_temporal_valid_pr_auc_std"),
        "holdout PR-AUC": m("pr_auc"),
        "IC 95% baixo": m("pr_auc_ci_low"),
        "IC 95% alto": m("pr_auc_ci_high"),
        "holdout ROC-AUC": m("roc_auc"),
        "recall@0,5": m("recall"),
        "precisão@0,5": m("precision"),
        "F1@0,5": m("f1"),
        "treino (s)": m("fit_seconds"),
        "predição (ms / mil tx)": m("predict_ms_per_1k"),
    }
)
print("Runs:", dict(runs["run_id"].str[:8]))
table.round(4)
"""
)
md("table")

md(
    "imb_md",
    """## 2. Tratamento do desbalanceamento, por modelo

Regra: `class_weight="balanced"` onde o modelo aceita (logística, árvore, Random Forest). Onde não aceita diretamente (AdaBoost, MLP, KNN), testamos SMOTE **dentro do `imblearn.Pipeline`** (só no treino de cada fold) contra uma alternativa: no MLP, sem tratamento; no KNN, só a subamostra; no AdaBoost, o stump com `class_weight="balanced"` como estimador base. A escolha da variante usa **só o desenvolvimento** (escolher pelo holdout gastaria o holdout numa decisão de hiperparâmetro) e o **5-fold temporal**, que é o que estima o desempenho futuro (seção 4).""",
)
code(
    """
pairs = [("adaboost-balanced", "adaboost-smote"), ("mlp", "mlp-smote"), ("knn-under", "knn-under-smote")]
rows = []
for a, b in pairs:
    if a not in folds or b not in folds:
        continue
    d = tfolds[b]["valid_pr_auc"] - tfolds[a]["valid_pr_auc"]  # mesmos splits: diferença pareada
    rows.append(
        {
            "alternativa": a,
            "com SMOTE": b,
            "estrat. alt.": folds[a]["valid_pr_auc"].mean(),
            "estrat. SMOTE": folds[b]["valid_pr_auc"].mean(),
            "temporal alt.": tfolds[a]["valid_pr_auc"].mean(),
            "temporal SMOTE": tfolds[b]["valid_pr_auc"].mean(),
            "splits temporais em que SMOTE ganha": f"{int((d > 0).sum())} de {len(d)}",
            "treino alt. (s)": m("fit_seconds")[a],
            "treino SMOTE (s)": m("fit_seconds")[b],
        }
    )
pd.DataFrame(rows).round(4)
"""
)
md("imb")

md("fig1_md", "## 3. PR-AUC no 5-fold estratificado: média, desvio e holdout")
code(
    """
names = list(runs.index)
mean, std, hold = m("cv_valid_pr_auc_mean"), m("cv_valid_pr_auc_std"), m("pr_auc")
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ypos = np.arange(len(names))[::-1]
for y_, n in zip(ypos, names):
    smote = "smote" in n
    ax.barh(y_, mean[n], color=COLOR[family(n)], height=0.62, alpha=0.55 if smote else 1.0,
            hatch="///" if smote else None, edgecolor="#fcfcfb", linewidth=0)
    ax.errorbar(mean[n], y_, xerr=std[n], color="#52514e", capsize=3, lw=1.2)
    ax.plot(hold[n], y_, marker="D", ms=7, color="#0b0b0b", mec="#fcfcfb", mew=1.5, ls="none")
    ax.annotate(f"{mean[n]:.3f} ± {std[n]:.3f}", (0.01, y_), va="center", fontsize=8, color="#0b0b0b")
ax.plot([], [], marker="D", ms=7, color="#0b0b0b", ls="none", label="Holdout temporal")
ax.errorbar([], [], xerr=[], color="#52514e", capsize=3, label="Média ± desvio (5-fold estratificado)")
ax.set_yticks(ypos, names)
ax.set_xlim(0.0, 1.0)
ax.set_xlabel("PR-AUC (classe fraude)")
ax.set_title("Comparação de modelos — PR-AUC")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.savefig(FIGS / "cmp_01_pr_auc.png")
"""
)
md("fig1")

md(
    "fig7_md",
    """## 4. Estratificado × temporal × holdout

O stratified k-fold embaralha as linhas: transações de momentos diferentes caem no treino e na validação. O 5-fold temporal (`TimeSeriesSplit`) treina sempre no passado e valida no bloco seguinte, como o holdout. Se os modelos flexíveis estiverem aproveitando fraudes próximas no tempo (rajadas), o estratificado fica acima dos outros dois.""",
)
code(
    """
tmean, tstd = m("cv_temporal_valid_pr_auc_mean"), m("cv_temporal_valid_pr_auc_std")
lo, hi = m("pr_auc_ci_low"), m("pr_auc_ci_high")
fig, ax = plt.subplots(figsize=(7.5, 5))
for y_, n in zip(ypos, names):
    ax.errorbar(mean[n], y_ + 0.22, xerr=std[n], fmt="o", color="#2a78d6", ms=6, capsize=3, lw=1.2)
    ax.errorbar(tmean[n], y_, xerr=tstd[n], fmt="s", color="#eb6834", ms=6, capsize=3, lw=1.2)
    ax.errorbar(hold[n], y_ - 0.22, xerr=[[hold[n] - lo[n]], [hi[n] - hold[n]]], fmt="D", color="#0b0b0b", ms=6, capsize=3, lw=1.2)
ax.errorbar([], [], xerr=[], fmt="o", color="#2a78d6", label="5-fold estratificado (média ± desvio)")
ax.errorbar([], [], xerr=[], fmt="s", color="#eb6834", label="5-fold temporal (média ± desvio)")
ax.errorbar([], [], xerr=[], fmt="D", color="#0b0b0b", label="Holdout temporal (IC 95%, bootstrap)")
ax.set_yticks(ypos, names)
ax.set_xlim(0.0, 1.0)
ax.set_xlabel("PR-AUC (classe fraude)")
ax.set_title("Três estimativas da PR-AUC por modelo")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.savefig(FIGS / "cmp_07_estratificado_temporal_holdout.png")

three = pd.DataFrame({"estratificado": mean, "temporal": tmean, "holdout": hold})
three["estrat. − temporal"] = three["estratificado"] - three["temporal"]
three["temporal − holdout"] = three["temporal"] - three["holdout"]
three["treino (estrat.)"] = m("cv_train_pr_auc_mean")
three.round(4)
"""
)
md("fig7")

md(
    "boot_md",
    """### Quais diferenças no holdout são reais?

O holdout tem só 52 fraudes. Reamostramos as 45.401 linhas do holdout 1.000 vezes (`random_state=42`) e, **em cada reamostragem, calculamos a PR-AUC de todos os modelos nas mesmas linhas**. A diferença pareada entre dois modelos tem IC bem mais estreito que a comparação de dois ICs separados, porque o "azar" da amostra afeta os dois igualmente.""",
)
code(
    """
from sklearn.metrics import average_precision_score

y_h = scores[names[0]]["y"].to_numpy()
rng = np.random.RandomState(42)
idx = [i for i in (rng.randint(0, len(y_h), len(y_h)) for _ in range(1000)) if y_h[i].any()]
boot = {n: np.array([average_precision_score(y_h[i], scores[n]["score"].to_numpy()[i]) for i in idx]) for n in names}
ref = max(names, key=lambda n: hold[n])
rows = []
for n in names:
    if n == ref:
        continue
    d = boot[ref] - boot[n]
    rows.append({"comparação": f"{ref} − {n}", "diferença": hold[ref] - hold[n],
                 "IC 95% baixo": np.quantile(d, 0.025), "IC 95% alto": np.quantile(d, 0.975),
                 f"P({ref} melhor)": np.mean(d > 0), "significativa": np.quantile(d, 0.025) > 0})
pd.DataFrame(rows).round(4)
"""
)
md("boot")

md("fig2_md", "## 5. Estabilidade fold a fold (estratificado e temporal)")
code(
    """
best = {}  # melhor variante de cada família, escolhida pelo 5-fold TEMPORAL (não pelo holdout)
for n in names:
    fam = family(n)
    if fam not in best or tmean[n] > tmean[best[fam]]:
        best[fam] = n
print("Variante escolhida por família (pelo 5-fold temporal):", best)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, cv_folds, title in [(axes[0], folds, "5-fold estratificado"), (axes[1], tfolds, "5-fold temporal (TimeSeriesSplit)")]:
    for fam, n in best.items():
        f = cv_folds[n]
        ax.plot(f.fold, f.valid_pr_auc, "o-", color=COLOR[fam], lw=2, ms=6, label=FAMILY[fam])
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Fold / split")
    ax.set_title(title)
axes[0].set_ylabel("PR-AUC de validação")
axes[0].legend(frameon=False, fontsize=8, ncol=6, loc="upper left", bbox_to_anchor=(0.0, -0.18))
fig.suptitle("PR-AUC por fold (melhor variante de cada família)")
fig.savefig(FIGS / "cmp_02_folds.png")

ranks = pd.DataFrame({n: tfolds[n]["valid_pr_auc"].values for n in names}).rank(axis=1, ascending=False)
ranks.index = [f"split temporal {i}" for i in range(1, 6)]
ranks.astype(int)
"""
)
md("fig2")

md("fig3_md", "## 6. Curvas precisão × recall no holdout temporal")
code(
    """
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for fam, n in best.items():
    p, r, _ = precision_recall_curve(scores[n]["y"], scores[n]["score"])
    ax.plot(r, p, color=COLOR[fam], lw=2, label=f"{FAMILY[fam]} ({n}) — PR-AUC {hold[n]:.3f}")
prev = scores[names[0]]["y"].mean()
ax.axhline(prev, color="#52514e", ls="--", lw=1, label=f"Aleatório = prevalência ({prev:.4f})")
ax.set_xlabel("Recall (fraude)")
ax.set_ylabel("Precisão (fraude)")
ax.set_xlim(0, 1.01)
ax.set_ylim(0, 1.02)
ax.set_title("Curvas PR no holdout temporal (52 fraudes em 45.401)")
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
fig.savefig(FIGS / "cmp_03_curvas_pr.png")
"""
)
md("fig3")

md("fig4_md", "## 7. Matrizes de confusão no holdout (limiar 0,5)")
code(
    """
fig, axes = plt.subplots(2, 3, figsize=(9.5, 6))
for ax, (fam, n) in zip(axes.ravel(), best.items()):
    s = scores[n]
    cm = confusion_matrix(s["y"], s["score"] >= 0.5)
    ax.imshow(cm, cmap="Blues", norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=cm.max()))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center",
                color="#fcfcfb" if v > 3000 else "#0b0b0b", fontsize=10)
    ax.set_xticks([0, 1], ["Legítima", "Fraude"])
    ax.set_yticks([0, 1], ["Legítima", "Fraude"])
    ax.set_title(f"{FAMILY[fam]}\\nrecall {cm[1, 1] / cm[1].sum():.2f} · precisão {cm[1, 1] / max(cm[:, 1].sum(), 1):.2f}", fontsize=9)
    ax.grid(False)
for ax in axes[:, 0]:
    ax.set_ylabel("Real")
for ax in axes[1]:
    ax.set_xlabel("Previsto")
fig.suptitle("Matrizes de confusão no holdout temporal (limiar 0,5)")
fig.tight_layout()
fig.savefig(FIGS / "cmp_04_confusao.png")
"""
)
md("fig4")

md("fig5_md", "## 8. Treino × validação (overfitting e underfitting)")
code(
    """
diag = pd.DataFrame(
    {
        "CV treino": m("cv_train_pr_auc_mean"),
        "CV validação": m("cv_valid_pr_auc_mean"),
        "CV temporal validação": m("cv_temporal_valid_pr_auc_mean"),
        "holdout: treino (dev)": m("train_pr_auc"),
        "holdout: validação": m("pr_auc"),
    }
)
diag["gap CV (treino − val.)"] = diag["CV treino"] - diag["CV validação"]
diag["gap holdout"] = diag["holdout: treino (dev)"] - diag["holdout: validação"]

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for y_, n in zip(ypos, names):
    tr, va, tv = diag.loc[n, "CV treino"], diag.loc[n, "CV validação"], diag.loc[n, "CV temporal validação"]
    ax.plot([min(va, tv), tr], [y_, y_], color="#c3c2b7", lw=2, zorder=1)
    ax.scatter(tr, y_, s=60, color="#2a78d6", zorder=2, edgecolors="#fcfcfb", linewidths=1.5)
    ax.scatter(va, y_, s=60, color="#eb6834", zorder=2, edgecolors="#fcfcfb", linewidths=1.5)
    ax.scatter(tv, y_, s=60, marker="s", color="#0b0b0b", zorder=2, edgecolors="#fcfcfb", linewidths=1.5)
ax.scatter([], [], s=60, color="#2a78d6", label="Treino (média dos folds estratificados)")
ax.scatter([], [], s=60, color="#eb6834", label="Validação, 5-fold estratificado")
ax.scatter([], [], s=60, marker="s", color="#0b0b0b", label="Validação, 5-fold temporal")
ax.set_yticks(ypos, names)
ax.set_xlabel("PR-AUC (desenvolvimento)")
ax.set_title("Treino × validação: distância = sinal de overfitting")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.savefig(FIGS / "cmp_05_treino_validacao.png")
diag.round(4)
"""
)
md("fig5")

md("fig6_md", "## 9. Custo × desempenho")
code(
    """
OFF = {"knn-under": (6, -12), "mlp": (-7, 5), "decision-tree": (-7, -12), "mlp-smote": (7, -9)}  # rótulos sem colisão
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for n in names:
    ax.scatter(m("fit_seconds")[n], hold[n], s=70, color=COLOR[family(n)],
               alpha=0.55 if "smote" in n else 1.0, edgecolors="#fcfcfb", linewidths=1.5)
    ax.annotate(n, (m("fit_seconds")[n], hold[n]), xytext=OFF.get(n, (6, 2)), textcoords="offset points", fontsize=8, ha="right" if OFF.get(n, (6, 2))[0] < 0 else "left")
ax.set_xscale("log")
ax.set_xlabel("Tempo de treino no desenvolvimento (s, escala log)")
ax.set_ylabel("PR-AUC no holdout temporal")
ax.set_title("Custo de treino × PR-AUC")
fig.savefig(FIGS / "cmp_06_custo.png")
"""
)
md("fig6")
md("conclusao_md", "## 10. Análise crítica e finalistas da A7")
md("conclusao")

nb = {
    "cells": [
        {
            "cell_type": t,
            "id": str(i),
            "metadata": {},
            "source": s.splitlines(keepends=True),
        }
        | ({"execution_count": None, "outputs": []} if t == "code" else {})
        for i, (t, s) in enumerate(cells)
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebooks" / "03_comparacao.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("ok", out)
