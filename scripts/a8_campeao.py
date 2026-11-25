"""A8: tabela de candidatos (A6 + A7), limiar do campeão na curva PR do holdout e figuras.

Uso: python scripts/a8_campeao.py                 -> só a tabela de candidatos (holdout, IC, temporal)
     python scripts/a8_campeao.py <run> <modelo>  -> limiar da run <run> e grava src/best_params.json
        ex.: python scripts/a8_campeao.py random-forest-tuned random-forest

Critério do limiar: o que maximiza o F1 da classe fraude no holdout (validation.choose_threshold).
Não toca no stream.csv (isso é `python -m src.train stream_test`, uma única vez, depois do `final`).
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
from src import validation as val

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
FIGS = ROOT / "reports" / "figs"
BEST = ROOT / "src" / "best_params.json"

runs = pd.concat(
    mlflow.search_runs(experiment_names=[e], order_by=["attributes.start_time DESC"])
    for e in ["comparacao", "otimizacao"]
)
runs = runs[runs["metrics.pr_auc"].notna()].drop_duplicates("tags.mlflow.runName")
runs = runs.set_index("tags.mlflow.runName")
table = pd.DataFrame(
    {
        "holdout": runs["metrics.pr_auc"],
        "IC baixo": runs["metrics.pr_auc_ci_low"],
        "IC alto": runs["metrics.pr_auc_ci_high"],
        "temporal": runs["metrics.cv_temporal_valid_pr_auc_mean"],
        "temporal dp": runs["metrics.cv_temporal_valid_pr_auc_std"],
        "estratificado": runs["metrics.cv_valid_pr_auc_mean"],
        "treino (s)": runs["metrics.fit_seconds"],
    }
).sort_values("holdout", ascending=False)
print(table.round(4).to_string())
if len(sys.argv) < 3:
    sys.exit(0)

run_name, model_name = sys.argv[1], sys.argv[2]
run_id = runs.loc[run_name, "run_id"]
s = pd.read_csv(
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="holdout_scores.csv"
    )
)
y, scores = s["y"].to_numpy(), s["score"].to_numpy()
t_exact, _, _, f1_exact = val.choose_threshold(y, scores)
# Valor publicado com 6 casas (FRAUD_THRESHOLD, tag do modelo): as métricas e figuras usam ESTE valor.
t = round(t_exact, 6)
m_t = val.metrics(y, scores, t)
p_t, r_t, f1_t = m_t["precision"], m_t["recall"], m_t["f1"]
print(f"limiar exato (máx. F1) = {t_exact:.8f}, F1 {f1_exact:.4f}")
print(
    f"\nlimiar (máx. F1) = {t:.6f}: precisão {p_t:.4f}, recall {r_t:.4f}, F1 {f1_t:.4f}"
)
print("limiar 0,5:", {k: round(v, 4) for k, v in val.metrics(y, scores, 0.5).items()})
print(
    f"limiar {t:.4f}:", {k: round(v, 4) for k, v in val.metrics(y, scores, t).items()}
)

# Alternativas (para o relatório): recall mínimo com a melhor precisão possível.
p, r, th = precision_recall_curve(y, scores)
for target in [0.8, 0.75, 0.7]:
    ok = np.where(r[:-1] >= target)[0]
    i = ok[np.argmax(p[ok])]
    print(
        f"recall >= {target}: limiar {th[i]:.4f}, precisão {p[i]:.4f}, recall {r[i]:.4f}"
    )

# Fig. 1: precisão, recall e F1 em função do limiar
f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(th, p[:-1], color="#2a78d6", lw=2, label="Precisão")
ax.plot(th, r[:-1], color="#eb6834", lw=2, label="Recall")
ax.plot(th, f1, color="#1baf7a", lw=2, label="F1")
ax.axvline(t, color="#0b0b0b", ls="--", lw=1)
ax.annotate(
    f"limiar publicado = {t:.6f}\n(F1 = {f1_t:.3f} no holdout)",
    (t, 0.45),
    xytext=(-8, 0),
    textcoords="offset points",
    ha="right",
    fontsize=9,
)
ax.axvline(0.5, color="#52514e", ls=":", lw=1)
ax.annotate(
    "0,5",
    (0.5, 1.0),
    xytext=(3, -12),
    textcoords="offset points",
    fontsize=8,
    color="#52514e",
)
ax.set_xlabel("Limiar de decisão (score ≥ limiar → fraude)")
ax.set_ylabel("Métrica da classe fraude")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_title(f"Escolha do limiar no holdout temporal ({run_name})")
ax.legend(frameon=False, loc="lower left")
fig.savefig(FIGS / "a8_01_limiar.png")

# Fig. 2: matriz de confusão antes (0,5) e depois (limiar escolhido)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, thr, title in [
    (axes[0], 0.5, "Antes: limiar 0,5"),
    (axes[1], t, f"Depois: limiar {t:.3f}"),
]:
    cm = np.array(
        [[((y == a) & ((scores >= thr) == b)).sum() for b in (0, 1)] for a in (0, 1)]
    )
    ax.imshow(
        cm, cmap="Blues", norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=cm.max())
    )
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{v:,}".replace(",", "."),
            ha="center",
            va="center",
            color="#fcfcfb" if v > 3000 else "#0b0b0b",
            fontsize=11,
        )
    ax.set_xticks([0, 1], ["Legítima", "Fraude"])
    ax.set_yticks([0, 1], ["Legítima", "Fraude"])
    ax.set_xlabel("Previsto")
    ax.set_title(
        f"{title}\nrecall {cm[1, 1] / cm[1].sum():.2f} · precisão {cm[1, 1] / max(cm[:, 1].sum(), 1):.2f}",
        fontsize=10,
    )
    ax.grid(False)
axes[0].set_ylabel("Real")
fig.suptitle(
    f"Holdout temporal ({run_name}): matriz de confusão antes e depois do limiar"
)
fig.tight_layout()
fig.savefig(FIGS / "a8_02_confusao_antes_depois.png")

cfg = json.loads(BEST.read_text()) if BEST.exists() else {"models": {}}
cfg["champion"] = model_name
cfg["threshold"] = t
cfg["threshold_criterion"] = "max F1 no holdout temporal"
cfg["champion_source_run"] = run_id
BEST.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"\nsrc/best_params.json: champion={model_name}, threshold={cfg['threshold']}")
