"""Dashboard Streamlit: transações, previsões, métricas ao vivo e saúde dos serviços.

Visual: o tema fica em .streamlit/config.toml e os tokens/regras globais em app/style.css.
Todo componente novo segue os dois (veja as regras no topo do style.css).
"""

from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine

from src.common import config

st.set_page_config(page_title="FraudaZero", layout="wide")
st.html(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>")
st.title("FraudaZero")
st.caption("Detecção de fraude em transações em tempo real")

engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)


@st.fragment(run_every=5)
def live():
    try:
        df = pd.read_sql(
            "SELECT * FROM transactions ORDER BY received_at DESC LIMIT 5000", engine
        )
    except Exception:
        st.info("Aguardando as primeiras transações...")
        return
    if df.empty:
        st.info("Aguardando as primeiras transações...")
        return

    scored = df.dropna(subset=["score", "label"]).copy()
    scored["is_fraud"] = scored["is_fraud"].astype(bool)
    tp = int((scored["is_fraud"] & (scored["label"] == 1)).sum())
    fp = int((scored["is_fraud"] & (scored["label"] == 0)).sum())
    fn = int((~scored["is_fraud"] & (scored["label"] == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    try:
        api_ok = (
            requests.get(f"{config.API_URL}/health", timeout=2)
            .json()
            .get("model_loaded", False)
        )
    except Exception:
        api_ok = False

    kpis = [
        ("Transações", len(df)),
        ("Alertas de fraude", int(scored["is_fraud"].sum())),
        ("Precisão (ao vivo)", f"{precision:.2%}"),
        ("Recall (ao vivo)", f"{recall:.2%}"),
        (
            "Latência média (ms)",
            f"{df['latency_ms'].mean():.0f}" if df["latency_ms"].notna().any() else "-",
        ),
        ("API / modelo", "ok" if api_ok else "indisponível"),
    ]
    for i, (col, (label, value)) in enumerate(zip(st.columns(len(kpis)), kpis)):
        with col.container(border=True, key=f"card-kpi-{i}"):
            st.metric(label, value)

    st.subheader("Transações por minuto")
    per_min = df.set_index("received_at").resample("1min").size().rename("transações")
    st.line_chart(per_min)

    st.subheader("Últimos alertas")
    alerts = scored[scored["is_fraud"]][
        ["tx_id", "received_at", "score", "label"]
    ].head(20)
    st.dataframe(alerts, use_container_width=True)


live()
