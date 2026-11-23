"""Smoke test: requer os serviços de pé (`make up`) e a DAG já executada."""

import pytest
import requests

from src.common.schema import FEATURES

API = "http://localhost:8000"


def _api_up():
    try:
        return requests.get(f"{API}/health", timeout=2).ok
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not _api_up(), reason="API não está no ar (rode `make up`)"
)


def test_health():
    assert requests.get(f"{API}/health", timeout=5).json()["status"] == "ok"


def test_predict_returns_score():
    resp = requests.post(
        f"{API}/predict", json={"features": {f: 0.0 for f in FEATURES}}, timeout=10
    )
    assert resp.status_code in (200, 503)  # 503 = modelo ainda não registrado
    if resp.status_code == 200:
        assert 0.0 <= resp.json()["score"] <= 1.0
