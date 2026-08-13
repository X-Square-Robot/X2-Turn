from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")

from demo_turn import server


def test_index_loads_without_private_scenarios():
    server.ARGS = SimpleNamespace(
        backend="hf",
        test_jsonl="",
        preds_jsonl="",
    )
    server.ENGINE = object()
    app = server.create_app()

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "X2 Turn Demo" in response.text


def test_health_does_not_eagerly_load_model():
    server.ARGS = SimpleNamespace(
        backend="hf",
        test_jsonl="",
        preds_jsonl="",
    )
    server.ENGINE = None
    app = server.create_app()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.json() == {
        "status": "ok",
        "backend": "hf",
        "model_loaded": False,
    }
