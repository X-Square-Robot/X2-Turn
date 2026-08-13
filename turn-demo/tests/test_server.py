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
        logo = client.get("/assets/x-square-logo.png")

    assert response.status_code == 200
    assert "X2 Turn Demo" in response.text
    assert "原始 Turn 模型输出" in response.text
    assert "ACCEPT" not in response.text
    assert "barge_in_frames" not in response.text
    assert "/assets/x-square-logo.png" in response.text
    assert logo.status_code == 200
    assert logo.headers["content-type"] == "image/png"


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
