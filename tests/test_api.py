from fastapi.testclient import TestClient

import api
from src.data import generate_demo_dataset, validate_dataset
from src.modeling import save_model, train_and_compare


def test_api_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    client = TestClient(api.app)
    assert client.get("/health").json()["model_ready"] is False
    assert client.get("/model").status_code == 503


def test_api_predicts_with_trained_model(tmp_path, monkeypatch):
    df = validate_dataset(generate_demo_dataset(600, 3))
    path = save_model(train_and_compare(df, random_state=3)[0], tmp_path / "model.joblib")
    monkeypatch.setattr(api, "MODEL_PATH", path)
    client = TestClient(api.app)
    record = df.drop(columns="label").iloc[0].to_dict()
    result = client.post("/predict", json=record)
    assert result.status_code == 200
    assert result.json()["prediction"] in {"normal", "attack"}
    # Unknown fields are rejected, so typos in a client do not silently pass.
    assert client.post("/predict", json={**record, "extra": 1}).status_code == 422
