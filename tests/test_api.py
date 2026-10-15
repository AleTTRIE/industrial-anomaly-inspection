import io

from fastapi.testclient import TestClient
from PIL import Image

from inspection import api

client = TestClient(api.app)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.onnx")
    monkeypatch.setattr(api, "_model", None)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_predict_without_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.onnx")
    monkeypatch.setattr(api, "_model", None)
    response = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert response.status_code == 503


class _FakeModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, image):
        return self.score, None


def test_predict_decision(monkeypatch):
    monkeypatch.setattr(api, "_model", _FakeModel(0.8))
    monkeypatch.setattr(api, "_threshold", 0.5)
    response = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["decision"] == "KO"


def test_predict_rejects_non_image(monkeypatch):
    monkeypatch.setattr(api, "_model", _FakeModel(0.1))
    monkeypatch.setattr(api, "_threshold", 0.5)
    response = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert response.status_code == 400
