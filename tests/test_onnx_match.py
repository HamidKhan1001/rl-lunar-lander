"""Parity test: SB3 PPO.predict vs ONNX Runtime output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient
from stable_baselines3 import PPO

from src.export import export_to_onnx
from src.server import create_app
from src.train import build_vec_env


@pytest.fixture(scope="module", params=[False, True], ids=["discrete", "continuous"])
def artifacts(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    """Train a tiny PPO model and export it; returns (model, onnx_path, continuous)."""
    continuous: bool = request.param
    tmp: Path = tmp_path_factory.mktemp("rl")
    env = build_vec_env(1, continuous, True, seed=0, subproc=False)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=0, device="cpu")
    model.learn(total_timesteps=128)
    env.close()
    zip_path = tmp / "model.zip"
    model.save(str(zip_path))
    onnx_path = export_to_onnx(str(zip_path), str(tmp / "policy.onnx"))
    return model, onnx_path, continuous


def _states(n: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    states = rng.uniform(-1.0, 1.0, size=(n, 8)).astype(np.float32)
    states[:, 6:] = rng.integers(0, 2, size=(n, 2))
    return states


def test_onnx_matches_pytorch(artifacts) -> None:
    model, onnx_path, _ = artifacts
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    states = _states()
    expected, _ = model.predict(states, deterministic=True)
    (actual,) = session.run(None, {"observation": states})
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32).reshape(expected.shape),
        np.asarray(expected, dtype=np.float32),
        rtol=1e-4,
        atol=1e-5,
    )


def test_server_predict(artifacts) -> None:
    model, onnx_path, continuous = artifacts
    state = _states(1)[0]
    expected, _ = model.predict(state, deterministic=True)
    with TestClient(create_app(str(onnx_path))) as client:
        resp = client.post("/predict", json={"state": state.tolist()})
        assert resp.status_code == 200
        action = resp.json()["action"]
        np.testing.assert_allclose(
            np.asarray(action, dtype=np.float32).reshape(-1),
            np.asarray(expected, dtype=np.float32).reshape(-1),
            rtol=1e-4,
            atol=1e-5,
        )
        assert client.post("/predict", json={"state": [0.0] * 7}).status_code == 422
