"""FastAPI microservice serving the exported ONNX policy."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_MODEL = "./models/lunar_lander_policy.onnx"
ACTION_NAMES = {0: "noop", 1: "fire_left_engine", 2: "fire_main_engine", 3: "fire_right_engine"}


class PredictRequest(BaseModel):
    """Lander state: [x, y, vx, vy, angle, angular_velocity, leg1, leg2]."""

    state: list[float] = Field(..., min_length=8, max_length=8)


class PredictResponse(BaseModel):
    """Predicted action. ``action`` is an int (discrete) or a list (continuous)."""

    action: int | list[float]
    action_name: str | None = None
    latency_ms: float


def create_app(model_path: str | None = None) -> FastAPI:
    """Build the FastAPI app; the ONNX session is loaded once at startup."""
    path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not path.exists():
            raise RuntimeError(f"ONNX model not found at {path}. Run src.export first.")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        app.state.session = ort.InferenceSession(
            str(path), opts, providers=["CPUExecutionProvider"]
        )
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        app.state.continuous = bool(meta.get("continuous", False))
        yield

    app = FastAPI(title="LunarLander ONNX Policy", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        """Return the deterministic action for a single state."""
        if not np.all(np.isfinite(req.state)):
            raise HTTPException(status_code=422, detail="state must be finite")
        obs = np.asarray(req.state, dtype=np.float32)[None, :]
        start = time.perf_counter()
        (out,) = app.state.session.run(None, {"observation": obs})
        latency = (time.perf_counter() - start) * 1000.0
        if app.state.continuous:
            return PredictResponse(action=[float(v) for v in out[0]], latency_ms=latency)
        idx = int(out[0])
        return PredictResponse(action=idx, action_name=ACTION_NAMES.get(idx), latency_ms=latency)

    return app


app = create_app()
