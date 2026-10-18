# rl-lunar-lander

Train a PPO agent to land the Gymnasium `LunarLander-v3` rocket, then ship the trained policy as an ONNX file behind a small FastAPI service.

I built this to practice the whole path from training to serving rather than stopping at a notebook: a reward-shaping wrapper, vectorized PPO training, evaluation, ONNX export, a test that checks the exported model agrees with the original, and an HTTP endpoint.

## What's in here

```
src/
  wrappers.py    reward shaping (penalize thruster use and tilt)
  train.py       PPO training, best-checkpoint saving, TensorBoard logs
  evaluate.py    deterministic evaluation, optional MP4 recording
  export.py      PyTorch policy -> ONNX, validated with onnx.checker
  server.py      FastAPI app serving the ONNX model with onnxruntime
  benchmark.py   reward / fuel use / latency numbers for the README
  dev.py         builds a throwaway model so you can try the API without training
tests/
  test_onnx_match.py   PyTorch vs ONNX output parity, plus API checks
examples/        sample states to send to the API
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`gymnasium[box2d]` builds Box2D from source, so you need `swig` installed first (`brew install swig` on macOS).

Intel Mac note: the last PyTorch wheels for Intel macOS are 2.2.2, which needs Python 3.11 or older (not 3.13), `numpy<2`, and `onnx<=1.17`. On Apple Silicon or Linux, current versions are fine.

## Usage

Train (about 25 minutes for 1M steps on a laptop CPU with 4 envs):

```bash
python -m src.train --timesteps 1000000 --n-envs 4
# baseline without shaping, for comparison:
python -m src.train --timesteps 1000000 --n-envs 4 --no-shaping \
    --model-dir models_baseline --log-dir logs_baseline
```

Use `--continuous` for the continuous-action version of the environment.

Watch training:

```bash
tensorboard --logdir .      # http://localhost:6006
```

Evaluate, export, benchmark:

```bash
python -m src.evaluate --model models/best_model.zip --episodes 20 --video
python -m src.export   --model models/best_model.zip
python -m src.benchmark --model models/best_model.zip
```

Serve:

```bash
uvicorn src.server:app --port 8000
curl -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"state": [0.0, 1.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0]}'
```

The state is `[x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]`. For discrete models the response has an integer `action` (0 noop, 1 left engine, 2 main engine, 3 right engine). `GET /health` is a liveness check. Set `MODEL_PATH` to serve a different `.onnx` file.

No trained model yet? `python -m src.dev` builds a barely-trained one in `models_dev/` so you can exercise the API. It behaves close to randomly, so don't read anything into its actions.

## How it works

**Reward shaping.** `FuelEfficientLandingWrapper` subtracts a small cost for firing the main and side engines, and a cost for tilting more than a tolerance. The tolerance grows with distance from the pad centre, since the lander has to lean to move sideways. It works for both action spaces. A `RewardWrapper` only sees the reward, so the wrapper overrides `step()` to remember the action and observation first.

**Fair comparison.** `Monitor` sits inside the shaping wrapper and the evaluation env is unshaped, so every reward you see in logs and evals is the real environment reward. Shaped and unshaped runs can be compared directly.

**ONNX export.** `OnnxablePolicy` wraps the actor path of the SB3 policy and bakes in the deterministic action (argmax for discrete, clipped mean for continuous). The server then needs only `onnxruntime`, not PyTorch. A small `.json` file next to the model records which action space it uses.

**Tests.** `pytest` trains a tiny model, exports it, and asserts the ONNX output matches `PPO.predict(deterministic=True)` for both action spaces, then hits the API through FastAPI's test client.

## Results

1M PPO steps, 4 envs, seed 0, best checkpoint by raw eval reward. Evaluated on 100 deterministic episodes (seeds 1000-1099) of the unshaped discrete environment. "Success" means a return of 200 or more. Raw numbers are in [results/results.json](results/results.json).

| | Mean return | Success | Main-engine firings / episode | Side-engine firings / episode |
| --- | --- | --- | --- | --- |
| Baseline (no shaping) | 256.3 +/- 33.7 | 97% | 127.8 | 113.3 |
| Shaped | 244.6 +/- 41.7 | 90% | 125.7 | 137.8 |

**Shaping did not help here.** The shaped agent scored slightly lower, succeeded less often, and used about the same main-engine fuel (2% fewer firings, within noise) while firing the side engines more. With one seed per run I can't say the gap is real, but there's no evidence for a benefit. The penalty weights were hand-picked and likely need tuning, or the default reward already charges enough for fuel.

**ONNX is faster for single-state inference:** median 0.025 ms with onnxruntime vs 0.35 ms for SB3's `predict` (about 14x), measured on CPU. Part of that gap is SB3's per-call overhead (input checks, tensor conversion), not the network itself.

## Limitations

- One seed per configuration, so the shaped vs. baseline gap may be noise. Run several seeds before drawing conclusions.
- Shaping weights were picked by hand, not tuned.
- The API handles one state per request; there is no batching, auth or rate limiting.

## License

MIT
