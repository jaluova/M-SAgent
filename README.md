# M-SAgent

`M-SAgent` is an agentic referring-image segmentation project built around a simple loop:

1. `Qwen2.5-VL` decides what to do next.
2. A localization tool proposes points or concepts.
3. `SAM3` produces masks.
4. The MLLM reviews the result and either accepts it or iterates again.

The current default localization backend is the embedded `GridGround` runtime, which reuses the same loaded `Qwen2.5-VL` backbone instead of starting a second Qwen process.

## Repository Layout

```text
M-SAgent/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── pipeline.py
├── mllm_processor.py
├── sam_processor.py
├── gridground_runtime/        # Embedded GridGround adapter runtime
├── tools/                     # object_locator / concept_generator / image_enhancer
├── utils/                     # image helpers, localization dataclass, shared backbone
├── scripts/                   # startup, deployment, and model download helpers
├── prompts/                   # MLLM prompt files
├── docs/                      # environment notes and HTML architecture demo
├── tests/                     # focused unit tests
└── sam3/                      # vendored SAM3 code
```

## Core Flow

- `main.py`: CLI entrypoint.
- `pipeline.py`: orchestrates the MLLM -> tool -> SAM3 -> review loop.
- `mllm_processor.py`: wraps `Qwen2.5-VL` for both tool selection and segmentation review.
- `tools/object_locator.py`: localization entry, now supporting embedded GridGround and HTTP fallback.
- `sam_processor.py`: wraps SAM3 image setup, point prompts, text prompts, and visualization.

## Setup

### 1. Create the environment

Follow [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the tested setup.

```bash
conda create -n m_sagent python=3.10 -y
conda activate m_sagent

# Pick the torch index that matches your CUDA version.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Prepare model assets

You need local access to:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- a `facebook/sam3` checkpoint
- `alpharho/GridGround-TextGuided` from ModelScope

You can download the GridGround snapshot with:

```bash
python scripts/download_gridground_model.py
```

### 3. Set runtime paths

Typical environment variables:

```bash
export M_SAGENT_QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export M_SAGENT_SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
export GRIDGROUND_MODEL_DIR=/path/to/GridGround-TextGuided
```

If you want to use the checked-in "single GPU stable" defaults, the wrapper script also understands:

```bash
export M_SAGENT_PYTHON=/path/to/python
export M_SAGENT_SAM3_MODEL_PATH=/path/to/M-SAgent/sam3
```

## Start the Project

There are three common ways to run `M-SAgent`, depending on whether you want the CLI, the backend API, or the web UI.

### Option 1: Run a single segmentation job from the CLI

This is the simplest way to verify the pipeline is working.

```bash
python main.py \
  --image sam3/assets/images/truck.jpg \
  --text "truck"
```

Recommended on the target server: use the wrapper script instead of calling `main.py` directly. It sets the expected model paths, GPU checks, and GridGround defaults.

```bash
./scripts/run_single_gpu_stable.sh \
  --image sam3/assets/images/truck.jpg \
  --text "truck"
```

### Option 2: Start the backend API

The FastAPI app is defined in [server/app.py](server/app.py). Start it with:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

On a single A100 deployment, prefer the wrapper script instead:

```bash
./scripts/start_web_single_gpu_stable.sh
```

This script forces the web service onto the embedded GridGround path, disables the external TrainAdapter fallback, and refuses to start if it detects another GPU-backed TrainAdapter service on `127.0.0.1:8765`.

Useful checks:

```bash
curl http://127.0.0.1:8000/api/health
```

Default server settings come from [server/config.py](server/config.py):

- host: `0.0.0.0`
- port: `8000`
- frontend dev origin allowlist: `http://127.0.0.1:5173,http://localhost:5173`

If you want to override them:

```bash
export M_SAGENT_SERVER_HOST=0.0.0.0
export M_SAGENT_SERVER_PORT=8000
export M_SAGENT_SERVER_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
```

### Option 3: Start the web frontend

The frontend lives in [frontend](frontend). For local development:

```bash
cd frontend
npm install
npm run dev
```

By default the Vite dev server runs on `http://127.0.0.1:5173` or `http://localhost:5173`.

### Full local dev workflow

If you want the full web app locally, run the backend and frontend in two terminals:

Terminal 1:

```bash
./scripts/start_web_single_gpu_stable.sh
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Then open:

- frontend: `http://127.0.0.1:5173`
- backend health: `http://127.0.0.1:8000/api/health`

### Serve the built frontend from FastAPI

If you want a single backend process to serve both API and static frontend files:

```bash
cd frontend
npm install
npm run build

cd ..
./scripts/start_web_single_gpu_stable.sh
```

After the build, the backend serves `frontend/dist` at `/`.

## Localization Backends

### Embedded GridGround

- Default mode.
- Reuses the already-loaded `Qwen2.5-VL`.
- Avoids the old “two Qwen models on one GPU” problem.

### HTTP TrainAdapter Fallback

- Still available for compatibility.
- Useful if you want to keep an external localization service.
- Do not run it on GPU next to the main web backend in the `single_gpu_stable` profile, or you will load a second `Qwen2.5-VL` and likely OOM.

## Avoid Double Qwen on Web Deployments

If the CLI command below works:

```bash
python main.py \
  --image sam3/assets/images/truck.jpg \
  --text "truck"
```

but the web service OOMs, the most common cause is that an extra TrainAdapter service was started with `--device cuda`. In that situation:

1. The web backend loads one `Qwen2.5-VL`.
2. TrainAdapter loads another `Qwen2.5-VL`.
3. Both compete for the same GPU memory.

Use embedded GridGround for the web backend and avoid starting a separate GPU TrainAdapter process. The safe startup path is:

```bash
./scripts/start_web_single_gpu_stable.sh
```

## Tests

Run the lightweight unit tests with:

```bash
python -m unittest discover -s tests -v
```

Current tests cover:

- localization result parsing
- HTTP retry and healthcheck logic
- embedded object locator plumbing
- segmentation evaluation return contract

## Useful Docs

- Environment notes: [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)
- Architecture demo: [docs/demo.html](docs/demo.html)

## Notes

- Runtime outputs such as `tool_calls_log/`, `outputs*/`, and `remote_artifacts/` are intentionally ignored by Git.
- `sam3/` is vendored third-party code; most project-specific work happens in the top-level Python files plus `tools/`, `utils/`, and `gridground_runtime/`.
