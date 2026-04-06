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

## Quick Start

### 1. Create the environment

Follow [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the tested setup.

Minimal flow:

```bash
conda create -n m_sagent python=3.10 -y
conda activate m_sagent

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Prepare model assets

You need local access to:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- a `SAM3` checkpoint
- `alpharho/GridGround-TextGuided` from ModelScope

You can download the GridGround snapshot with:

```bash
python scripts/download_gridground_model.py
```

### 3. Set paths

Typical environment variables:

```bash
export M_SAGENT_QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export M_SAGENT_SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
export GRIDGROUND_MODEL_DIR=/path/to/GridGround-TextGuided
```

### 4. Run

Direct CLI:

```bash
python main.py --image sam3/assets/images/truck.jpg --text "truck"
```

Single-GPU stable profile:

```bash
./scripts/run_single_gpu_stable.sh \
  --image sam3/assets/images/truck.jpg \
  --text "truck"
```

## Localization Backends

### Embedded GridGround

- Default mode.
- Reuses the already-loaded `Qwen2.5-VL`.
- Avoids the old “two Qwen models on one GPU” problem.

### HTTP TrainAdapter Fallback

- Still available for compatibility.
- Useful if you want to keep an external localization service.

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
