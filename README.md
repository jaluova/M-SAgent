# M-SAgent

This repository is a trimmed version of `M-SAgent` focused on running and deploying the referring-image segmentation pipeline.

## Current Structure

```text
M-SAgent/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── pipeline.py
├── mllm_processor.py
├── sam_processor.py
├── docs/ENVIRONMENT.md
├── example/                   # sample images for quick checks
├── frontend/                  # frontend source and built dist assets
├── gridground_runtime/        # embedded GridGround runtime
├── prompts/
├── sam3/                      # vendored SAM3 runtime code
├── scripts/                   # startup and deployment helpers
├── server/                    # FastAPI backend
├── tools/
└── utils/
```

## Required Environment

Install dependencies first:

```bash
conda create -n m_sagent python=3.10 -y
conda activate m_sagent

# Pick the torch index matching your CUDA version.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Environment notes are kept in `docs/ENVIRONMENT.md`.

## Required Model Paths

Set these before starting the service:

```bash
export M_SAGENT_PYTHON=/path/to/python
export M_SAGENT_QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export M_SAGENT_SAM3_MODEL_PATH=/path/to/M-SAgent/sam3
export M_SAGENT_SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
export GRIDGROUND_MODEL_DIR=/path/to/GridGround-TextGuided
```

If needed, download the GridGround snapshot with:

```bash
python scripts/download_gridground_model.py
```

## Start The Web Service

Preferred deployment entrypoint:

```bash
./scripts/start_web_single_gpu_stable.sh
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

The backend serves the built frontend from `frontend/dist`.

## Frontend Development

The frontend source is still kept under `frontend/src`. For local frontend development:

```bash
cd frontend
npm install
npm run dev
```

To rebuild the production assets:

```bash
cd frontend
npm install
npm run build
```

## CLI Smoke Test

You can run a quick single-image check with one of the sample images:

```bash
./scripts/run_single_gpu_stable.sh --image example/truck.jpg --text "truck"
```

## Notes

- The single-GPU web profile forces embedded GridGround to avoid loading a second Qwen model.
- `frontend/dist` is a build artifact used by the FastAPI server, while `frontend/src` remains available for development.
- The repository has been cleaned up, but example images and the frontend source are intentionally retained.
