# Environment

Tested working profile:

- Python: `3.10`
- GPU: `1x A100 40GB`
- CUDA runtime used in the verified environment: `12.4`

## Recommended Setup

```bash
conda create -n m_sagent python=3.10 -y
conda activate m_sagent

# Pick the torch index that matches your CUDA version.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Required Assets

You still need these model assets locally:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `facebook/sam3` checkpoint
- `alpharho/GridGround-TextGuided` from ModelScope

## Useful Environment Variables

```bash
export M_SAGENT_QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export M_SAGENT_SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
export GRIDGROUND_MODEL_DIR=/path/to/GridGround-TextGuided
```

## Notes

- `requirements.txt` installs the in-repo `sam3` package via `-e ./sam3`.
- If you only run CPU inference, a CPU torch build also works, but performance will be much lower.
- `scripts/download_gridground_model.py` can be used to pull the GridGround ModelScope snapshot and checkpoint.
