#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Download GridGround model snapshot from ModelScope")
    parser.add_argument("--model-id", default=os.environ.get("GRIDGROUND_MODEL_ID", "alpharho/GridGround-TextGuided"))
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("GRIDGROUND_CACHE_DIR", "/root/autodl-tmp/modelscope_cache_gridground"),
    )
    args = parser.parse_args()

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as exc:
        print(f"modelscope is required to download GridGround model: {exc}", file=sys.stderr)
        return 1

    path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
