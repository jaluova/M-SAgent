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
    parser.add_argument(
        "--checkpoint-file",
        default=os.environ.get("GRIDGROUND_CHECKPOINT_FILE", "best_model.pth"),
        help="Checkpoint file to fetch after the snapshot metadata is downloaded",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Only download the snapshot metadata and skip the large checkpoint file",
    )
    args = parser.parse_args()

    try:
        from modelscope.hub.snapshot_download import snapshot_download
        from modelscope.hub.file_download import model_file_download
    except Exception as exc:
        print(f"modelscope is required to download GridGround model: {exc}", file=sys.stderr)
        return 1

    path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    if not args.skip_checkpoint:
        checkpoint_path = model_file_download(
            model_id=args.model_id,
            file_path=args.checkpoint_file,
            cache_dir=args.cache_dir,
        )
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            print(
                f"failed to download checkpoint '{args.checkpoint_file}' for {args.model_id}",
                file=sys.stderr,
            )
            return 1
        print(f"checkpoint={checkpoint_path}")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
