#!/usr/bin/env python3
"""Render CLI artifact overlays from an existing artifact root."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.service.task_visuals import render_latest_artifact_visuals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render proposal/prompt/mask overlays from the latest artifacts in an artifact root."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Input image path used by the CLI task.",
    )
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="Artifact root that contains proposal_result/prompt_package/... JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write rendered overlay PNG files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created_files = render_latest_artifact_visuals(
        image_path=args.image,
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
    )
    if not created_files:
        raise SystemExit("No renderable artifacts were found under the given artifact root.")

    print("\n".join(str(path) for path in created_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
