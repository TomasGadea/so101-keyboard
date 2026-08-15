#!/usr/bin/env python
"""Push a locally-recorded LeRobotDataset to the HuggingFace Hub.

Reads HF_TOKEN from the repo-root .env (or env var). Token needs write access.
Get one at https://huggingface.co/settings/tokens.

    python scripts/push_to_hub.py \
        --src local/keyboard-27 \
        --dst tillwenke/robot_learning_project \
        --private \
"""
import argparse
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="local/keyboard-27",
                        help="Local repo_id under ~/.cache/huggingface/lerobot/")
    parser.add_argument("--dst", required=True,
                        help="Target repo_id on the Hub, e.g. tillwenke/robot_learning_project")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--tags", default="so101,teleop,keyboard",
                        help="Comma-separated tags")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not found in .env or environment")

    root = Path.home() / ".cache" / "huggingface" / "lerobot" / args.src
    if not (root / "data").is_dir():
        raise SystemExit(f"No dataset found at {root}")

    os.environ["HF_TOKEN"] = token  # picked up by HfApi inside push_to_hub

    ds = LeRobotDataset(args.src)
    ds.repo_id = args.dst  # push_to_hub uses self.repo_id, ignores any kwarg
    ds.push_to_hub(
        private=args.private,
        tags=[t.strip() for t in args.tags.split(",") if t.strip()],
    )
    print(f"Pushed to https://huggingface.co/datasets/{args.dst}")


if __name__ == "__main__":
    main()
