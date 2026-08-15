"""
Train an ACT policy on `tillwenke/robot_learning_project`.

Thin wrapper around the `lerobot-train` CLI with sensible defaults; any extra
CLI args are forwarded to `lerobot-train`, so you can override anything:

    python train.py
    python train.py --steps=200000 --batch_size=16
    python train.py --policy.type=diffusion

Run from inside the policy/.venv (see README.md).
"""

import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ID = "tillwenke/robot_learning_project"
POLICY_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = POLICY_DIR / "outputs"

# Load HF_TOKEN (and any other env) from policy/.env so we don't hit the
# unauthenticated rate limit on the HF Hub.
load_dotenv(POLICY_DIR / ".env")


def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main(argv):
    job_name = f"act_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = OUTPUT_ROOT / job_name
    device = detect_device()
    print(f"[train] using device: {device}")

    default_args = [
        "lerobot-train",
        "--policy.type=act",
        f"--dataset.repo_id={REPO_ID}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.device={device}",
        "--wandb.enable=false",
        # Tuned for tillwenke/robot_learning_project (27 episodes / 4549
        # frames @ 15 fps). Override on the command line if needed.
        "--steps=40000",
        "--policy.chunk_size=50",
        # ACT requires n_action_steps <= chunk_size; default is 100.
        "--policy.n_action_steps=50",
    ]

    # Optional: push the trained checkpoint to the HF Hub. Set
    # HF_POLICY_REPO=<user-or-org>/<repo-name> in policy/.env to enable.
    hf_repo = os.getenv("HF_POLICY_REPO")
    if hf_repo:
        default_args += [
            "--policy.push_to_hub=true",
            f"--policy.repo_id={hf_repo}",
        ]
    else:
        default_args += ["--policy.push_to_hub=false"]

    cmd = default_args + argv
    print("[train] running:", " ".join(shlex.quote(c) for c in cmd))

    # Wipe any pre-existing run at this path; lerobot-train will recreate it.
    if output_dir.exists():
        print(f"[train] removing existing output dir: {output_dir}")
        shutil.rmtree(output_dir)

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("error: `lerobot-train` not found. Activate the venv "
                 "(see policy/README.md) and `pip install -r requirements.txt`.")
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main(sys.argv[1:])
