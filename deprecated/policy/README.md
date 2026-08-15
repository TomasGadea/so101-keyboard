# policy

Training / inspection scripts for LeRobot policies on the
`tillwenke/robot_learning_project` dataset. Keep a separate virtualenv here
because LeRobot pins versions that can clash with the rest of the repo.

## Setup

LeRobot requires Python **3.10** (supports 3.10–3.12). If `python3.10` is
not installed:


Then create the venv:

```bash
cd policy
conda create -y -n policy python=3.12
conda activate policy
pip install -r requirements.txt
```

To leave the env: `deactivate`.

## Scripts

### `inspect_dataset.py`
Prints features (name / shape / dtype), episode and frame counts, camera
keys, and a single sample. Use it to confirm the input/output dims an ACT
policy will see.

```bash
source .venv/bin/activate
python inspect_dataset.py
```

### `train.py`
ACT training entrypoint. Thin wrapper around `lerobot-train`: sets the
policy type, dataset, a timestamped `outputs/<job>/` directory, and forwards
any extra CLI args.

```bash
source .venv/bin/activate
python train.py                                # defaults
python train.py --steps=200000 --batch_size=16 # override hyperparams
python train.py --policy.device=cpu            # CPU instead of CUDA
```

Checkpoints land in `policy/outputs/<job_name>/`.

To also push the trained policy to the HF Hub, set `HF_POLICY_REPO` in
`.env` (e.g. `tillwenke/robot_learning_act`). `HF_TOKEN` must have write
access to that namespace.

## HF token

Copy `example.env` to `.env` and set `HF_TOKEN` to avoid the unauthenticated
HF Hub rate limit. `train.py` and `inspect_dataset.py` load it
automatically via `python-dotenv`.

```bash
cp example.env .env
# then edit .env and paste your token
```

## Notes
- `.venv/` and `.env` should be gitignored.
- If `lerobot` isn't available on PyPI for your platform, install from source:
  `pip install "lerobot @ git+https://github.com/huggingface/lerobot.git"`.
