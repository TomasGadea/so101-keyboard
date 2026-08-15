"""
Inspect the HF LeRobot dataset `tillwenke/robot_learning_project`.

Prints feature names + shapes + dtypes, episode/frame counts, and a single
sample so you can confirm the inputs/outputs an ACT policy would see.

Run:
    python policy/inspect_dataset.py
"""

from pathlib import Path

from dotenv import load_dotenv

from lerobot.datasets.lerobot_dataset import LeRobotDataset

load_dotenv(Path(__file__).resolve().parent / ".env")

REPO_ID = "tillwenke/robot_learning_project"


def main():
    ds = LeRobotDataset(REPO_ID)

    print(f"repo_id:        {REPO_ID}")
    print(f"num_episodes:   {ds.num_episodes}")
    print(f"num_frames:     {ds.num_frames}")
    print(f"fps:            {ds.fps}")
    print(f"robot_type:     {getattr(ds.meta, 'robot_type', None)}")
    print()

    print("features:")
    for name, feat in ds.features.items():
        shape = feat.get("shape")
        dtype = feat.get("dtype")
        print(f"  {name:35s} shape={shape}  dtype={dtype}")
    print()

    print("camera keys:    ", ds.meta.camera_keys)
    print()

    sample = ds[0]
    print("sample[0] tensors:")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k:35s} {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k:35s} {type(v).__name__}={v}")


if __name__ == "__main__":
    main()
