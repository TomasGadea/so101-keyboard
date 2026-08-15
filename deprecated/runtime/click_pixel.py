"""Open an image, capture one click, print 'u v' to stdout.

Usage:
  python runtime/click_pixel.py calibration/calib_images/KEY_Q_raw_*.png

Then copy the printed 'u v' into the calibration prompt. Picks the
newest matching file if you pass a glob (handy for the latest snap).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: click_pixel.py <image-path-or-glob>", file=sys.stderr)
        return 2

    pat = sys.argv[1]
    matches = sorted(glob.glob(pat))
    if not matches:
        print(f"no file matched {pat}", file=sys.stderr)
        return 1
    path = Path(matches[-1])
    print(f"opening {path}")

    img = plt.imread(path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.set_title(f"{path.name}\nclick the feature, then close the window")

    clicked = {"u": None, "v": None}

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        u = int(round(event.xdata))
        v = int(round(event.ydata))
        clicked["u"], clicked["v"] = u, v
        print(f"\n{u} {v}\n", flush=True)
        ax.set_title(f"clicked ({u}, {v}) — close window to exit")
        ax.plot(u, v, "rx", markersize=15, markeredgewidth=2)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if clicked["u"] is None:
        print("no click registered", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
