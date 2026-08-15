# Hardware setup on Windows + WSL2

WSL2 cannot see USB devices natively. We use [`usbipd-win`](https://github.com/dorssel/usbipd-win)
to attach the SO-101 leader and follower (and later a webcam) into the WSL Ubuntu distro.

## One-time setup (per machine)

1. Install `usbipd-win` on Windows:
   ```powershell
   winget install --exact --id dorssel.usbipd-win
   ```
2. Plug in both arms via USB-C (and their power supplies).
3. From an **admin** PowerShell, find the BUSIDs:
   ```powershell
   usbipd list
   ```
   Look for two `1a86:55d3 USB-Enhanced-SERIAL CH343` rows. Note their BUSIDs (e.g. `2-1`, `2-3`).
4. Bind both (sticky — survives reboots):
   ```powershell
   usbipd bind --busid 2-1
   usbipd bind --busid 2-3
   ```
   After this, `usbipd list` shows them as `Shared`.

If `usbipd` is "not recognized" in a fresh shell, either open a new PowerShell window or use the
full path `& "C:\Program Files\usbipd-win\usbipd.exe" ...`.

## Each session (every Windows reboot)

Open **two** admin PowerShell windows and run one auto-attach per arm. Keep both windows open
while you're working — they re-attach the device on every replug.

```powershell
usbipd attach --wsl --busid 2-1 --auto-attach
```

```powershell
usbipd attach --wsl --busid 2-3 --auto-attach
```

Replace BUSIDs with whatever `usbipd list` shows on your machine. They can change if you plug
into a different USB port.

## Verify in WSL

```bash
ls -la /dev/ttyACM*
# expect:
# crw-rw---- 1 root dialout ... /dev/ttyACM0
# crw-rw---- 1 root dialout ... /dev/ttyACM1
```

You should be in the `dialout` group already. If not:
```bash
sudo usermod -aG dialout $USER
# log out + back in
```

## Identify which port is which arm

```bash
conda activate lerobot
lerobot-find-port
```
When prompted, unplug **one** arm's USB-C cable (note which one — leader or follower), press
Enter, and the script reports the port that disappeared. Replug; auto-attach reconnects it.
The remaining `/dev/ttyACM*` is the other arm.

> Note: `lerobot-find-port` is interactive and needs a real TTY. It does **not** work via
> Claude Code's `!` prefix (you'll get `EOFError: EOF when reading a line`). Run it from a
> regular WSL terminal, or use the `usbipd detach` trick described below.

Alternative without an interactive TTY: from Windows admin PowerShell, `usbipd detach --busid <id>`
removes a specific device from WSL while leaving Windows itself unaffected. Watch
`ls /dev/ttyACM*` in WSL to see which one disappears, then `usbipd attach --wsl --busid <id>`
to put it back.

### Current mapping on this machine

| Arm | BUSID | Windows COM | WSL device |
| --- | --- | --- | --- |
| Leader | 2-3 | COM5 | `/dev/ttyACM1` |
| Follower | 2-1 | COM6 | `/dev/ttyACM0` |

Port assignments aren't guaranteed across reboots, but they tend to be stable as long as
you don't replug into different USB ports. Re-verify after a reboot or USB-port change.

If `/dev/ttyACM*` numbering becomes unreliable, add a udev rule to bind stable names by
USB serial (e.g. `/dev/lerobot_leader`).

## Camera

The webcam follows the same flow as the motors:
1. `usbipd list` → find the camera's BUSID (look for `USB2.0_CAM` or `Integrated Camera`)
2. `usbipd bind --busid <id>` (admin, one-time, sticky across reboots)
3. `usbipd attach --wsl --busid <id> --auto-attach` (each session, in its own admin window)

**Current camera on this machine:** BUSID `2-6` → `/dev/video0` in WSL.

After attach, two device nodes appear (`/dev/video0` is the actual stream;
`/dev/video1` is the metadata/control node).

### MJPG required (important)

**`lerobot-find-cameras opencv` will time out** with `select() timeout` — its default
YUYV uncompressed format doesn't survive usbipd's USB-IP forwarding for isochronous
transfers. The camera is fine; we just need to force MJPG (compressed) so the
per-frame USB bandwidth is small.

When using `lerobot-record` (or any lerobot tool that takes a camera config),
always pass `fourcc=MJPG`:

```bash
--robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG}}"
```

To check the camera works, run a Python smoke test instead of `lerobot-find-cameras`:

```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
cfg = OpenCVCameraConfig(
    index_or_path="/dev/video0", fps=30, width=640, height=480,
    color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION, fourcc="MJPG")
with OpenCVCamera(cfg) as cam:
    frame = cam.async_read(timeout_ms=2000)
    print(frame.shape)  # expect (480, 640, 3)
```

Achieved frame rate over usbipd is ~8 fps (USB-IP throttles isochronous bandwidth)
even when 30 is requested — fine for the keyboard task. If higher fps is required
later, an external USB camera plugged directly into the host (native Windows
runtime) is the workaround.

## Troubleshooting

- **Device shows `Shared` but not `Attached`:** the auto-attach window isn't running. Re-run
  `usbipd attach --wsl --busid <id> --auto-attach`.
- **Permission denied on `/dev/ttyACM*`:** user not in `dialout` group, or attach happened
  before the `cdc_acm` driver loaded. Replug and check `dmesg | tail`.
- **Port disappeared mid-session:** the auto-attach window was closed, or the cable wiggled.
  Replug; if auto-attach is still running it'll come back; otherwise re-run the attach command.
