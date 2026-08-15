"""Fast keyboard-key pixel locator. Self-contained: only needs GEMINI_API_KEY.

Two-tier hybrid:
  1. ORB feature-match the new frame against `keyboard_reference/reference.png`,
     warp the cached anchors (Q,P,Z,M) and SPACE/ENTER pixels through the
     resulting homography. Templates A..L and Z..M from the 4 anchors via the
     canonical QWERTY layout. Total: ~50 ms.
  2. On cache miss (no reference / different keyboard / blur) → one Gemini
     flash-lite call to find Q, P, Z, M, SPACE, ENTER. Refresh the cache.
     ~2-3 s.

Public API:

    from runtime.key_locator import KeyLocator

    loc = KeyLocator()
    loc.prewarm()                            # cheap if cache exists
    key_pixels, source, dt = loc.locate(rgb) # {"Q":(u,v), ..., "ENTER":(u,v)}

`key_pixels` is dict[str, tuple[float, float]] for all 28 letters + SPACE +
ENTER. Drop directly into runtime.geometry.image_keys_to_base_xy via:

    import numpy as np
    key_px_np = {k: np.array(v, dtype=np.float64) for k, v in key_pixels.items()}
    key_base_xy = image_keys_to_base_xy(key_px_np, H_image_to_base)

The reference cache (`runtime/keyboard_reference/`) survives across
processes — first session pays a Gemini call, every later session is
sub-100 ms warm. If the keyboard is swapped mid-session, the next locate()
falls back to Gemini automatically and refreshes the cache.

CLI:
    python -m runtime.key_locator --build  <image.jpg>
    python -m runtime.key_locator --batch  <dir-of-jpgs>
    python -m runtime.key_locator --probe  <image.jpg>
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

# ---- Hard-coded constants -- DO NOT TUNE FROM .env -- only GEMINI_API_KEY -----

# Gemini latency-tuned settings. Tested 2026-05; flash-lite + thinking_level=LOW
# is ~2.4s median vs ~17s on gemini-3.1-pro-preview, with same accuracy on
# our 39-frame Logitech+Cherry test set.
VLM_MODEL = "gemini-3.1-flash-lite"
THINKING_LEVEL = "MEDIUM"
UPSCALE = 2  # 640x480 -> 1280x960; key glyphs are too small for VLM at native.
MAX_RETRIES = 3

# QWERTY template (industry standard 0.75" key pitch). Q at origin.
KEY_PITCH_MM = 19.05
ROW2_STAGGER_MM = 4.75      # ASDFGHJKL stagger from QWERTY
ROW3_STAGGER_MM = 9.50      # ZXCVBNM stagger from QWERTY
ANCHORS = ("Q", "P", "Z", "M")
DIRECT_KEYS = ("SPACE", "ENTER")

def _row(letters, y_mm, x0_mm):
    return {ch: (x0_mm + i * KEY_PITCH_MM, y_mm) for i, ch in enumerate(letters)}

TEMPLATE_MM = {
    **_row("QWERTYUIOP", 0.0, 0.0),
    **_row("ASDFGHJKL", KEY_PITCH_MM, ROW2_STAGGER_MM),
    **_row("ZXCVBNM", 2 * KEY_PITCH_MM, ROW3_STAGGER_MM),
    "SPACE": (85.0, 2.85 * KEY_PITCH_MM),               # below ZXCVBNM, centered
    "ENTER": (12.5 * KEY_PITCH_MM, 0.75 * KEY_PITCH_MM), # column 12.5, ISO-friendly y
}

# ORB parameters tuned for 640x480 keyboard frames.
ORB_FEATURES = 2000
LOWE_RATIO = 0.75
MIN_INLIERS = 12

# Cache location.
HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "keyboard_reference"
REF_IMG = REF_DIR / "reference.png"
REF_JSON = REF_DIR / "reference_layout.json"

load_dotenv()  # picks up GEMINI_API_KEY from a .env at cwd or any parent


# ---- VLM prompt (frozen — change at your own risk) ---------------------------

_PROMPT_BASE = """Look at the photo of a black QWERTY computer keyboard. The layout
may be ANSI (US), ISO (European), Brazilian or similar — adapt to what you see.
There may be a robot arm partially in view; With the position of the keys relative to each other, figure out the keys that are behind the arm if needed and provide the most possible accurate position for each key even if obstructed.

Detect the bounding box of EXACTLY these 6 keys:

  "Q" — leftmost letter of the QWERTY (top letter) row.
  "P" — rightmost letter of the QWERTY (top letter) row.
  "Z" — leftmost letter of the ZXCVBNM (bottom letter) row.
  "M" — rightmost letter of the ZXCVBNM (bottom letter) row.
  "SPACE" — the long horizontal SPACEBAR at the bottom of the alphabet area.
  "ENTER" — the Enter / Return key on the right side. May be a tall L-shape
            (ISO) or a wide rectangle on the home row (ANSI).

Return a JSON list of objects in EXACTLY this format:

[
  {{"label": "Q",     "box_2d": [ymin, xmin, ymax, xmax]}},
  {{"label": "P",     "box_2d": [ymin, xmin, ymax, xmax]}},
  {{"label": "Z",     "box_2d": [ymin, xmin, ymax, xmax]}},
  {{"label": "M",     "box_2d": [ymin, xmin, ymax, xmax]}},
  {{"label": "SPACE", "box_2d": [ymin, xmin, ymax, xmax]}},
  {{"label": "ENTER", "box_2d": [ymin, xmin, ymax, xmax]}}
]

Coordinates are normalized 0-1000 (Gemini standard). y goes top-to-bottom.

Sanity:
  - Q.xmin < P.xmin and Z.xmin < M.xmin
  - Q.ymin < Z.ymin and P.ymin < M.ymin
  - SPACE box is wider than tall.
  - ENTER box is roughly between Q-row and Z-row vertically, far right.
"""

_PROMPT_RETRY = """
Previous attempt failed: anchors were not consistent with a QWERTY layout.
You must place Q and P on the QWERTY (top letter) row.
You must place Z and M on the ZXCVBNM (bottom letter) row.
Then the SPACE bar and ENTER key.
"""


# ---- Gemini client (cached at module level) ----------------------------------

_CLIENT_CACHE: dict[str, object] = {}


def _get_client(api_key: str):
    if api_key not in _CLIENT_CACHE:
        from google import genai
        _CLIENT_CACHE[api_key] = genai.Client(api_key=api_key)
    return _CLIENT_CACHE[api_key]


def _image_to_png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _call_gemini(rgb: np.ndarray, prompt: str, api_key: str) -> list:
    from google.genai import types
    client = _get_client(api_key)
    image_part = types.Part.from_bytes(
        data=_image_to_png_bytes(rgb), mime_type="image/png"
    )
    response = client.models.generate_content(
        model=VLM_MODEL,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        ),
    )
    text = response.text
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw, _ = json.JSONDecoder().raw_decode(text.strip())
    if isinstance(raw, dict):
        for key in ("items", "results", "boxes", "anchors"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        return [{"label": k, "box_2d": v} for k, v in raw.items()
                if isinstance(v, (list, tuple))]
    if not isinstance(raw, list):
        raise RuntimeError(f"unexpected VLM response shape: {type(raw)}")
    return raw


# ---- Geometry: project the QWERTY template through the 4 anchors -------------

def _project_keys(anchor_pixels: dict) -> dict:
    """Compute homography template->image from 4 anchors, project all 28 keys."""
    template_pts = np.array([TEMPLATE_MM[a] for a in ANCHORS], dtype=np.float64)
    pixel_pts = np.array([anchor_pixels[a] for a in ANCHORS], dtype=np.float64)
    H, _ = cv2.findHomography(template_pts, pixel_pts)
    if H is None:
        raise ValueError("homography failed (anchors degenerate?)")
    out = {}
    for name, mm in TEMPLATE_MM.items():
        h = np.array([mm[0], mm[1], 1.0], dtype=np.float64)
        p = H @ h
        p /= p[2]
        out[name] = (float(p[0]), float(p[1]))
    return out


# ---- Anchor parsing + validation ---------------------------------------------

def _bbox_center(box_2d, W, H):
    ymin, xmin, ymax, xmax = box_2d
    return (xmin + xmax) / 2.0 / 1000.0 * W, (ymin + ymax) / 2.0 / 1000.0 * H


def _parse_centers(items, W, H) -> dict:
    """Walk the VLM response, extract pixel centers for accepted labels."""
    accepted = set(ANCHORS) | set(DIRECT_KEYS)
    centers = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        label = it.get("label") or it.get("name")
        box = it.get("box_2d") or it.get("bbox") or it.get("box")
        if label is None or box is None:
            continue
        label = str(label).strip().strip('"').upper()
        if label.endswith(" KEY"):
            label = label[:-4]
        if label == "RETURN":
            label = "ENTER"
        if label == "SPACEBAR":
            label = "SPACE"
        if label not in accepted:
            continue
        try:
            box = [float(x) for x in box]
            if len(box) != 4:
                continue
        except (TypeError, ValueError):
            continue
        u, v = _bbox_center(box, W, H)
        centers[label] = (u, v)
    return centers


def _validate_anchors(anchors, W, H) -> tuple[bool, str]:
    for k in ANCHORS:
        if k not in anchors:
            return False, f"missing anchor {k!r}"
        u, v = anchors[k]
        if not (0 <= u < W and 0 <= v < H):
            return False, f"anchor {k!r} out of bounds"
    if not anchors["Q"][0] < anchors["P"][0]:
        return False, "Q.x must be < P.x"
    if not anchors["Z"][0] < anchors["M"][0]:
        return False, "Z.x must be < M.x"
    if not anchors["Q"][1] < anchors["Z"][1]:
        return False, "Q.y must be < Z.y"
    if not anchors["P"][1] < anchors["M"][1]:
        return False, "P.y must be < M.y"
    return True, ""


def _validate_direct(name, u, v, anchors, W, H) -> tuple[bool, str]:
    """Topology bounds derived from the 4-anchor QWERTY grid (no template
    drift comparison — template extrapolation is unreliable on tilted views,
    so we accept any Gemini direct value that lives in a plausible region)."""
    if not (0 <= u < W and 0 <= v < H):
        return False, "out-of-bounds"
    col_dx = (anchors["P"][0] - anchors["Q"][0]) / 9.0
    col_dy = (anchors["P"][1] - anchors["Q"][1]) / 9.0
    col_w = float(np.hypot(col_dx, col_dy))
    row_span = max(anchors["Z"][1], anchors["M"][1]) - min(anchors["Q"][1], anchors["P"][1])
    if name == "ENTER":
        dx_from_p = u - anchors["P"][0]
        if not (1.5 * col_w <= dx_from_p <= 5.0 * col_w):
            return False, f"bad-x-offset({dx_from_p:.0f}px)"
        min_y = min(anchors["Q"][1], anchors["P"][1]) + 0.3 * row_span
        max_y = max(anchors["Z"][1], anchors["M"][1])
        if not (min_y <= v <= max_y):
            return False, f"bad-y({v:.0f})"
    elif name == "SPACE":
        if v <= max(anchors["Z"][1], anchors["M"][1]):
            return False, "above-bottom-row"
        if v > max(anchors["Z"][1], anchors["M"][1]) + 1.2 * row_span:
            return False, "too-low"
        if not (anchors["Q"][0] - col_w <= u <= anchors["P"][0] + col_w):
            return False, "bad-x"
    return True, ""


# ---- Cold path: VLM detection + layout build --------------------------------

def _gemini_detect(rgb: np.ndarray) -> dict:
    """One Gemini call; returns layout = {anchors, direct_keys, keys}."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment / .env")
    H_img, W_img = rgb.shape[:2]
    upscaled = (rgb if UPSCALE == 1 else
                cv2.resize(rgb, (W_img * UPSCALE, H_img * UPSCALE),
                           interpolation=cv2.INTER_CUBIC))

    last_err = ""
    for attempt in range(MAX_RETRIES):
        prompt = _PROMPT_BASE if attempt == 0 else _PROMPT_BASE + _PROMPT_RETRY
        try:
            items = _call_gemini(upscaled, prompt, api_key)
        except Exception as e:
            last_err = f"VLM call failed: {e}"
            print(f"[key_locator] attempt {attempt+1}: {last_err}", file=sys.stderr)
            continue
        centers = _parse_centers(items, W_img, H_img)
        anchors = {k: centers[k] for k in ANCHORS if k in centers}
        ok, why = _validate_anchors(anchors, W_img, H_img)
        if not ok:
            last_err = why
            print(f"[key_locator] attempt {attempt+1} validation: {why}",
                  file=sys.stderr)
            continue
        keys = _project_keys(anchors)
        direct_used = {}
        for k in DIRECT_KEYS:
            if k not in centers:
                continue
            u, v = centers[k]
            ok2, _ = _validate_direct(k, u, v, anchors, W_img, H_img)
            if ok2:
                keys[k] = (u, v)
                direct_used[k] = (u, v)
        return {
            "image_width": W_img, "image_height": H_img,
            "anchors": {k: list(anchors[k]) for k in ANCHORS},
            "direct_keys": {k: list(v) for k, v in direct_used.items()},
            "keys": {k: list(v) for k, v in keys.items()},
        }
    raise RuntimeError(f"all {MAX_RETRIES} VLM attempts failed; last: {last_err}")


def _save_reference(bgr: np.ndarray, layout: dict) -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(REF_IMG), bgr)
    REF_JSON.write_text(json.dumps({
        "image_width": layout["image_width"],
        "image_height": layout["image_height"],
        "anchors": layout["anchors"],
        "direct_keys": layout.get("direct_keys", {}),
    }, indent=2))


# ---- Warm path: ORB matcher --------------------------------------------------

def _orb():
    return cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=8)


class _ORBMatcher:
    """Match a new frame against the saved reference; warp anchors + cached
    SPACE/ENTER through the same homography."""

    def __init__(self):
        if not REF_IMG.exists() or not REF_JSON.exists():
            raise FileNotFoundError(f"no reference at {REF_DIR}")
        bgr = cv2.imread(str(REF_IMG))
        info = json.loads(REF_JSON.read_text())
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        orb = _orb()
        self.kp_ref, self.des_ref = orb.detectAndCompute(gray, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_anchor_px = np.array(
            [info["anchors"][k] for k in ANCHORS], dtype=np.float64)
        direct = info.get("direct_keys", {}) or {}
        self.direct_names = [k for k in DIRECT_KEYS if k in direct]
        self.ref_direct_px = (np.array([direct[k] for k in self.direct_names],
                                       dtype=np.float64)
                              if self.direct_names else None)

    def detect(self, rgb: np.ndarray) -> dict:
        H_img, W_img = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp, des = _orb().detectAndCompute(gray, None)
        if des is None or len(kp) < MIN_INLIERS:
            raise RuntimeError(f"too few keypoints: {0 if kp is None else len(kp)}")
        matches = self.bf.knnMatch(self.des_ref, des, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < LOWE_RATIO * n.distance]
        if len(good) < MIN_INLIERS:
            raise RuntimeError(f"only {len(good)} good matches")
        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hmat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if Hmat is None or int(mask.sum()) < MIN_INLIERS:
            raise RuntimeError("RANSAC failed")
        ref_pts = self.ref_anchor_px.reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(ref_pts, Hmat).reshape(-1, 2)
        anchors = {ANCHORS[i]: (float(warped[i, 0]), float(warped[i, 1]))
                   for i in range(4)}
        ok, why = _validate_anchors(anchors, W_img, H_img)
        if not ok:
            raise RuntimeError(f"warped anchors fail QWERTY sanity: {why}")
        keys = _project_keys(anchors)
        if self.ref_direct_px is not None and len(self.ref_direct_px):
            dp = cv2.perspectiveTransform(
                self.ref_direct_px.reshape(-1, 1, 2), Hmat).reshape(-1, 2)
            for i, name in enumerate(self.direct_names):
                keys[name] = (float(dp[i, 0]), float(dp[i, 1]))
        return {
            "image_width": W_img, "image_height": H_img,
            "anchors": {k: list(anchors[k]) for k in ANCHORS},
            "keys": {k: list(v) for k, v in keys.items()},
        }


# ---- Public API --------------------------------------------------------------

class KeyLocator:
    """Hybrid ORB+Gemini keyboard-key pixel locator. Stateless across calls;
    re-instantiates the ORB matcher each time so a Gemini-driven cache refresh
    in one call is visible to the next."""

    def has_reference(self) -> bool:
        return REF_IMG.exists() and REF_JSON.exists()

    def prewarm(self, image_rgb: np.ndarray | None = None) -> str:
        """Returns 'orb' if cache hit, 'gemini' if seeded a fresh cache from
        image_rgb, 'noop' otherwise. Call concurrently with robot.connect().
        """
        if self.has_reference():
            _ORBMatcher()  # builds descriptors
            return "orb"
        if image_rgb is None:
            return "noop"
        layout = _gemini_detect(image_rgb)
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _save_reference(bgr, layout)
        return "gemini"

    def locate(self, image_rgb: np.ndarray, force_gemini: bool = False
               ) -> tuple[dict[str, tuple[float, float]], str, float]:
        """Returns ({KEY: (u, v)}, source, elapsed_seconds).
        source ∈ {'orb','gemini'}. ~50 ms warm, ~2-3 s on cache miss.

        Pass force_gemini=True to bypass ORB and always call the VLM. Used
        by eval-3, where the keyboard may move between rollouts and the
        ORB cache against a stale reference would give wrong positions.
        Each call still refreshes the reference image so the cache stays
        usable for other tools.
        """
        t0 = time.perf_counter()
        if force_gemini:
            layout = _gemini_detect(image_rgb)
            _save_reference(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), layout)
            source = "gemini"
        else:
            try:
                layout = _ORBMatcher().detect(image_rgb)
                source = "orb"
            except Exception:
                layout = _gemini_detect(image_rgb)
                _save_reference(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), layout)
                source = "gemini"
        dt = time.perf_counter() - t0
        return ({k: (float(v[0]), float(v[1])) for k, v in layout["keys"].items()},
                source, dt)


# ---- Debug overlay -----------------------------------------------------------

def draw_overlay(rgb: np.ndarray, layout: dict, out_path: Path) -> None:
    """Big red dots on Q/P/Z/M and SPACE/ENTER, small green on the rest."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    if "anchors" in layout:
        quad = np.array([layout["anchors"]["Q"], layout["anchors"]["P"],
                         layout["anchors"]["M"], layout["anchors"]["Z"]],
                        dtype=np.int32)
        cv2.polylines(bgr, [quad], True, (0, 220, 255), 2)
    anchor_set = set(layout.get("anchors", {}).keys()) or set(ANCHORS)
    for name, (u, v) in layout["keys"].items():
        u, v = int(u), int(v)
        emph = name in anchor_set or name in DIRECT_KEYS
        color = ((0, 0, 255) if name in anchor_set
                 else ((0, 0, 200) if emph else (0, 180, 0)))
        cv2.circle(bgr, (u, v), 7 if emph else 3, color, -1 if emph else 1)
        cv2.putText(bgr, name, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)


# ---- CLI ---------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, metavar="IMG",
                    help="solve+save reference from one frame (one Gemini call)")
    ap.add_argument("--probe", type=Path, metavar="IMG",
                    help="locate keys in one frame; print + dump overlay")
    ap.add_argument("--batch", type=Path, metavar="DIR",
                    help="run hybrid pipeline over every JPG in DIR")
    args = ap.parse_args()
    if args.build:
        loc = KeyLocator()
        bgr = cv2.imread(str(args.build))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        REF_DIR.mkdir(parents=True, exist_ok=True)
        layout = _gemini_detect(rgb)
        _save_reference(bgr, layout)
        n_direct = len(layout.get("direct_keys", {}))
        print(f"reference saved -> {REF_IMG} (4 anchors + {n_direct} direct keys)")
    elif args.probe:
        loc = KeyLocator()
        bgr = cv2.imread(str(args.probe))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        keys, src, dt = loc.locate(rgb)
        print(f"source={src}  dt={dt*1000:.0f}ms")
        for k in ("Q", "P", "Z", "M", "L", "SPACE", "ENTER"):
            print(f"  {k}: {keys[k]}")
        layout_for_overlay = {"anchors": {k: keys[k] for k in ANCHORS},
                              "keys": keys}
        out = HERE / f"key_locator_probe_{args.probe.stem}.png"
        draw_overlay(rgb, layout_for_overlay, out)
        print(f"overlay -> {out}")
    elif args.batch:
        loc = KeyLocator()
        out_dir = HERE / "key_locator_batch_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(args.batch.glob("*.jpg")) + sorted(args.batch.glob("*.png"))
        times, who = [], []
        for i, p in enumerate(frames):
            bgr = cv2.imread(str(p))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                keys, src, dt = loc.locate(rgb)
                times.append(dt); who.append(src)
                draw_overlay(rgb, {"anchors": {k: keys[k] for k in ANCHORS},
                                   "keys": keys}, out_dir / f"{p.stem}.png")
                print(f"  [{i:02d}] {dt*1000:6.0f}ms  {src.upper():>6}  {p.name}")
            except Exception as e:
                who.append("FAIL")
                print(f"  [{i:02d}] FAIL  {e}  {p.name}")
        if times:
            print(f"\norb={who.count('orb')}  gemini={who.count('gemini')}  "
                  f"fail={who.count('FAIL')}  "
                  f"median={sorted(times)[len(times)//2]*1000:.0f}ms  "
                  f"max={max(times)*1000:.0f}ms")
            print(f"overlays -> {out_dir}")
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
