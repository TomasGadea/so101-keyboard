"""Local OCR keyboard locator for pipeline.py.

This module uses RapidOCR to read visible key labels, fits the known QWERTY
keyboard geometry with RANSAC, and returns the same normalized_1000 JSON shape
as the Gemini route.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


KEYS_JSON = Path(__file__).resolve().parent / "keys.json"
TARGET_SIZE = (640, 480)
UPSCALE = 2
MIN_SCORE = 0.55
STRONG_UPSCALE = 3
STRONG_MIN_SCORE = 0.45
STRONG_TEXT_SCORE = 0.30
STRONG_BOX_THRESH = 0.30
RANSAC_REPROJ_THRESHOLD_PX = 12.0
MIN_UNIQUE_INLIERS = 4
STRONG_MIN_UNIQUE_INLIERS = 6
STRONG_EARLY_ACCEPT_INLIERS = 8
STRONG_EARLY_ACCEPT_ROWS = 3
STRONG_FAST_ACCEPT_INLIERS = int(os.getenv("OCR_STRONG_FAST_ACCEPT_INLIERS", "8"))
STRONG_FAST_ACCEPT_ROWS = int(os.getenv("OCR_STRONG_FAST_ACCEPT_ROWS", "3"))
STRONG_FAST_ACCEPT_MAX_ERROR_PX = float(
    os.getenv("OCR_STRONG_FAST_ACCEPT_MAX_ERROR_PX", "4.0")
)
STRONG_FAST_SKIP_FALLBACK_UNIQUE_LABELS = int(
    os.getenv("OCR_STRONG_FAST_SKIP_FALLBACK_UNIQUE_LABELS", "14")
)
STRONG_FALLBACK_LOW_LABEL_COUNT = int(
    os.getenv("OCR_STRONG_FALLBACK_LOW_LABEL_COUNT", "5")
)
STRONG_FALLBACK_MAX_PASSES = int(os.getenv("OCR_STRONG_FALLBACK_MAX_PASSES", "4"))
STRONG_FALLBACK_MAX_PASSES_LOW_LABELS = int(
    os.getenv("OCR_STRONG_FALLBACK_MAX_PASSES_LOW_LABELS", "8")
)
DEFAULT_OCR_DEADLINE_S = float(os.getenv("OCR_DEADLINE_S", "0") or "0")
OCR_PASS_TIME_BUDGET_S = float(os.getenv("OCR_PASS_TIME_BUDGET_S", "1.15"))
STRONG_ANGLES = (0.0, -10.0, 10.0, -20.0, 20.0, -25.0, 25.0, -30.0, 30.0)
STRONG_LIGHTING_VARIANTS = ("none", "clahe", "brighten", "darken")
STRONG_ROTATED_LIGHTING_VARIANTS = ("clahe",)

KEY_PITCH_MM = 19.05
ROW2_STAGGER_MM = 4.75
ROW3_STAGGER_MM = 9.50
ENTER_X_PITCH = float(os.getenv("OCR_ENTER_X_PITCH", "12.8"))
ENTER_Y_PITCH = float(os.getenv("OCR_ENTER_Y_PITCH", "0.0"))
ENTER_LABEL_X_PITCH = float(os.getenv("OCR_ENTER_LABEL_X_PITCH", "12.55"))
ENTER_LABEL_Y_PITCH = float(os.getenv("OCR_ENTER_LABEL_Y_PITCH", "0.18"))
ENTER_DIRECT_CORRECTION = os.getenv("OCR_ENTER_DIRECT_CORRECTION", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
ENTER_DIRECT_VALIDATION_MAX_PITCH = float(
    os.getenv("OCR_ENTER_DIRECT_VALIDATION_MAX_PITCH", "3.0")
)
REQUESTED_KEY_MARGIN_PX = float(os.getenv("OCR_REQUESTED_KEY_MARGIN_PX", "6.0"))
STRONG_LABEL_OFFSET_MM = (
    float(os.getenv("OCR_STRONG_LABEL_OFFSET_X_PITCH", "0.0")) * KEY_PITCH_MM,
    float(os.getenv("OCR_STRONG_LABEL_OFFSET_Y_PITCH", "0.0")) * KEY_PITCH_MM,
)
ENTER_SUPPORT_LABELS = set("OPKLM")


def _row(letters, y_mm, x0_mm, row_index):
    return {
        ch: {"pt": (x0_mm + i * KEY_PITCH_MM, y_mm), "row": row_index}
        for i, ch in enumerate(letters)
    }


TEMPLATE = {
    **_row("QWERTYUIOP", 0.0, 0.0, 0),
    **_row("ASDFGHJKL", KEY_PITCH_MM, ROW2_STAGGER_MM, 1),
    **_row("ZXCVBNM", 2 * KEY_PITCH_MM, ROW3_STAGGER_MM, 2),
    "SPACE": {"pt": (85.0, 2.85 * KEY_PITCH_MM), "row": 3},
    "ENTER": {"pt": (ENTER_X_PITCH * KEY_PITCH_MM,
                     ENTER_Y_PITCH * KEY_PITCH_MM), "row": 0},
}

LETTER_KEYS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPECIAL_ALIASES = {
    "ENTER": "ENTER",
    "RETURN": "ENTER",
    "SPACE": "SPACE",
    "SPACEBAR": "SPACE",
}

_ENGINE = None

VERBOSE = False


def vprint(*args, **kwargs):
    """print() that only emits when this module's VERBOSE flag is set."""
    if VERBOSE:
        print(*args, **kwargs)


@dataclass(frozen=True)
class Detection:
    label: str
    center: tuple[float, float]
    score: float
    text: str
    angle: float = 0.0


@dataclass(frozen=True)
class FitResult:
    H: np.ndarray
    projected: dict[str, tuple[float, float]]
    inlier_labels: set[str]
    inlier_rows: set[int]
    mean_error_px: float
    fit_elapsed_s: float
    label_offset_mm: tuple[float, float]
    fit_model: str = "homography"


def _get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    t0 = time.perf_counter()
    try:
        from rapidocr import (
            EngineType,
            LangDet,
            LangRec,
            ModelType,
            OCRVersion,
            RapidOCR,
        )
    except ImportError as exc:
        raise RuntimeError(
            "OCR mode requires rapidocr and onnxruntime. "
            "Install with: pip install rapidocr==3.8.1 onnxruntime==1.26.0"
        ) from exc

    # RapidOCR 3.8.1 supports PP-OCRv5 detection through the CH detector.
    # Recognition is English-only, which is what the keyboard labels need.
    _ENGINE = RapidOCR(
        params={
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.EN,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        }
    )
    vprint(f"[ocr] engine init: {time.perf_counter() - t0:.2f}s")
    return _ENGINE


def _normalize_label(text: str) -> str | None:
    raw = str(text)
    cleaned = re.sub(r"[^A-Za-z]", "", raw).upper()
    if not cleaned:
        return None
    if cleaned in SPECIAL_ALIASES:
        return SPECIAL_ALIASES[cleaned]
    if len(cleaned) == 1 and cleaned in LETTER_KEYS:
        # Avoid accepting number-row/function-key OCR like "F5" as the letter F.
        if re.search(r"\d", raw):
            return None
        return cleaned
    return None


def _polygon_center(box) -> tuple[float, float]:
    pts = np.asarray(box, dtype=np.float64).reshape(-1, 2)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _read_image(image_path, target_size=TARGET_SIZE):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    if (image.shape[1], image.shape[0]) != tuple(target_size):
        image = cv2.resize(image, tuple(target_size), interpolation=cv2.INTER_AREA)
    return image


def _apply_affine(point, matrix) -> tuple[float, float]:
    x, y = point
    mapped = matrix @ np.array([x, y, 1.0], dtype=np.float64)
    return float(mapped[0]), float(mapped[1])


def _rotate_for_ocr(image, angle: float):
    height, width = image.shape[:2]
    if abs(angle) < 1e-6:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return image, identity
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    rotated_width = int(height * sin + width * cos)
    rotated_height = int(height * cos + width * sin)
    matrix[0, 2] += rotated_width / 2.0 - center[0]
    matrix[1, 2] += rotated_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        image,
        matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    inverse = cv2.invertAffineTransform(matrix)
    return rotated, inverse


def _crop_dark_region(image, margin: int = 24):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < 245
    if not np.any(mask):
        return image, (0.0, 0.0)
    ys, xs = np.where(mask)
    height, width = image.shape[:2]
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(width, int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(height, int(ys.max()) + margin + 1)
    return image[y0:y1, x0:x1], (float(x0), float(y0))


def _adjust_brightness_contrast(image, alpha: float, beta: float):
    adjusted = image.astype(np.float32) * alpha + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_lighting_variant(image, variant: str):
    if variant == "none":
        return image
    if variant == "brighten":
        return _adjust_brightness_contrast(image, alpha=1.10, beta=24.0)
    if variant == "darken":
        return _adjust_brightness_contrast(image, alpha=1.20, beta=-28.0)
    if variant == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    raise ValueError(f"Unknown OCR lighting variant: {variant}")


def _extract_detections(
    result,
    upscale=UPSCALE,
    min_score: float = MIN_SCORE,
    angle: float = 0.0,
    inverse_matrix=None,
    source_offset=(0.0, 0.0),
    image_size: tuple[int, int] | None = None,
) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or texts is None or scores is None:
        return []

    detections = []
    for box, text, score in zip(boxes, texts, scores):
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if score < min_score:
            continue
        label = _normalize_label(text)
        if label is None:
            continue
        cx, cy = _polygon_center(box)
        center = (
            cx / upscale + source_offset[0],
            cy / upscale + source_offset[1],
        )
        if inverse_matrix is not None:
            center = _apply_affine(center, inverse_matrix)
        if image_size is not None:
            width, height = image_size
            if not (-2.0 <= center[0] < width + 2.0 and
                    -2.0 <= center[1] < height + 2.0):
                continue
        detections.append(
            Detection(
                label=label,
                center=center,
                score=score,
                text=str(text),
                angle=angle,
            )
        )
    return detections


def _candidate_summary(detections: list[Detection]) -> str:
    best = {}
    for det in detections:
        if det.label not in best or det.score > best[det.label].score:
            best[det.label] = det
    labels = sorted(best)
    return ", ".join(f"{label}:{best[label].score:.2f}" for label in labels)


def _deadline_from_seconds(deadline_s: float | None):
    if deadline_s is None:
        deadline_s = DEFAULT_OCR_DEADLINE_S
    try:
        deadline_s = float(deadline_s)
    except (TypeError, ValueError):
        return None
    if deadline_s <= 0:
        return None
    return time.perf_counter() + deadline_s


def _deadline_remaining_s(deadline) -> float | None:
    if deadline is None:
        return None
    return deadline - time.perf_counter()


def _can_start_ocr_pass(deadline, reserve_s: float = OCR_PASS_TIME_BUDGET_S) -> bool:
    remaining = _deadline_remaining_s(deadline)
    return remaining is None or remaining > reserve_s


def _deadline_status(deadline) -> str:
    remaining = _deadline_remaining_s(deadline)
    if remaining is None:
        return "none"
    return f"{remaining:.2f}s remaining"


def _template_point(label: str, label_offset_mm=(0.0, 0.0)):
    x, y = TEMPLATE[label]["pt"]
    return x + label_offset_mm[0], y + label_offset_mm[1]


def _project_template_point(H, point) -> tuple[float, float]:
    src = np.array([point[0], point[1], 1.0], dtype=np.float64)
    dst = H @ src
    dst /= dst[2]
    return float(dst[0]), float(dst[1])


def _projected_pitch(projected) -> float:
    q = np.array(projected["Q"], dtype=np.float64)
    p = np.array(projected["P"], dtype=np.float64)
    return float(np.linalg.norm(p - q) / 9.0)


def _direct_enter_center_from_label(H, detection: Detection) -> tuple[float, float]:
    label_pt = (
        ENTER_LABEL_X_PITCH * KEY_PITCH_MM,
        ENTER_LABEL_Y_PITCH * KEY_PITCH_MM,
    )
    center_pt = TEMPLATE["ENTER"]["pt"]
    projected_label = np.array(_project_template_point(H, label_pt), dtype=np.float64)
    projected_center = np.array(_project_template_point(H, center_pt), dtype=np.float64)
    corrected = np.array(detection.center, dtype=np.float64) + (
        projected_center - projected_label
    )
    return float(corrected[0]), float(corrected[1])


def _try_fit_keyboard_homography(
    detections: list[Detection],
    width: int,
    height: int,
    min_unique_inliers: int,
    label_offset_mm=(0.0, 0.0),
    requested_keys: list[str] | None = None,
) -> FitResult | None:
    letter_detections = [det for det in detections if det.label in LETTER_KEYS]
    unique_letters = {det.label for det in letter_detections}
    if len(unique_letters) < min_unique_inliers:
        return None

    src = np.array([_template_point(det.label, label_offset_mm)
                    for det in letter_detections],
                   dtype=np.float64)
    dst = np.array([det.center for det in letter_detections], dtype=np.float64)
    t0 = time.perf_counter()
    H_homography, homography_mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD_PX,
    )
    fit_elapsed = time.perf_counter() - t0

    candidates: list[FitResult] = []
    homography_fit = _build_fit_result(
        H_homography,
        homography_mask,
        letter_detections,
        width=width,
        height=height,
        detections=detections,
        label_offset_mm=label_offset_mm,
        fit_elapsed=fit_elapsed,
        fit_model="homography",
        min_unique_inliers=min_unique_inliers,
        requested_keys=requested_keys,
    )
    if homography_fit is not None:
        candidates.append(homography_fit)

    t0 = time.perf_counter()
    affine, affine_mask = cv2.estimateAffine2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD_PX,
    )
    fit_elapsed = time.perf_counter() - t0
    if affine is not None:
        H_affine = np.vstack([affine, np.array([0.0, 0.0, 1.0])])
        affine_fit = _build_fit_result(
            H_affine,
            affine_mask,
            letter_detections,
            width=width,
            height=height,
            detections=detections,
            label_offset_mm=label_offset_mm,
            fit_elapsed=fit_elapsed,
            fit_model="affine",
            min_unique_inliers=min_unique_inliers,
            requested_keys=requested_keys,
        )
        if affine_fit is not None:
            candidates.append(affine_fit)

    if not candidates:
        return None
    return max(candidates, key=_fit_rank)


def _build_fit_result(
    H,
    mask,
    letter_detections: list[Detection],
    width: int,
    height: int,
    detections: list[Detection],
    label_offset_mm: tuple[float, float],
    fit_elapsed: float,
    fit_model: str,
    min_unique_inliers: int,
    requested_keys: list[str] | None,
) -> FitResult | None:
    if H is None or mask is None:
        return None

    inlier_dets = [
        det for det, is_inlier in zip(letter_detections, mask.ravel())
        if int(is_inlier)
    ]
    inlier_labels = {det.label for det in inlier_dets}
    inlier_rows = {TEMPLATE[det.label]["row"] for det in inlier_dets}
    if len(inlier_labels) < min_unique_inliers or len(inlier_rows) < 2:
        return None

    if inlier_dets:
        inlier_src = np.array([_template_point(det.label, label_offset_mm)
                               for det in inlier_dets],
                              dtype=np.float64).reshape(-1, 1, 2)
        inlier_dst = np.array([det.center for det in inlier_dets],
                              dtype=np.float64)
        projected_inliers = cv2.perspectiveTransform(inlier_src, H).reshape(-1, 2)
        errors = np.linalg.norm(projected_inliers - inlier_dst, axis=1)
        mean_error_px = float(np.mean(errors))
    else:
        mean_error_px = float("inf")
    projected = _project_all_keys(H)
    try:
        _validate_projected_layout(
            projected,
            width,
            height,
            detections,
            H,
            requested_keys=requested_keys,
        )
    except RuntimeError:
        return None

    return FitResult(
        H=H,
        projected=projected,
        inlier_labels=inlier_labels,
        inlier_rows=inlier_rows,
        mean_error_px=mean_error_px,
        fit_elapsed_s=fit_elapsed,
        label_offset_mm=label_offset_mm,
        fit_model=fit_model,
    )


def _fit_keyboard_homography(
    detections: list[Detection],
    width: int,
    height: int,
    requested_keys: list[str],
    min_unique_inliers: int = MIN_UNIQUE_INLIERS,
    label_offset_mm=(0.0, 0.0),
    log_prefix="[ocr]",
) -> FitResult:
    fit = _try_fit_keyboard_homography(
        detections,
        width=width,
        height=height,
        min_unique_inliers=min_unique_inliers,
        label_offset_mm=label_offset_mm,
        requested_keys=requested_keys,
    )
    if fit is None:
        unique_letters = {
            det.label for det in detections if det.label in LETTER_KEYS
        }
        raise RuntimeError(
            "OCR could not fit keyboard homography: "
            f"detected labels={_candidate_summary(detections) or 'none'}, "
            f"inliers={len(unique_letters)}, missing requested keys={requested_keys}"
        )
    vprint(
        f"{log_prefix} {fit.fit_model} fit: {fit.fit_elapsed_s:.3f}s, "
        f"inliers={len(fit.inlier_labels)}, rows={len(fit.inlier_rows)}, "
        f"err={fit.mean_error_px:.1f}px"
    )
    return fit


def _project_all_keys(H) -> dict[str, tuple[float, float]]:
    out = {}
    for name, spec in TEMPLATE.items():
        out[name] = _project_template_point(H, spec["pt"])
    return out


def _validate_projected_layout(
    projected,
    width,
    height,
    detections,
    H,
    requested_keys: list[str] | None = None,
):
    for key in ("Q", "P", "Z", "M"):
        x, y = projected[key]
        if not (0 <= x < width and 0 <= y < height):
            raise RuntimeError(f"OCR projected {key} out of bounds: {(x, y)}")
    if not projected["Q"][0] < projected["P"][0]:
        raise RuntimeError("OCR projected invalid layout: Q.x must be < P.x")
    if not projected["Z"][0] < projected["M"][0]:
        raise RuntimeError("OCR projected invalid layout: Z.x must be < M.x")
    if not projected["Q"][1] < projected["Z"][1]:
        raise RuntimeError("OCR projected invalid layout: Q.y must be < Z.y")
    if not projected["P"][1] < projected["M"][1]:
        raise RuntimeError("OCR projected invalid layout: P.y must be < M.y")

    enter_detections = [det for det in detections if det.label == "ENTER"]
    if enter_detections:
        direct = max(enter_detections, key=lambda det: det.score)
        pitch = _projected_pitch(projected)
        expected_center = _direct_enter_center_from_label(H, direct)
        dist = float(np.linalg.norm(np.array(projected["ENTER"]) -
                                    np.array(expected_center)))
        if pitch > 1.0 and dist > ENTER_DIRECT_VALIDATION_MAX_PITCH * pitch:
            raise RuntimeError(
                "OCR projected ENTER far from directly detected ENTER label: "
                f"projected={projected['ENTER']}, expected={expected_center}, "
                f"direct={direct.center}, dist={dist:.1f}"
            )

    if requested_keys is not None:
        direct_enter_available = ENTER_DIRECT_CORRECTION and bool(enter_detections)
        skip_keys = {"ENTER"} if direct_enter_available else set()
        _validate_requested_key_bounds(
            projected,
            requested_keys,
            width,
            height,
            skip_keys=skip_keys,
        )


def _validate_requested_key_bounds(
    projected,
    requested_keys: list[str],
    width: int,
    height: int,
    skip_keys: set[str] | None = None,
) -> None:
    skip_keys = skip_keys or set()
    for key in requested_keys:
        if key in skip_keys or key not in projected:
            continue
        x, y = projected[key]
        if not (-REQUESTED_KEY_MARGIN_PX <= x < width + REQUESTED_KEY_MARGIN_PX and
                -REQUESTED_KEY_MARGIN_PX <= y < height + REQUESTED_KEY_MARGIN_PX):
            raise RuntimeError(
                f"OCR projected requested key {key} out of bounds: {(x, y)}"
            )


def _apply_direct_enter_correction(
    fit: FitResult,
    detections: list[Detection],
    requested_keys: list[str],
    log_prefix="[ocr]",
) -> None:
    if not ENTER_DIRECT_CORRECTION or "ENTER" not in requested_keys:
        return
    enter_detections = [det for det in detections if det.label == "ENTER"]
    if not enter_detections:
        return

    direct = max(enter_detections, key=lambda det: det.score)
    corrected = _direct_enter_center_from_label(fit.H, direct)
    old = np.array(fit.projected["ENTER"], dtype=np.float64)
    new = np.array(corrected, dtype=np.float64)
    pitch = _projected_pitch(fit.projected)
    shift_px = float(np.linalg.norm(new - old))
    fit.projected["ENTER"] = corrected
    shift_pitch = shift_px / pitch if pitch > 1.0 else 0.0
    vprint(
        f"{log_prefix} ENTER direct correction: "
        f"shift={shift_px:.1f}px ({shift_pitch:.2f} pitch), "
        f"score={direct.score:.2f}"
    )


def _to_normalized_1000(point, width: int, height: int) -> tuple[float, float]:
    x, y = point
    return x / width * 1000.0, y / height * 1000.0


def _build_output(keys, projected, width: int, height: int):
    missing = [key for key in keys if key not in TEMPLATE]
    if missing:
        raise ValueError(f"OCR route does not support keys: {missing}")

    letters = []
    for key in keys:
        nx, ny = _to_normalized_1000(projected[key], width, height)
        letters.append({"char": key, "x": float(nx), "y": float(ny)})
    return {
        "letters": letters,
        "count": len(letters),
        "coordinate_system": "normalized_1000",
    }


def _save_overlay(image, keys, projected, overlay_out) -> Path:
    """Draw the located keys on the image and save it (overlay_script style)."""
    canvas = image.copy()
    for key in keys:
        x, y = projected[key]
        px, py = int(round(x)), int(round(y))
        cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(canvas, key, (px - 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    overlay_out = Path(overlay_out)
    overlay_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_out), canvas)
    vprint(f"[ocr] overlay saved: {overlay_out}")
    return overlay_out


def _run_ocr_pass(
    image,
    angle: float = 0.0,
    use_cls: bool = False,
    crop: bool = False,
    lighting_variant: str = "none",
    upscale: int = UPSCALE,
    min_score: float = MIN_SCORE,
    text_score=None,
    box_thresh=None,
    log_prefix="[ocr]",
    engine=None,
) -> list[Detection]:
    height, width = image.shape[:2]
    rotated, inverse_matrix = _rotate_for_ocr(image, angle)
    if crop:
        ocr_source, source_offset = _crop_dark_region(rotated)
    else:
        ocr_source, source_offset = rotated, (0.0, 0.0)
    ocr_source = _apply_lighting_variant(ocr_source, lighting_variant)
    rotated_height, rotated_width = ocr_source.shape[:2]
    ocr_image = cv2.resize(
        ocr_source,
        (rotated_width * upscale, rotated_height * upscale),
        interpolation=cv2.INTER_CUBIC,
    )
    engine = engine or _get_engine()
    t0 = time.perf_counter()
    result = engine(
        ocr_image,
        use_det=True,
        use_cls=use_cls,
        use_rec=True,
        return_word_box=False,
        return_single_char_box=False,
        text_score=text_score,
        box_thresh=box_thresh,
    )
    inference_elapsed = time.perf_counter() - t0
    engine_elapsed = getattr(result, "elapse", None)
    angle_part = "" if abs(angle) < 1e-6 else f" angle={angle:+.0f}"
    lighting_part = "" if lighting_variant == "none" else f" light={lighting_variant}"
    if engine_elapsed is None:
        vprint(f"{log_prefix} inference{angle_part}{lighting_part}: "
               f"{inference_elapsed:.2f}s")
    else:
        vprint(f"{log_prefix} inference{angle_part}{lighting_part}: "
               f"{inference_elapsed:.2f}s "
               f"(engine={float(engine_elapsed):.2f}s)")

    return _extract_detections(
        result,
        upscale=upscale,
        min_score=min_score,
        angle=angle,
        inverse_matrix=inverse_matrix,
        source_offset=source_offset,
        image_size=(width, height),
    )


def _fit_rank(fit: FitResult):
    return (
        len(fit.inlier_labels),
        len(fit.inlier_rows),
        len(fit.inlier_labels & ENTER_SUPPORT_LABELS),
        -fit.mean_error_px,
    )


def _strong_pass_plan():
    for variant in STRONG_LIGHTING_VARIANTS:
        yield 0.0, variant


def _strong_fallback_pass_plan():
    seen = {(0.0, variant) for variant in STRONG_LIGHTING_VARIANTS}
    nonzero_angles = [angle for angle in STRONG_ANGLES if abs(angle) >= 1e-6]
    for angle in nonzero_angles:
        for variant in ("none", *STRONG_ROTATED_LIGHTING_VARIANTS):
            item = (angle, variant)
            if item not in seen:
                seen.add(item)
                yield item


def _accept_strong_fit(
    fit: FitResult | None,
    min_inliers: int = STRONG_EARLY_ACCEPT_INLIERS,
    min_rows: int = STRONG_EARLY_ACCEPT_ROWS,
    max_error_px: float = RANSAC_REPROJ_THRESHOLD_PX,
) -> bool:
    return (
        fit is not None and
        len(fit.inlier_labels) >= min_inliers and
        len(fit.inlier_rows) >= min_rows and
        fit.mean_error_px <= max_error_px
    )


def _strong_fast_accepts(fit: FitResult | None) -> bool:
    return _accept_strong_fit(
        fit,
        min_inliers=STRONG_FAST_ACCEPT_INLIERS,
        min_rows=STRONG_FAST_ACCEPT_ROWS,
        max_error_px=STRONG_FAST_ACCEPT_MAX_ERROR_PX,
    )


def _update_strong_fit_candidates(
    detections: list[Detection],
    all_detections: list[Detection],
    width: int,
    height: int,
    keys: list[str],
    best_fit: FitResult | None,
    best_detection_count: int,
) -> tuple[FitResult | None, int]:
    for candidate_detections in (detections, all_detections):
        fit = _try_fit_keyboard_homography(
            candidate_detections,
            width=width,
            height=height,
            min_unique_inliers=STRONG_MIN_UNIQUE_INLIERS,
            label_offset_mm=STRONG_LABEL_OFFSET_MM,
            requested_keys=keys,
        )
        if fit is None:
            continue
        if best_fit is None or _fit_rank(fit) > _fit_rank(best_fit):
            best_fit = fit
            best_detection_count = len(candidate_detections)
    return best_fit, best_detection_count


def _locate_standard(image, width: int, height: int, keys: list[str], engine=None):
    detections = _run_ocr_pass(image, engine=engine)
    if detections:
        vprint(f"[ocr] detected labels: {_candidate_summary(detections)}")
    else:
        vprint("[ocr] detected labels: none")

    fit = _fit_keyboard_homography(
        detections,
        width=width,
        height=height,
        requested_keys=keys,
    )
    return detections, fit


def _locate_strong(image, width: int, height: int, keys: list[str], engine=None,
                   deadline_s: float | None = None):
    all_detections: list[Detection] = []
    best_fit: FitResult | None = None
    best_detection_count = 0
    deadline = _deadline_from_seconds(deadline_s)
    deadline_hit = False
    fast_accepted = False

    if deadline is not None:
        vprint(f"[ocr-strong] deadline: {deadline_s:.2f}s")
    vprint("[ocr-strong] fast path: zero-rotation lighting sweep")
    for angle, lighting_variant in _strong_pass_plan():
        if not _can_start_ocr_pass(deadline):
            deadline_hit = True
            vprint(f"[ocr-strong] deadline before fast pass: "
                   f"{_deadline_status(deadline)}")
            break
        detections = _run_ocr_pass(
            image,
            angle=angle,
            crop=True,
            lighting_variant=lighting_variant,
            upscale=STRONG_UPSCALE,
            min_score=STRONG_MIN_SCORE,
            text_score=STRONG_TEXT_SCORE,
            box_thresh=STRONG_BOX_THRESH,
            log_prefix="[ocr-strong]",
            engine=engine,
        )
        all_detections.extend(detections)
        summary = _candidate_summary(detections) or "none"
        light_part = "" if lighting_variant == "none" else f" light={lighting_variant}"
        vprint(f"[ocr-strong] angle={angle:+.0f}{light_part} labels: {summary}")

        best_fit, best_detection_count = _update_strong_fit_candidates(
            detections,
            all_detections,
            width,
            height,
            keys,
            best_fit,
            best_detection_count,
        )

        if _strong_fast_accepts(best_fit):
            vprint("[ocr-strong] fast path accepted")
            fast_accepted = True
            break

    if not fast_accepted:
        fast_unique_letters = {
            det.label for det in all_detections if det.label in LETTER_KEYS
        }
        if best_fit is not None:
            vprint(
                "[ocr-strong] fast path candidate rejected: "
                f"inliers={len(best_fit.inlier_labels)}, "
                f"rows={len(best_fit.inlier_rows)}, "
                f"err={best_fit.mean_error_px:.1f}px"
            )
        else:
            vprint("[ocr-strong] fast path found no valid fit")

        if deadline_hit:
            vprint("[ocr-strong] fallback skipped: deadline reached")
        elif (best_fit is None and
              len(fast_unique_letters) >= STRONG_FAST_SKIP_FALLBACK_UNIQUE_LABELS):
            vprint(
                "[ocr-strong] fallback skipped: "
                f"{len(fast_unique_letters)} upright labels but no safe fit"
            )
        else:
            if len(fast_unique_letters) <= STRONG_FALLBACK_LOW_LABEL_COUNT:
                max_fallback_passes = STRONG_FALLBACK_MAX_PASSES_LOW_LABELS
            else:
                max_fallback_passes = STRONG_FALLBACK_MAX_PASSES
            vprint(
                "[ocr-strong] fallback: rotation/lighting sweep "
                f"(max_passes={max_fallback_passes})"
            )
            fallback_passes = 0
            for angle, lighting_variant in _strong_fallback_pass_plan():
                if max_fallback_passes > 0 and fallback_passes >= max_fallback_passes:
                    vprint("[ocr-strong] fallback pass budget exhausted")
                    break
                if not _can_start_ocr_pass(deadline):
                    deadline_hit = True
                    vprint(f"[ocr-strong] deadline before fallback pass: "
                           f"{_deadline_status(deadline)}")
                    break
                fallback_passes += 1
                detections = _run_ocr_pass(
                    image,
                    angle=angle,
                    crop=True,
                    lighting_variant=lighting_variant,
                    upscale=STRONG_UPSCALE,
                    min_score=STRONG_MIN_SCORE,
                    text_score=STRONG_TEXT_SCORE,
                    box_thresh=STRONG_BOX_THRESH,
                    log_prefix="[ocr-strong]",
                    engine=engine,
                )
                all_detections.extend(detections)
                summary = _candidate_summary(detections) or "none"
                light_part = "" if lighting_variant == "none" else f" light={lighting_variant}"
                vprint(f"[ocr-strong] angle={angle:+.0f}{light_part} labels: {summary}")

                best_fit, best_detection_count = _update_strong_fit_candidates(
                    detections,
                    all_detections,
                    width,
                    height,
                    keys,
                    best_fit,
                    best_detection_count,
                )

                if _accept_strong_fit(best_fit):
                    break

    if deadline_hit and best_fit is not None:
        vprint("[ocr-strong] deadline reached; using best valid fit so far")

    if best_fit is None:
        unique_letters = {
            det.label for det in all_detections if det.label in LETTER_KEYS
        }
        raise RuntimeError(
            "OCR strong mode could not fit keyboard homography: "
            f"detected labels={_candidate_summary(all_detections) or 'none'}, "
            f"inliers={len(unique_letters)}, missing requested keys={keys}"
        )

    dx, dy = best_fit.label_offset_mm
    vprint(
        "[ocr-strong] homography fit: "
        f"{best_fit.fit_elapsed_s:.3f}s, "
        f"inliers={len(best_fit.inlier_labels)}, "
        f"rows={len(best_fit.inlier_rows)}, "
        f"err={best_fit.mean_error_px:.1f}px, "
        f"detections={best_detection_count}"
    )
    vprint(
        "[ocr-strong] legend offset: "
        f"dx={dx / KEY_PITCH_MM:.2f} pitch, "
        f"dy={dy / KEY_PITCH_MM:.2f} pitch"
    )
    vprint(f"[ocr-strong] combined labels: {_candidate_summary(all_detections) or 'none'}")
    return all_detections, best_fit


def run(image_path, keys, keys_json=KEYS_JSON, target_size=TARGET_SIZE,
        strong: bool = False, overlay_out=None, verbose: bool = False, engine=None,
        deadline_s: float | None = None):
    """Return normalized key centers for the requested keys using local OCR.

    `engine` is an optional pre-initialized RapidOCR instance. Pass one in to
    avoid paying the engine init cost inside the timed pipeline; if None it is
    lazily created (and cached) on first use.
    """
    global VERBOSE
    VERBOSE = verbose

    keys = [str(key).upper() for key in keys]
    image = _read_image(image_path, target_size=target_size)
    height, width = image.shape[:2]

    if strong:
        detections, fit = _locate_strong(image, width=width, height=height, keys=keys,
                                         engine=engine, deadline_s=deadline_s)
        _apply_direct_enter_correction(fit, detections, keys, log_prefix="[ocr-strong]")
    else:
        detections, fit = _locate_standard(image, width=width, height=height, keys=keys,
                                           engine=engine)
        _apply_direct_enter_correction(fit, detections, keys, log_prefix="[ocr]")

    _validate_requested_key_bounds(fit.projected, keys, width, height)

    data = _build_output(keys, fit.projected, width=width, height=height)

    returned = {entry["char"] for entry in data["letters"]}
    missing_requested = [key for key in keys if key not in returned]
    if missing_requested:
        raise RuntimeError(
            "OCR did not return all requested keys: "
            f"detected labels={_candidate_summary(detections) or 'none'}, "
            f"inliers={len(fit.inlier_labels)}, missing requested keys={missing_requested}"
        )

    keys_json = Path(keys_json)
    keys_json.parent.mkdir(parents=True, exist_ok=True)
    keys_json.write_text(json.dumps(data, indent=2))

    if overlay_out is not None:
        _save_overlay(image, keys, fit.projected, overlay_out)
    return data


def _main():
    parser = argparse.ArgumentParser(description="Locate keyboard keys with local OCR.")
    parser.add_argument("image")
    parser.add_argument("keys", nargs="+")
    parser.add_argument("--keys-json", default=str(KEYS_JSON))
    parser.add_argument("--strong", action="store_true",
                        help="Use angle-sweep OCR and legend-to-center correction.")
    parser.add_argument("--overlay-out", default=None,
                        help="Output path for an overlay image of the located keys.")
    parser.add_argument("--deadline-s", type=float, default=None,
                        help="Stop OCR passes after this many seconds and use the best valid fit.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress and timing information.")
    args = parser.parse_args()
    data = run(args.image, args.keys, keys_json=args.keys_json, strong=args.strong,
               overlay_out=args.overlay_out, verbose=args.verbose,
               deadline_s=args.deadline_s)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    _main()
