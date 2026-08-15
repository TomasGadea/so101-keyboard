from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from runtime.geometry import order_quad_points


WARP_WIDTH = 900
WARP_HEIGHT = 300
LETTER_ROWS = {
    "q_row": "QWERTYUIOP",
    "a_row": "ASDFGHJKL",
    "z_row": "ZXCVBNM",
}


@dataclass
class FittedRow:
    keys: str
    x0: float
    dx: float
    y: float
    matched: int
    residual: float

    def point_for(self, index: int) -> np.ndarray:
        return np.array([self.x0 + index * self.dx, self.y], dtype=np.float32)


def _warp_keyboard(image_rgb: np.ndarray, quad_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = order_quad_points(quad_px).astype(np.float32)
    dst = np.array(
        [[0.0, 0.0], [WARP_WIDTH - 1.0, 0.0], [WARP_WIDTH - 1.0, WARP_HEIGHT - 1.0], [0.0, WARP_HEIGHT - 1.0]],
        dtype=np.float32,
    )
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_to_warp = cv2.getPerspectiveTransform(q, dst)
    warp_bgr = cv2.warpPerspective(image_bgr, image_to_warp, (WARP_WIDTH, WARP_HEIGHT))
    warp_to_image = cv2.getPerspectiveTransform(dst, q)
    return cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2RGB), warp_to_image


def _legend_components(warp_rgb: np.ndarray) -> np.ndarray:
    warp_bgr = cv2.cvtColor(warp_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
    mask = ((gray > 95) & (hsv[:, :, 1] < 115)).astype(np.uint8) * 255
    mask[:35, :] = 0
    mask[245:, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    points: list[tuple[float, float]] = []
    for idx in range(1, count):
        _x, _y, width, height, area = stats[idx]
        cx, cy = centroids[idx]
        if 2 <= width <= 25 and 2 <= height <= 20 and 3 <= area <= 120 and 35 < cy < 245:
            points.append((float(cx), float(cy)))
    return np.asarray(points, dtype=np.float32)


def _group_by_y(points: np.ndarray, gap_px: float = 14.0) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    ordered = points[np.argsort(points[:, 1])]
    groups: list[list[np.ndarray]] = []
    for point in ordered:
        if not groups or point[1] - groups[-1][-1][1] > gap_px:
            groups.append([point])
        else:
            groups[-1].append(point)
    rows = [np.asarray(group, dtype=np.float32) for group in groups if len(group) >= 8]
    return sorted(rows, key=lambda row: float(np.mean(row[:, 1])))


def _x_clusters(row_points: np.ndarray, gap_px: float = 19.0) -> np.ndarray:
    xs = sorted(float(x) for x in row_points[:, 0])
    groups: list[list[float]] = []
    for x in xs:
        if not groups or x - groups[-1][-1] > gap_px:
            groups.append([x])
        else:
            groups[-1].append(x)
    centers = [float(np.mean(group)) for group in groups]
    return np.asarray(centers, dtype=np.float32)


def _row_origin_prior(xs: np.ndarray) -> tuple[float | None, float | None]:
    xs = np.asarray(sorted(float(x) for x in xs), dtype=np.float32)
    if len(xs) < 4:
        return None, None

    diffs = np.diff(xs)
    key_gaps = diffs[(25.0 <= diffs) & (diffs <= 65.0)]
    if len(key_gaps) == 0:
        return float(xs[0]), None

    dx = float(np.median(key_gaps))
    x0 = float(xs[0])
    # Q/A/Z rows usually have Tab/Caps/Shift just to their left. When that
    # modifier is detected, the first gap is visibly larger than a key pitch;
    # the second cluster is the first letter.
    if len(xs) >= 2 and float(xs[1] - xs[0]) > 1.08 * dx:
        x0 = float(xs[1])
    return x0, dx


def _fit_regular_row(
    xs: np.ndarray,
    y: float,
    keys: str,
    *,
    expected_x0: float | None = None,
    expected_dx: float | None = None,
) -> FittedRow | None:
    xs = np.asarray(sorted(float(x) for x in xs), dtype=np.float32)
    n = len(keys)
    if len(xs) < max(4, min(n, 6)):
        return None

    best: tuple[float, FittedRow] | None = None
    for left_idx, left_x in enumerate(xs):
        for right_idx in range(left_idx + 1, len(xs)):
            right_x = float(xs[right_idx])
            for left_key_idx in range(n):
                for right_key_idx in range(left_key_idx + 1, n):
                    dx = (right_x - float(left_x)) / float(right_key_idx - left_key_idx)
                    if not 25.0 <= dx <= 65.0:
                        continue
                    if expected_dx is not None and not 0.75 * expected_dx <= dx <= 1.22 * expected_dx:
                        continue
                    x0 = float(left_x) - left_key_idx * dx
                    if expected_x0 is not None and abs(x0 - expected_x0) > 1.6 * max(dx, expected_dx or dx):
                        continue
                    predicted = np.array([x0 + i * dx for i in range(n)], dtype=np.float32)
                    tolerance = max(13.0, 0.34 * dx)
                    residuals = []
                    matched = 0
                    for px in predicted:
                        nearest = float(np.min(np.abs(xs - px)))
                        if nearest <= tolerance:
                            matched += 1
                            residuals.append(nearest)
                    if matched < max(5, n - 2):
                        continue
                    residual = float(np.mean(residuals)) if residuals else 999.0
                    edge_penalty = 0.0
                    if predicted[0] < 0:
                        edge_penalty += abs(float(predicted[0])) / dx
                    if predicted[-1] > WARP_WIDTH - 1:
                        edge_penalty += (float(predicted[-1]) - (WARP_WIDTH - 1)) / dx
                    prior_penalty = 0.0
                    if expected_x0 is not None:
                        prior_penalty += 45.0 * abs(x0 - expected_x0) / dx
                    if expected_dx is not None:
                        prior_penalty += 14.0 * abs(dx - expected_dx) / dx
                    score = matched * 20.0 - residual - 8.0 * edge_penalty - prior_penalty
                    row = FittedRow(keys=keys, x0=x0, dx=dx, y=float(y), matched=matched, residual=residual)
                    if best is None or score > best[0]:
                        best = (score, row)
    return None if best is None else best[1]


def _select_letter_row_groups(row_groups: list[np.ndarray]) -> list[np.ndarray] | None:
    if len(row_groups) < 3:
        return None
    # Usually rows are: number, Q, A, Z, bottom modifiers. Pick the three
    # letter rows by looking for three consecutive rows with plausible spacing.
    best: tuple[float, int] | None = None
    centers = [float(np.mean(row[:, 1])) for row in row_groups]
    for start in range(0, len(row_groups) - 2):
        y0, y1, y2 = centers[start : start + 3]
        gaps = np.array([y1 - y0, y2 - y1], dtype=np.float32)
        if np.any(gaps < 20.0) or np.any(gaps > 55.0):
            continue
        count_score = sum(min(len(row_groups[start + offset]), 28) for offset in range(3))
        spacing_score = -float(abs(gaps[0] - gaps[1]))
        # Prefer the Q/A/Z triplet over number/Q/A when both are plausible.
        prior = 10.0 if start == 1 else 0.0
        score = count_score + spacing_score + prior
        if best is None or score > best[0]:
            best = (score, start)
    if best is None:
        return None
    start = best[1]
    return row_groups[start : start + 3]


def fit_qwerty_key_pixels(image_rgb: np.ndarray, layout_quad_px: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    warp_rgb, warp_to_image = _warp_keyboard(image_rgb, layout_quad_px)
    components = _legend_components(warp_rgb)
    row_groups = _group_by_y(components)
    letter_groups = _select_letter_row_groups(row_groups)
    if letter_groups is None:
        raise RuntimeError("Could not find enough key legend rows.")

    fitted_rows: dict[str, FittedRow] = {}
    a_points = letter_groups[1]
    a_xs = _x_clusters(a_points)
    a_x0_prior, a_dx_prior = _row_origin_prior(a_xs)
    a_row = _fit_regular_row(
        a_xs,
        float(np.mean(a_points[:, 1])),
        LETTER_ROWS["a_row"],
        expected_x0=a_x0_prior,
        expected_dx=a_dx_prior,
    )
    if a_row is None:
        raise RuntimeError("Could not fit a_row from detected key legends.")
    fitted_rows["a_row"] = a_row

    for row_name, row_points in (("q_row", letter_groups[0]), ("z_row", letter_groups[2])):
        keys = LETTER_ROWS[row_name]
        xs = _x_clusters(row_points)
        x0_prior, dx_prior = _row_origin_prior(xs)
        fitted = _fit_regular_row(
            xs,
            float(np.mean(row_points[:, 1])),
            keys,
            expected_x0=x0_prior,
            expected_dx=dx_prior or a_row.dx,
        )
        if fitted is None:
            raise RuntimeError(f"Could not fit {row_name} from detected key legends.")
        fitted_rows[row_name] = fitted

    warp_points: dict[str, np.ndarray] = {}
    for row in fitted_rows.values():
        for index, key in enumerate(row.keys):
            warp_points[key] = row.point_for(index)

    z_row = fitted_rows["z_row"]
    a_row = fitted_rows["a_row"]
    row_gap = z_row.y - a_row.y
    bottom_y = z_row.y + 1.05 * row_gap
    if len(row_groups) >= 5:
        lower_rows = [row for row in row_groups if float(np.mean(row[:, 1])) > z_row.y + 12.0]
        if lower_rows:
            bottom_y = float(np.mean(lower_rows[0][:, 1]))
    warp_points["SPACE"] = np.array([z_row.x0 + 4.55 * z_row.dx, bottom_y], dtype=np.float32)
    warp_points["ENTER"] = np.array([a_row.x0 + 11.2 * a_row.dx, a_row.y], dtype=np.float32)

    names = list(warp_points.keys())
    pts = np.array([warp_points[name] for name in names], dtype=np.float32).reshape(-1, 1, 2)
    image_pts = cv2.perspectiveTransform(pts, warp_to_image).reshape(-1, 2)
    key_pixels = {name: image_pts[idx].astype(np.float64) for idx, name in enumerate(names)}
    details = {
        "method": "keycap_fit",
        "component_count": int(len(components)),
        "row_centers_warp": [float(np.mean(row[:, 1])) for row in row_groups],
        "fitted_rows": {
            name: {
                "keys": row.keys,
                "x0": row.x0,
                "dx": row.dx,
                "y": row.y,
                "matched": row.matched,
                "residual": row.residual,
            }
            for name, row in fitted_rows.items()
        },
    }
    return key_pixels, details
