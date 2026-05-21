"""Real Estate Vision Analyzer — JSON bounding box output.

Spec (theo mày yêu cầu):
    Input  : 1 ảnh BĐS
    Output : JSON {project_type, detected_elements: {ceiling, windows, ...}}
    Coords : normalized [ymin, xmin, ymax, xmax] thang đo [0, 1000]

Chain-of-thought segmentation:
    Step 1. Identify camera angle (wide / close / bird's eye) via vanishing point analysis.
    Step 2. Detect perspective lines → locate ceiling/floor boundaries.
    Step 3. SegFormer ADE20K semantic masks:
                ceiling=5, wall=0, floor=3, window=8, door=14, curtain=18, sky=2
    Step 4. Subtract curtain region khỏi window mask → giữ kính + khung.
    Step 5. Tight ceiling boundary tới mép phào chỉ (crown molding edge) nếu có.
    Step 6. Instance separation cho windows: connected components + filter.
    Step 7. Classify window type theo aspect ratio + position.
    Step 8. Compute confidence = mean softmax prob trong mask.

Output format:
    {
      "project_type": "Real Estate Segmentation",
      "image_size": {"width": 4608, "height": 3072},
      "camera_angle": "wide_angle",
      "detected_elements": {
        "ceiling": {"coordinates": [0,0,320,1000], "confidence": 0.94, "has_crown_molding": false},
        "windows": [
          {"id": 1, "coordinates": [250,120,680,450], "type": "Casement", "confidence": 0.91}
        ],
        "walls": {"coordinates": [320,0,820,1000], "confidence": 0.88, "area_pct": 51.7},
        "floor": {"coordinates": [820,0,1000,1000], "confidence": 0.96}
      },
      "reasoning_steps": ["...", "..."]
    }
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

ADE_CLS = {
    "wall": 0,
    "sky": 2,
    "floor": 3,
    "ceiling": 5,
    "window": 8,
    "door": 14,
    "curtain": 18,
    "lamp": 36,
    "light": 82,
}


@dataclass
class BBox:
    """Normalized [ymin, xmin, ymax, xmax] on [0, 1000] scale."""

    ymin: int
    xmin: int
    ymax: int
    xmax: int

    def to_list(self) -> list[int]:
        return [self.ymin, self.xmin, self.ymax, self.xmax]


def _bbox_from_mask(mask: np.ndarray, *, h: int, w: int) -> Optional[BBox]:
    """Find tight bbox of mask, return normalized [0..1000]."""
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return BBox(
        ymin=int(round(y0 / max(1, h - 1) * 1000)),
        xmin=int(round(x0 / max(1, w - 1) * 1000)),
        ymax=int(round(y1 / max(1, h - 1) * 1000)),
        xmax=int(round(x1 / max(1, w - 1) * 1000)),
    )


def _detect_camera_angle(image_bgr: np.ndarray, ceiling_mask: np.ndarray, floor_mask: np.ndarray) -> str:
    """Phân loại góc chụp theo tỉ lệ ceiling/floor visible."""
    h, w = image_bgr.shape[:2]
    ceil_cov = ceiling_mask.sum() / (h * w)
    floor_cov = floor_mask.sum() / (h * w)

    if ceil_cov > 0.35 and floor_cov > 0.25:
        return "wide_angle"  # rộng, thấy cả trần và sàn
    if floor_cov > 0.4:
        return "bird_eye"  # chụp xuống
    if ceil_cov < 0.05 and floor_cov < 0.1:
        return "close_up"  # cận
    return "standard"


def _classify_window_type(bbox: BBox, image_aspect: float) -> str:
    """Heuristic phân loại type theo aspect ratio."""
    h_norm = bbox.ymax - bbox.ymin
    w_norm = bbox.xmax - bbox.xmin
    if h_norm == 0 or w_norm == 0:
        return "Unknown"
    aspect = w_norm / h_norm
    if aspect > 1.8:
        return "Sliding Window"  # ngang dài
    if aspect < 0.55:
        return "Casement Window"  # đứng cao
    if 0.85 <= aspect <= 1.15:
        return "Picture Window"  # vuông
    if 0.55 <= aspect <= 0.85:
        return "Casement Window"  # cao hơn rộng
    return "Standard Window"


def _detect_crown_molding(image_bgr: np.ndarray, ceiling_mask: np.ndarray, wall_mask: np.ndarray) -> bool:
    """Detect phào trần (crown molding) ở seam wall|ceiling.
    Return True nếu có edge mạnh ngang sát ranh giới.
    """
    c = (ceiling_mask > 0).astype(np.uint8)
    w_ = (wall_mask > 0).astype(np.uint8)
    if c.sum() == 0 or w_.sum() == 0:
        return False
    c_dil = cv2.dilate(c, np.ones((5, 5), np.uint8), iterations=1)
    w_dil = cv2.dilate(w_, np.ones((5, 5), np.uint8), iterations=1)
    seam = c_dil & w_dil
    if seam.sum() < 100:
        return False
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_in_seam = cv2.bitwise_and(edges, (seam * 255).astype(np.uint8))
    # Detect horizontal lines in seam
    try:
        lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_ADV)
        lines = lsd.detect(gray)[0]
        if lines is None:
            return edges_in_seam.sum() > 500
        n_horizontal_in_seam = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length < 80:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 15 or angle > 165:
                cy = int((y1 + y2) / 2)
                cx = int((x1 + x2) / 2)
                if 0 <= cy < seam.shape[0] and 0 <= cx < seam.shape[1] and seam[cy, cx] > 0:
                    n_horizontal_in_seam += 1
        return n_horizontal_in_seam >= 2
    except (AttributeError, cv2.error):
        return edges_in_seam.sum() > 500


def _split_window_panes(
    window_mask: np.ndarray,
    *,
    min_area_pct: float = 0.001,
    max_panes: int = 12,
) -> list[np.ndarray]:
    """Connected components → individual pane masks.

    Filter:
        - Skip components < min_area_pct của ảnh.
        - Cap số pane về max_panes (lớn nhất area).
    """
    binary = (window_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = window_mask.shape
    min_area = h * w * min_area_pct

    panes_with_area = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        panes_with_area.append((area, (labels == i).astype(np.uint8)))

    panes_with_area.sort(reverse=True, key=lambda t: t[0])
    return [m for _, m in panes_with_area[:max_panes]]


def _mean_confidence(soft_prob: np.ndarray, mask_binary: np.ndarray) -> float:
    """Mean softmax probability inside binary mask."""
    if mask_binary.sum() == 0:
        return 0.0
    inside = soft_prob[mask_binary > 0]
    return float(inside.mean())


def analyze_image_to_json(
    image_bgr: np.ndarray,
    *,
    segmenter=None,
    include_walls: bool = True,
    include_floor: bool = True,
    include_doors: bool = True,
) -> dict[str, Any]:
    """MAIN ENTRY — analyze 1 ảnh BĐS, trả JSON dict.

    Args:
        image_bgr: (H,W,3) uint8 BGR.
        segmenter: pre-loaded SemanticSegmenter (None → tự load).
        include_walls/floor/doors: extras để xuất trong JSON.

    Returns:
        dict matching schema spec.
    """
    if segmenter is None:
        from .semantic import SemanticSegmenter
        segmenter = SemanticSegmenter()

    h, w = image_bgr.shape[:2]
    reasoning: list[str] = []

    # === STEP 1: Semantic segmentation ===
    reasoning.append("Step 1: SegFormer ADE20K inference → 9 class soft prob")
    sem = segmenter.segment(image_bgr)

    soft = {name: sem.get_soft(cls_id) for name, cls_id in ADE_CLS.items()}
    binary = {name: (s >= 0.5).astype(np.uint8) for name, s in soft.items()}

    # === STEP 2: Camera angle ===
    angle = _detect_camera_angle(image_bgr, binary["ceiling"], binary["floor"])
    reasoning.append(f"Step 2: camera_angle = {angle}")

    # === STEP 3: Curtain subtract from window ===
    if binary["curtain"].sum() > 100:
        reasoning.append(
            f"Step 3: subtract curtain ({binary['curtain'].mean()*100:.1f}% cov) khỏi window mask"
        )
        window_clean = binary["window"] & (~binary["curtain"].astype(bool)).astype(np.uint8)
    else:
        window_clean = binary["window"]

    # Union window + sky (sky qua kính thường được model phân loại sky thay vì window)
    window_clean = window_clean | binary["sky"]

    # === STEP 4: Crown molding detect ===
    has_crown = _detect_crown_molding(image_bgr, binary["ceiling"], binary["wall"])
    reasoning.append(f"Step 4: crown_molding = {has_crown}")

    # === STEP 5: Window pane instance separation ===
    panes = _split_window_panes(window_clean)
    reasoning.append(f"Step 5: window panes detected = {len(panes)}")

    # === STEP 6: Build output ===
    out: dict[str, Any] = {
        "project_type": "Real Estate Segmentation",
        "image_size": {"width": int(w), "height": int(h)},
        "camera_angle": angle,
        "detected_elements": {},
        "reasoning_steps": reasoning,
    }

    # Ceiling
    ceil_bbox = _bbox_from_mask(binary["ceiling"], h=h, w=w)
    if ceil_bbox is not None:
        out["detected_elements"]["ceiling"] = {
            "coordinates": ceil_bbox.to_list(),
            "confidence": round(_mean_confidence(soft["ceiling"], binary["ceiling"]), 3),
            "has_crown_molding": has_crown,
            "area_pct": round(binary["ceiling"].mean() * 100, 2),
        }

    # Windows (per pane)
    windows_out: list[dict] = []
    for i, pane in enumerate(panes, 1):
        bbox = _bbox_from_mask(pane, h=h, w=w)
        if bbox is None:
            continue
        wtype = _classify_window_type(bbox, image_aspect=w / h)
        conf = _mean_confidence(soft["window"], pane)
        windows_out.append({
            "id": i,
            "coordinates": bbox.to_list(),
            "type": wtype,
            "confidence": round(conf, 3),
            "area_pct": round(pane.mean() * 100, 2),
        })
    out["detected_elements"]["windows"] = windows_out

    # Walls (optional)
    if include_walls:
        wall_bbox = _bbox_from_mask(binary["wall"], h=h, w=w)
        if wall_bbox is not None:
            out["detected_elements"]["walls"] = {
                "coordinates": wall_bbox.to_list(),
                "confidence": round(_mean_confidence(soft["wall"], binary["wall"]), 3),
                "area_pct": round(binary["wall"].mean() * 100, 2),
            }

    # Floor (optional)
    if include_floor:
        floor_bbox = _bbox_from_mask(binary["floor"], h=h, w=w)
        if floor_bbox is not None:
            out["detected_elements"]["floor"] = {
                "coordinates": floor_bbox.to_list(),
                "confidence": round(_mean_confidence(soft["floor"], binary["floor"]), 3),
                "area_pct": round(binary["floor"].mean() * 100, 2),
            }

    # Doors (optional, separate from windows)
    if include_doors and binary["door"].sum() > 100:
        door_bbox = _bbox_from_mask(binary["door"], h=h, w=w)
        if door_bbox is not None:
            out["detected_elements"]["doors"] = [{
                "id": 1,
                "coordinates": door_bbox.to_list(),
                "type": "Glass Door" if binary["sky"].sum() > 100 else "Standard Door",
                "confidence": round(_mean_confidence(soft["door"], binary["door"]), 3),
                "area_pct": round(binary["door"].mean() * 100, 2),
            }]

    return out


def analyze_file_to_json(image_path: Path | str, **kwargs) -> dict[str, Any]:
    """Convenience: đọc file → analyze."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Không đọc được: {image_path}")
    return analyze_image_to_json(img, **kwargs)
