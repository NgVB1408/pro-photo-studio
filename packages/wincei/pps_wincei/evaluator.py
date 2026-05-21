"""AI self-evaluator — chấm điểm output sau pipeline.

7 metrics + overall verdict (pass / review / fail). Toàn bộ scorer là deterministic,
chạy nhanh, KHÔNG cần inference model riêng — reuse segmentation result đã có.

Metric đánh giá:
    1. window_highlight_recovery  — vùng cửa sổ blown (>0.95 luma) đã giảm chưa
    2. window_no_halo             — không có ringing/halo viền cửa sổ
    3. ceiling_neutrality_lab     — A,B trong vùng ceiling đã gần 128 chưa
    4. ceiling_luminance_uniform  — độ đồng đều brightness của ceiling
    5. global_consistency_psnr    — toàn ảnh không bị over-process (vùng KHÔNG mask)
    6. edge_preservation_ssim     — biên window/ceiling không bị smudge
    7. natural_look_chroma_balance — saturation toàn ảnh hợp lý
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import SegmentationResult


@dataclass
class SelfEvaluation:
    verdict: str  # "pass" | "review" | "fail" | "scope_violation" | "no_target"
    overall_score: float  # 0..1
    scope_delta_e: float = 0.0  # mean LAB ΔE on UNTOUCHED region; must be < 1.0
    scope_max_delta_e: float = 0.0  # p99 ΔE; must be < 3.0
    scope_ok: bool = True
    scores: dict[str, float] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def _ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Single-channel SSIM (Wang 2004)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    K1, K2, L = 0.01, 0.03, 255.0
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2

    win = (11, 11)
    mu_a = cv2.GaussianBlur(a, win, 1.5)
    mu_b = cv2.GaussianBlur(b, win, 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a * a, win, 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, win, 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, win, 1.5) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = num / (den + 1e-12)
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    if mse < 1e-9:
        return 60.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _score_window_highlight_recovery(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> tuple[float, str]:
    if mask.sum() == 0:
        return 1.0, "Không có vùng cửa sổ — bỏ qua"
    m = mask > 64
    lum_b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)[m].astype(np.float32) / 255.0
    lum_a = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)[m].astype(np.float32) / 255.0
    clipped_b = float((lum_b > 0.96).mean())
    clipped_a = float((lum_a > 0.96).mean())
    if clipped_b < 1e-3:
        return 1.0, "Cửa sổ không có blown highlight để hồi"
    reduction = max(0.0, (clipped_b - clipped_a) / clipped_b)
    score = float(np.clip(reduction, 0.0, 1.0))
    msg = f"clip {clipped_b * 100:.1f}% → {clipped_a * 100:.1f}% (giảm {reduction * 100:.0f}%)"
    return score, msg


def _score_window_no_halo(after: np.ndarray, mask: np.ndarray) -> tuple[float, str]:
    if mask.sum() == 0:
        return 1.0, "Không có cửa sổ"
    gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=4)
    high = gray - low

    # Edge band JUST OUTSIDE window mask (within 8px)
    m_in = (mask > 128).astype(np.uint8)
    dilated = cv2.dilate(m_in, np.ones((9, 9), np.uint8))
    edge_band = (dilated > 0) & (m_in == 0)
    if not edge_band.any():
        return 1.0, "Không có biên cửa sổ"
    overshoot = float(np.percentile(np.abs(high[edge_band]), 95))
    # < 12 = perfect, > 30 = bad
    score = float(np.clip(1.0 - max(0.0, overshoot - 12) * 0.04, 0.0, 1.0))
    msg = f"edge overshoot p95={overshoot:.1f}"
    return score, msg


def _score_ceiling_neutrality_lab(after: np.ndarray, mask: np.ndarray) -> tuple[float, str]:
    if mask.sum() == 0:
        return 1.0, "Không có vùng trần"
    m = mask > 64
    lab = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_dev = abs(lab[..., 1][m].mean() - 128.0)
    b_dev = abs(lab[..., 2][m].mean() - 128.0)
    cast = float(a_dev + b_dev)
    # < 4 = excellent, > 14 = strong cast
    score = float(np.clip(1.0 - cast / 18.0, 0.0, 1.0))
    msg = f"|ΔA|+|ΔB|={cast:.1f}"
    return score, msg


def _score_ceiling_luminance_uniform(after: np.ndarray, mask: np.ndarray) -> tuple[float, str]:
    if mask.sum() == 0:
        return 1.0, "Không có vùng trần"
    m = mask > 64
    lab = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0][m]
    if L.size < 10:
        return 1.0, "Vùng trần quá nhỏ"
    std = float(L.std())
    # std < 8 = uniform, > 28 = uneven
    score = float(np.clip(1.0 - max(0.0, std - 8) / 20.0, 0.0, 1.0))
    msg = f"L std={std:.1f}"
    return score, msg


def _score_global_consistency_psnr(before: np.ndarray, after: np.ndarray, untouched_mask: np.ndarray) -> tuple[float, str]:
    if untouched_mask.sum() == 0:
        return 1.0, "Cả ảnh đã chỉnh sửa"
    m_3 = cv2.cvtColor(untouched_mask, cv2.COLOR_GRAY2BGR) > 64
    diff = (before.astype(np.float64) - after.astype(np.float64)) ** 2
    diff = diff[m_3]
    if diff.size < 100:
        return 1.0, "Vùng không touched quá nhỏ"
    mse = float(diff.mean())
    if mse < 1e-3:
        psnr = 60.0
    else:
        psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
    # PSNR > 40 = perfect (no change outside mask), < 28 = bad bleed
    score = float(np.clip((psnr - 28.0) / 14.0, 0.0, 1.0))
    msg = f"PSNR untouched={psnr:.1f}dB"
    return score, msg


def _score_edge_preservation_ssim(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> tuple[float, str]:
    """SSIM in the EDGE BAND of mask — should remain high (no smudge)."""
    if mask.sum() == 0:
        return 1.0, "Không có biên"
    m_in = (mask > 128).astype(np.uint8)
    dilated = cv2.dilate(m_in, np.ones((11, 11), np.uint8))
    eroded = cv2.erode(m_in, np.ones((5, 5), np.uint8))
    band = (dilated > 0) & (eroded == 0)
    if not band.any():
        return 1.0, "Biên quá hẹp"
    b_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    a_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    # SSIM on a small crop containing band
    ys, xs = np.where(band)
    y0, y1 = max(0, ys.min() - 8), min(b_gray.shape[0], ys.max() + 8)
    x0, x1 = max(0, xs.min() - 8), min(b_gray.shape[1], xs.max() + 8)
    score = _ssim_simple(b_gray[y0:y1, x0:x1], a_gray[y0:y1, x0:x1])
    msg = f"SSIM biên={score:.3f}"
    return score, msg


def _score_natural_look_chroma_balance(after: np.ndarray) -> tuple[float, str]:
    """Saturation distribution should not be cartoonish."""
    hsv = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0
    mean_s = float(S.mean())
    p99 = float(np.percentile(S, 99))
    # mean ~ 0.18..0.42 natural; mean > 0.55 oversaturated
    if mean_s < 0.05:
        score = 0.6  # dull / desaturated
        msg = f"saturation thấp (mean={mean_s:.2f})"
    elif mean_s > 0.55:
        score = float(np.clip(1.0 - (mean_s - 0.55) * 4, 0.3, 0.9))
        msg = f"saturation cao (mean={mean_s:.2f})"
    else:
        score = float(np.clip(1.0 - abs(mean_s - 0.3) * 1.5, 0.7, 1.0))
        msg = f"saturation OK (mean={mean_s:.2f}, p99={p99:.2f})"
    return score, msg


# ── Public API ────────────────────────────────────────────────────────────────


WEIGHTS = {
    "window_highlight_recovery": 0.20,
    "window_no_halo": 0.10,
    "ceiling_neutrality_lab": 0.20,
    "ceiling_luminance_uniform": 0.10,
    "global_consistency_psnr": 0.20,
    "edge_preservation_ssim": 0.10,
    "natural_look_chroma_balance": 0.10,
}


def _measure_scope_violation(
    before: np.ndarray, after: np.ndarray, edit_mask: np.ndarray
) -> tuple[float, float, bool]:
    """ΔE LAB trên vùng KHÔNG được phép edit (wall/floor/furniture).

    Real-estate scope rule:
        mean ΔE  < 1.0  → imperceptible (PASS)
        mean ΔE  < 2.0  → barely visible (WARN)
        max ΔE  >= 3.0  → visible bleed (FAIL — scope violation)

    Returns (mean_dE, p99_dE, ok).
    """
    untouched = edit_mask < 32  # hard threshold: ANY mask presence excludes
    if not untouched.any():
        return 0.0, 0.0, True
    lab_b = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_a = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
    de = np.sqrt(
        (lab_b[..., 0] - lab_a[..., 0]) ** 2
        + (lab_b[..., 1] - lab_a[..., 1]) ** 2
        + (lab_b[..., 2] - lab_a[..., 2]) ** 2
    )
    untouched_de = de[untouched]
    mean_de = float(untouched_de.mean())
    p99_de = float(np.percentile(untouched_de, 99))
    ok = mean_de < 2.0 and p99_de < 5.0
    return mean_de, p99_de, ok


def evaluate(
    before: np.ndarray,
    after: np.ndarray,
    window_mask: np.ndarray,
    ceiling_mask: np.ndarray,
    *,
    seg: SegmentationResult | None = None,
) -> SelfEvaluation:
    """Tự chấm điểm output. Trả về SelfEvaluation."""

    edit_mask = cv2.bitwise_or(window_mask, ceiling_mask)
    untouched = 255 - edit_mask

    # SCOPE CHECK — wall/floor/furniture MUST NOT change (real-estate constraint)
    scope_mean_de, scope_p99_de, scope_ok = _measure_scope_violation(before, after, edit_mask)

    scorers = [
        ("window_highlight_recovery", lambda: _score_window_highlight_recovery(before, after, window_mask)),
        ("window_no_halo", lambda: _score_window_no_halo(after, window_mask)),
        ("ceiling_neutrality_lab", lambda: _score_ceiling_neutrality_lab(after, ceiling_mask)),
        ("ceiling_luminance_uniform", lambda: _score_ceiling_luminance_uniform(after, ceiling_mask)),
        ("global_consistency_psnr", lambda: _score_global_consistency_psnr(before, after, untouched)),
        ("edge_preservation_ssim", lambda: _score_edge_preservation_ssim(before, after, cv2.bitwise_or(window_mask, ceiling_mask))),
        ("natural_look_chroma_balance", lambda: _score_natural_look_chroma_balance(after)),
    ]

    scores: dict[str, float] = {}
    findings: list[str] = []
    for name, fn in scorers:
        try:
            score, msg = fn()
        except Exception as exc:
            score, msg = 0.5, f"scorer error: {exc}"
        scores[name] = float(score)
        if score < 0.6:
            findings.append(f"⚠ {name}: {msg}")
        elif score < 0.85:
            findings.append(f"• {name}: {msg}")
        else:
            findings.append(f"✓ {name}: {msg}")

    overall = float(sum(scores[k] * WEIGHTS[k] for k in scores))

    # No-target check
    no_window = window_mask.sum() == 0
    no_ceiling = ceiling_mask.sum() == 0
    if no_window and no_ceiling:
        verdict = "no_target"
        findings.insert(
            0,
            "⚠ AI không nhận diện được vùng cửa sổ HOẶC trần — ảnh có thể là exterior, "
            "synthetic, hoặc model cần upgrade (thử GPU + SegFormer-B3).",
        )
    elif not scope_ok:
        # SCOPE VIOLATION trumps overall score — real-estate constraint hard rule
        verdict = "scope_violation"
        findings.insert(
            0,
            f"🚫 SCOPE VIOLATION — tool đã edit ra ngoài vùng cửa sổ/trần. "
            f"ΔE wall/floor mean={scope_mean_de:.2f} p99={scope_p99_de:.2f} (giới hạn 2.0/5.0). "
            f"Output KHÔNG nên giao khách.",
        )
    elif overall >= 0.85:
        verdict = "pass"
    elif overall >= 0.65:
        verdict = "review"
    else:
        verdict = "fail"

    recommendations = _make_recommendations(scores)
    if verdict == "no_target":
        recommendations = [
            "Mask rỗng — thử: 1) ảnh interior thật (không synthetic); "
            "2) GPU + model B3 (`PPS_FORCE_CPU=0` + VRAM ≥6GB); "
            "3) `--include-lamps` nếu chỉ thấy đèn không thấy trần.",
        ]
    elif verdict == "scope_violation":
        recommendations = [
            "Mask đang bleed sang wall/floor. Sửa: "
            "1) Re-run với --no-clip + giảm strength 50%; "
            "2) Kiểm tra mask trong --debug — nếu mask sai, ảnh đó không phù hợp với SegFormer-B0, "
            "thử GPU SegFormer-B3; "
            "3) Tắt --include-lamps nếu lamp tô ra ngoài.",
        ]

    return SelfEvaluation(
        verdict=verdict,
        overall_score=overall,
        scope_delta_e=scope_mean_de,
        scope_max_delta_e=scope_p99_de,
        scope_ok=scope_ok,
        scores=scores,
        findings=findings,
        recommendations=recommendations,
    )


def _make_recommendations(scores: dict[str, float]) -> list[str]:
    """Heuristic param suggestions based on weakest scores."""
    recs: list[str] = []
    if scores.get("window_highlight_recovery", 1.0) < 0.6:
        recs.append("Tăng --window 1.2 hoặc đổi sang model birefnet-general để bắt vùng cửa rộng hơn.")
    if scores.get("window_no_halo", 1.0) < 0.7:
        recs.append("Tăng guide_radius khi gọi fix_window_highlights (đang 24, thử 36-48).")
    if scores.get("ceiling_neutrality_lab", 1.0) < 0.6:
        recs.append("Tăng --ceiling 1.0 (đang 0.85) để CAT kéo về D65 mạnh hơn.")
    if scores.get("ceiling_luminance_uniform", 1.0) < 0.6:
        recs.append("Bật luminance_equalize hoặc kiểm tra mask có tràn sang wall không.")
    if scores.get("global_consistency_psnr", 1.0) < 0.7:
        recs.append("Mask bị tràn ra ngoài ceiling/window — thử --model birefnet-general để mask sạch hơn.")
    if scores.get("edge_preservation_ssim", 1.0) < 0.75:
        recs.append("Biên smudge — giảm strength hoặc tăng guide_radius cho cả 2 fixer.")
    if scores.get("natural_look_chroma_balance", 1.0) < 0.7:
        recs.append("Saturation off — kiểm tra chroma_recover (default 1.15, thử 1.0 nếu ảnh trông unnatural).")
    if not recs:
        recs.append("Tất cả metric đều ổn — output sẵn sàng gửi khách.")
    return recs
