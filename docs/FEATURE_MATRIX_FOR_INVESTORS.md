# Feature Matrix — PPS v2 vs Vendors

**One-page positioning:** chúng tôi không cố beat AutoEnhance/Manuka ở
color enhance (table stakes — ai cũng làm được). Wedge là **3 feature
KHÔNG vendor nào có**.

---

## Đối sánh đầy đủ

Legend: ✅ Có · ⚠ Có nhưng kém · ❌ Không có · 🔜 Đã engineer Phase 3

| Tính năng | AutoEnhance | Manuka | **PPS v2** | Status |
|---|:---:|:---:|:---:|---|
| **Color enhance baseline** | ✅ | ✅ | ✅ | Done — comparable, không vượt trội |
| White balance auto | ✅ | ✅ | ✅ | Done |
| HDR fusion (multi-bracket) | ❌ | ✅ | ✅ + deghost + LAB norm | **Better** |
| Sky replacement (procedural) | ✅ | ✅ | ✅ | Done |
| Window pull (highlight recovery) | ✅ | ✅ | ✅ | Done |
| Lawn enhancement | ✅ | ✅ | ✅ | Done |
| Perspective upright | ✅ | ✅ | ✅ Hough VP | Done |
| Lens distortion correction | ❌ | ⚠ | ✅ Brown-Conrady | **Better** |
| Auto privacy (face/plate blur) | ❌ | ❌ | ✅ + GDPR audit log | **Net-new** |
| TV blackout / fire fireplace | ❌ | ❌ | ✅ | **Net-new** |
| Photographer reflection removal | ❌ | ❌ | ✅ | **Net-new** |
| Twilight transform (Day → Sunset) | ❌ ($add-on) | ✅ | ✅ + diffusion refine | Done |
| Tone coherency (batch-wide LAB) | ❌ | ❌ | ✅ batch anchor | **Net-new** |
| **Virtual Staging** (empty → furnished) | ❌ | ❌ | 🔜 SD3.5 + IPAdapter | **Net-new** |
| **Multi-angle Synthesis** (1 ảnh → N góc) | ❌ | ❌ | 🔜 Qwen-Edit-2509 | **Net-new** |
| **Instruction Editing** (NLP) | ❌ | ❌ | 🔜 Qwen-Image-Lightning | **Net-new** |
| AI upscale 4K (SOTA 2025) | ✅ unspec | ❌ | 🔜 SUPIR + ncnn fallback | **Better** |
| Object removal (click-mask) | ❌ | ❌ | 🔜 SAM 2 + LaMa | **Net-new** |
| Public REST API | ✅ | ✅ | ✅ + GraphQL + webhooks | Done |
| Deterministic seeded execution | ❌ | ❌ | ✅ blake2b per-stage seed | **Net-new** |
| White-label B2B portal | ❌ | ❌ | 🔜 subdomain + custom branding | **Net-new** |
| GPU autoscale (RunPod serverless) | unclear | unclear | 🔜 0→N theo queue | **Better** |
| GDPR delete-on-request | unclear | unclear | 🔜 audit-logged | **Better** |

---

## TLDR cho NĐT

**Họ thắng:** color enhance baseline (subtle pixel-level tuning, năm kinh nghiệm).
Chúng tôi không định cạnh tranh ở đây — table stakes.

**Chúng tôi thắng:** 3 chức năng AI thế hệ mới mà họ chưa làm:
1. **Virtual Staging** — bán thêm $50-150/ảnh (tỷ lệ conversion gấp 3 vs phòng trống)
2. **Multi-angle Synthesis** — listing đa góc từ 1 ảnh duy nhất ($30-80/ảnh)
3. **Instruction Editing** — UX tự nhiên, không cần 20 toggle ($10-30/edit)

→ Mỗi feature là 1 doanh thu line **mới**, không cannibalize color enhance hiện có.

---

## So sánh giá hiện tại (tham khảo)

| | AutoEnhance | Manuka | PPS v2 (planned) |
|---|---|---|---|
| Color enhance | $0.50-1.50/ảnh | $1-3/ảnh (human) | $0.30/ảnh |
| HDR fusion | n/a | $2-5/set | $0.50/set |
| Twilight | $1-3/ảnh add-on | included | $1/ảnh |
| **Virtual Staging** | — | — | **$25-50/ảnh** |
| **Multi-angle** | — | — | **$15-40/ảnh** |
| **Instruction edit** | — | — | **$5-20/edit** |

→ ARPU PPS v2 ≈ $80-200 / listing (10 ảnh) vs $15-30 / listing competitor.

---

## Verify

Toàn bộ feature ở cột "Done" reproducible:
```bash
git clone git@github.com:NgVB1408/pro-photo-studio.git
cd pro-photo-studio && uv sync
uv run pytest packages/core/tests packages/api/tests   # 242 pass
uv run pps-api && open http://localhost:8000/docs
```

Feature ở cột 🔜 (Phase 3) — xem `MANUAL_ML_DEMO_GUIDE.md` để tự chạy demo
trên Colab cho pitch deck.
