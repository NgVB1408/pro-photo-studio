"""HTML comparison viewer — slider before/after giống realtyedit.app.

Static HTML — không cần server, double-click mở trong browser.
Output: 1 single .html file embed JSON metadata + thumbnails.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from .io_quality import make_thumbnail


@dataclass
class ComparisonItem:
    """One image pair displayed in viewer."""

    name: str
    before_url: str
    after_url: str
    width: int
    height: int
    verdict: str = ""
    score: float = 0.0
    scope_de: float = 0.0
    context_summary: str = ""
    decisions: list[str] = None  # type: ignore[assignment]
    window_clipped_before: float = 0.0
    window_clipped_after: float = 0.0
    ceiling_cast_before: float = 0.0
    ceiling_cast_after: float = 0.0


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0; background: #0f1419; color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  header {{
    padding: 18px 24px; border-bottom: 1px solid #21262d;
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    position: sticky; top: 0; z-index: 50;
  }}
  header h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
  header .sub {{ font-size: 12px; color: #7d8590; margin-top: 4px; }}
  .toolbar {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; align-items: center; }}
  .toolbar button, .toolbar select {{
    background: #21262d; color: #e6edf3; border: 1px solid #30363d;
    padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 13px;
  }}
  .toolbar button.active {{ background: #1f6feb; border-color: #388bfd; }}
  .toolbar button:hover {{ background: #30363d; }}
  .stats {{ font-size: 12px; color: #7d8590; margin-left: auto; }}
  main {{ padding: 24px; max-width: 1600px; margin: 0 auto; }}
  .gallery {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 24px;
  }}
  .card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    overflow: hidden; transition: transform 0.15s;
  }}
  .card:hover {{ transform: translateY(-2px); border-color: #58a6ff; }}
  .compare {{
    position: relative; user-select: none; cursor: ew-resize; overflow: hidden;
    background: #000;
  }}
  .compare img {{ display: block; width: 100%; height: auto; }}
  .compare .after {{
    position: absolute; top: 0; left: 0; clip-path: inset(0 50% 0 0);
  }}
  .compare .slider {{
    position: absolute; top: 0; bottom: 0; left: 50%;
    width: 2px; background: rgba(255,255,255,0.9);
    box-shadow: 0 0 8px rgba(0,0,0,0.5); pointer-events: none;
  }}
  .compare .knob {{
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 36px; height: 36px; border-radius: 50%;
    background: rgba(255,255,255,0.95); border: 2px solid #1f6feb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5); pointer-events: none;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; color: #1f6feb; font-weight: bold;
  }}
  .compare .label {{
    position: absolute; top: 12px; padding: 4px 10px;
    border-radius: 4px; background: rgba(0,0,0,0.7); color: white;
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
  }}
  .compare .label.before {{ left: 12px; }}
  .compare .label.after {{ right: 12px; }}
  .meta {{ padding: 12px 16px; }}
  .meta .name {{ font-size: 13px; font-weight: 600; margin-bottom: 4px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .meta .ctx {{ font-size: 11px; color: #7d8590; margin-bottom: 8px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .badge.pass {{ background: #1a4731; color: #56d364; }}
  .badge.review {{ background: #4a3a17; color: #f0b35a; }}
  .badge.fail, .badge.scope_violation {{ background: #4a1f1f; color: #f85149; }}
  .badge.no_target {{ background: #2c3038; color: #8b949e; }}
  .badge.scope-ok {{ background: #1a4731; color: #56d364; }}
  .badge.scope-bad {{ background: #4a1f1f; color: #f85149; }}
  .badge.score {{ background: #1f2937; color: #58a6ff; }}
  .decisions {{ font-size: 11px; color: #8b949e; margin-top: 8px;
    border-top: 1px dashed #30363d; padding-top: 8px; }}
  .decisions div {{ margin: 2px 0; }}
  .mode-sbs .compare .after {{ clip-path: none !important; left: 50% !important;
    width: 50% !important; }}
  .mode-sbs .compare .before {{ width: 50% !important; }}
  .mode-sbs .compare .slider, .mode-sbs .compare .knob {{ display: none; }}
</style>
</head>
<body>
<header>
  <h1>🪟🏠 pps-wincei — Window + Ceiling AI Comparison</h1>
  <div class="sub">{subtitle}</div>
  <div class="toolbar">
    <button id="mode-slider" class="active">↔️ Slider</button>
    <button id="mode-sbs">⬛⬜ Side-by-side</button>
    <select id="filter">
      <option value="all">Tất cả ({total})</option>
      <option value="pass">✅ Pass ({pass_count})</option>
      <option value="review">⚠️ Review ({review_count})</option>
      <option value="fail">❌ Fail ({fail_count})</option>
      <option value="scope_violation">🚫 Scope violation ({scope_count})</option>
      <option value="no_target">⏭️ No target ({no_target_count})</option>
    </select>
    <span class="stats">{stats}</span>
  </div>
</header>
<main>
  <div class="gallery" id="gallery"></div>
</main>
<script>
const ITEMS = {items_json};

function makeCard(item) {{
  const verdict_cls = (item.verdict || 'review').toLowerCase();
  const decisions_html = (item.decisions || []).map(d => `<div>${{d}}</div>`).join('');
  const score_pct = (item.score * 100).toFixed(0);
  return `
    <div class="card" data-verdict="${{verdict_cls}}">
      <div class="compare">
        <img class="before" src="${{item.before_url}}" alt="before">
        <img class="after" src="${{item.after_url}}" alt="after">
        <div class="label before">BEFORE</div>
        <div class="label after">AFTER</div>
        <div class="slider"></div>
        <div class="knob">⇔</div>
      </div>
      <div class="meta">
        <div class="name">${{item.name}}</div>
        <div class="ctx">${{item.context_summary || ''}}</div>
        <div class="badges">
          <span class="badge ${{verdict_cls}}">${{item.verdict.toUpperCase()}}</span>
          <span class="badge score">${{score_pct}}/100</span>
          <span class="badge ${{item.scope_de < 2.0 ? 'scope-ok' : 'scope-bad'}}">
            🛡️ ΔE ${{item.scope_de.toFixed(2)}}
          </span>
        </div>
        ${{decisions_html ? `<div class="decisions">${{decisions_html}}</div>` : ''}}
      </div>
    </div>
  `;
}}

function render(filter) {{
  const gallery = document.getElementById('gallery');
  const items = filter === 'all' ? ITEMS : ITEMS.filter(i => (i.verdict || '').toLowerCase() === filter);
  gallery.innerHTML = items.map(makeCard).join('');
  attachSliders();
}}

function attachSliders() {{
  document.querySelectorAll('.compare').forEach(el => {{
    const after = el.querySelector('.after');
    const slider = el.querySelector('.slider');
    const knob = el.querySelector('.knob');
    let pressed = false;
    function move(x) {{
      const rect = el.getBoundingClientRect();
      const pct = Math.max(0, Math.min(100, ((x - rect.left) / rect.width) * 100));
      after.style.clipPath = `inset(0 ${{100 - pct}}% 0 0)`;
      slider.style.left = pct + '%';
      knob.style.left = pct + '%';
    }}
    el.addEventListener('mousedown', e => {{ pressed = true; move(e.clientX); }});
    document.addEventListener('mousemove', e => {{ if (pressed) move(e.clientX); }});
    document.addEventListener('mouseup', () => {{ pressed = false; }});
    el.addEventListener('mousemove', e => {{
      // hover also moves (realtyedit.app style)
      if (!pressed) move(e.clientX);
    }});
    el.addEventListener('touchmove', e => {{ if (e.touches[0]) {{ move(e.touches[0].clientX); e.preventDefault(); }} }}, {{ passive: false }});
  }});
}}

document.getElementById('mode-slider').addEventListener('click', () => {{
  document.body.classList.remove('mode-sbs');
  document.getElementById('mode-slider').classList.add('active');
  document.getElementById('mode-sbs').classList.remove('active');
}});
document.getElementById('mode-sbs').addEventListener('click', () => {{
  document.body.classList.add('mode-sbs');
  document.getElementById('mode-sbs').classList.add('active');
  document.getElementById('mode-slider').classList.remove('active');
}});
document.getElementById('filter').addEventListener('change', e => render(e.target.value));

render('all');
</script>
</body>
</html>
"""


def _encode_thumbnail_b64(bgr_thumb) -> str:
    """JPEG-encode thumbnail to base64 data URL."""
    ok, enc = cv2.imencode(".jpg", bgr_thumb, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def generate_html(
    items: list[ComparisonItem],
    out_path: str | Path,
    *,
    title: str = "Window + Ceiling AI Comparison",
    subtitle: str = "",
) -> Path:
    """Render comparison HTML viewer (single file, no server)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"pass": 0, "review": 0, "fail": 0, "scope_violation": 0, "no_target": 0}
    for it in items:
        v = (it.verdict or "review").lower()
        if v in counts:
            counts[v] += 1

    if items:
        avg_score = sum(it.score for it in items) / len(items)
        avg_scope = sum(it.scope_de for it in items) / len(items)
        stats = f"{len(items)} ảnh · ⭐ {avg_score:.2f} · 🛡️ ΔE {avg_scope:.2f}"
    else:
        stats = "0 ảnh"

    items_dict = [
        {
            "name": it.name,
            "before_url": it.before_url,
            "after_url": it.after_url,
            "width": it.width,
            "height": it.height,
            "verdict": it.verdict,
            "score": it.score,
            "scope_de": it.scope_de,
            "context_summary": it.context_summary,
            "decisions": it.decisions or [],
            "window_clipped_before": it.window_clipped_before,
            "window_clipped_after": it.window_clipped_after,
            "ceiling_cast_before": it.ceiling_cast_before,
            "ceiling_cast_after": it.ceiling_cast_after,
        }
        for it in items
    ]

    html = HTML_TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        total=len(items),
        pass_count=counts["pass"],
        review_count=counts["review"],
        fail_count=counts["fail"],
        scope_count=counts["scope_violation"],
        no_target_count=counts["no_target"],
        stats=stats,
        items_json=json.dumps(items_dict, ensure_ascii=False),
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


def build_comparison_item(
    name: str,
    before_bgr,
    after_bgr,
    *,
    process_result,
    embed_thumbnails: bool = True,
    thumb_side: int = 1024,
) -> ComparisonItem:
    """Construct a ComparisonItem from pipeline ProcessResult + image pair."""
    if embed_thumbnails:
        before_thumb = make_thumbnail(before_bgr, max_side=thumb_side)
        after_thumb = make_thumbnail(after_bgr, max_side=thumb_side)
        before_url = _encode_thumbnail_b64(before_thumb)
        after_url = _encode_thumbnail_b64(after_thumb)
    else:
        before_url = ""
        after_url = ""

    return ComparisonItem(
        name=name,
        before_url=before_url,
        after_url=after_url,
        width=process_result.width,
        height=process_result.height,
        verdict=process_result.evaluation.get("verdict", "review"),
        score=process_result.evaluation.get("overall_score", 0.0),
        scope_de=process_result.evaluation.get("scope_delta_e", 0.0),
        context_summary=process_result.context.get("summary", ""),
        decisions=list(process_result.tuning.get("reasoning", [])),
        window_clipped_before=process_result.window.get("clipped_pct_before", 0.0),
        window_clipped_after=process_result.window.get("clipped_pct_after", 0.0),
        ceiling_cast_before=process_result.ceiling.get("cast_magnitude_before", 0.0),
        ceiling_cast_after=process_result.ceiling.get("cast_magnitude_after", 0.0),
    )
