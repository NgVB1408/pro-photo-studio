"""VLM client — query Ollama (local) hoặc remote VLM cho click points.

Ollama default endpoint: http://localhost:11434/api/generate
Models recommended:
    - qwen2.5-vl:7b   (4.7GB, balance speed/quality)
    - llama3.2-vision:11b (7.1GB, smartest)
    - moondream:1.8b  (1.7GB, fast nhưng kém)

Output structured JSON:
    {
        "ceiling": [x, y],
        "windows": [[x1, y1], [x2, y2], ...],
        "walls": [x, y],
        "floor": [x, y],
        "doors": [[x, y], ...],
        "molding": [[x, y], ...]
    }
    where (x, y) are pixel coords trong ảnh native (NOT normalized).
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np
import requests

log = logging.getLogger(__name__)


DEFAULT_PROMPT = """You are a real estate vision expert specializing in interior architecture segmentation. Analyze this photo and identify navigation points for each architectural element.

Return ONLY valid JSON, no prose. Coordinates in pixels (x=column, y=row). Use ARRAYS of points for multi-region elements.

Required schema:
{
  "ceiling": [[x1, y1], [x2, y2], [x3, y3]],
  "windows": [[x, y], ...],
  "walls": [[x, y], ...],
  "floor": [[x, y], [x, y]],
  "doors": [[x, y], ...],
  "crown_molding": [[x, y], ...],
  "baseboard": [[x, y], ...]
}

CRITICAL RULES:

CEILING DETECTION:
- Return 3-5 points spread horizontally across the visible ceiling area.
- For modern architecture WITHOUT crown molding: locate ceiling boundary using
  (1) Recessed downlights / lampholders embedded in ceiling
  (2) Drywall ceiling box edges (where ceiling drops)
  (3) The vertical line where wall paint ENDS at the top
  (4) Track lighting or hanging fixtures mounted to ceiling
- DO NOT rely on crown molding heuristic for modern interiors.
- Place points near each downlight + 1-2 points in plain ceiling areas to cover full extent.

WALLS:
- Return 2-3 points per visible wall plane (corner-adjacent walls counted separately).
- Avoid placing points on wall-mounted objects (TVs, paintings, switches).

FLOOR:
- Return 2-3 points spread across visible floor (avoid sofa/furniture footprint).

WINDOWS:
- ONE click point per individual GLASS PANE (not per window unit). 4-pane window = 4 points.
- Center of each glass surface, NOT on the frame/mullion.

DOORS:
- Separate from windows. ONE point per door (centered on glass for glass doors).

CROWN_MOLDING / BASEBOARD:
- Only return if visibly present as a protruding architectural strip.
- If modern flat ceiling/baseboard with no protrusion, return EMPTY array [].

Skip elements smaller than 1% of image area. Be precise — coordinates feed directly into SAM 2 segmentation."""


@dataclass
class VLMResponse:
    raw_text: str
    parsed_points: dict[str, list]
    model: str
    elapsed_ms: float


@dataclass
class ChainOfThoughtResponse:
    reasoning: str               # Step 1 text analysis
    raw_text: str                # Step 2 JSON raw
    parsed_points: dict[str, list]
    model: str
    elapsed_ms: float            # total 2-step time
    target_class: str


class OllamaVLM:
    """HTTP client cho Ollama VLM. Hỗ trợ 2 endpoints:
        - /api/generate   (single-turn, format=json mode)
        - /api/chat       (chat-style, dùng với custom Modelfile như bds-brain)
    """

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: float = 120.0,
        use_chat_api: bool = False,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.use_chat_api = use_chat_api

    def _image_to_base64(self, image_bgr: np.ndarray, max_side: int = 1280) -> str:
        h, w = image_bgr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img = image_bgr
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Cannot encode image")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def query_chain_of_thought(
        self,
        image_bgr: np.ndarray,
        max_side: int = 1280,
        target_class: str = "ceiling",
    ) -> "ChainOfThoughtResponse":
        """2-step CoT prompting cho ảnh khó.

        Step 1: VLM mô tả TRỰC QUAN vị trí + đặc điểm class target.
        Step 2: VLM trả COORDS dựa trên Step 1 reasoning (passed as context).

        Args:
            image_bgr: full-res BGR.
            max_side: VLM input scale.
            target_class: 'ceiling' | 'walls' | 'floor' | 'windows' | 'doors'.

        Returns:
            ChainOfThoughtResponse (reasoning + parsed_points).
        """
        import time

        b64 = self._image_to_base64(image_bgr, max_side=max_side)
        t0 = time.perf_counter()

        # === STEP 1: Visual reasoning DEEP (cho ảnh khó) ===
        step1_prompt = f"""Phân tích ảnh BĐS độ khó cao này. Bằng tiếng Việt, mô tả:

A. NGUỒN SÁNG + ĐỔ BÓNG:
   1. Hướng nguồn sáng chính (cửa sổ trái/phải/sau lưng camera)?
   2. Liệt kê vùng bị shadow đè (trần ám tối, góc khuất sàn, etc.)?
   3. Có hot-spot blown highlight không (cửa kính/đèn)?

B. CẤU TRÚC {target_class.upper()}:
   4. Vị trí trong ảnh (top/center/bottom × left/right)?
   5. Đặc điểm nhận diện: phào, đèn âm trần, vân sàn, viền kính, drywall edge?
   6. Trần có giật cấp (multi-tier dropped ceiling)? Đếm số tầng nếu có.
   7. Bị che bởi vật gì (nội thất, rèm, đồ trang trí)?

C. RANH GIỚI GIẢ (false boundaries) — TRÁNH:
   8. Vết phản chiếu kính cửa sổ trên tường/sàn?
   9. Bóng đồ nội thất đổ lên trần?
   10. Đường nẹp trang trí KHÔNG phải mép thật của {target_class}?

D. PERSPECTIVE:
   11. Vanishing points (1-point / 2-point / 3-point)?
   12. Camera angle (eye-level / low-angle dưới lên / high-angle trên xuống)?

Trả văn bản tự nhiên, KHÔNG JSON. Ngắn gọn 80-150 từ."""

        s1_payload = self._build_payload(step1_prompt, b64, force_json=False)
        s1_endpoint = self._resolve_endpoint()
        try:
            r1 = requests.post(s1_endpoint, json=s1_payload, timeout=self.timeout)
            r1.raise_for_status()
            d1 = r1.json()
            reasoning = d1.get("message", {}).get("content") or d1.get("response", "")
        except Exception as exc:
            raise RuntimeError(f"CoT Step 1 fail: {exc}") from exc

        # === STEP 2: Coords based on Step 1 (multi-tier aware) ===
        step2_prompt = f"""Dựa trên phân tích sau của BẠN:

\"\"\"
{reasoning[:1500]}
\"\"\"

Bây giờ trả DUY NHẤT JSON với tọa độ pixel (x = cột, y = hàng) cho SAM 2.

Yêu cầu CRITICAL:
- Rải ÍT NHẤT 5 điểm trên bề mặt {target_class} thực tế.
- Nếu trần GIẬT CẤP (multi-tier): rải điểm vào TỪNG TẦNG khác nhau (≥1 điểm/tầng).
- TRÁNH các "ranh giới giả" mà bạn đã liệt kê ở câu C (vết phản chiếu, bóng đồ vật,
  nẹp trang trí không phải mép thật).
- TRÁNH vùng bị shadow đè nặng hoặc blown highlight (kết quả SAM kém ở đó).
- Cách đều ngang qua bề mặt visible.
- Tọa độ theo ảnh gốc (full resolution).

Format BẮT BUỘC: {{"points": [[x1, y1], [x2, y2], ...]}}
KHÔNG văn bản, KHÔNG comment, chỉ JSON."""

        s2_payload = self._build_payload(step2_prompt, b64, force_json=True)
        try:
            r2 = requests.post(s2_endpoint := self._resolve_endpoint(), json=s2_payload, timeout=self.timeout)
            r2.raise_for_status()
            d2 = r2.json()
            raw = d2.get("message", {}).get("content") or d2.get("response", "")
        except Exception as exc:
            raise RuntimeError(f"CoT Step 2 fail: {exc}") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000
        parsed = self._parse_points(raw, image_bgr.shape[:2], max_side=max_side)

        return ChainOfThoughtResponse(
            reasoning=reasoning,
            raw_text=raw,
            parsed_points=parsed,
            model=self.model,
            elapsed_ms=elapsed_ms,
            target_class=target_class,
        )

    def _resolve_endpoint(self) -> str:
        if self.use_chat_api or self.endpoint.endswith("/api/chat"):
            return self.endpoint if self.endpoint.endswith("/api/chat") else self.endpoint.replace("/api/generate", "/api/chat")
        return self.endpoint

    def _build_payload(self, prompt: str, b64_image: str, *, force_json: bool) -> dict:
        endpoint = self._resolve_endpoint()
        if endpoint.endswith("/api/chat"):
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt, "images": [b64_image]}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1024},
            }
        else:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1024},
            }
        if force_json:
            payload["format"] = "json"
            payload["options"]["num_predict"] = 512
        return payload

    def query(
        self,
        image_bgr: np.ndarray,
        prompt: str = DEFAULT_PROMPT,
        max_side: int = 1280,
    ) -> VLMResponse:
        """Send image + prompt → parse JSON response."""
        import time

        b64 = self._image_to_base64(image_bgr, max_side=max_side)

        if self.use_chat_api or self.endpoint.endswith("/api/chat"):
            # Chat-style (works tốt với custom Modelfile có SYSTEM prompt như bds-brain)
            chat_endpoint = self.endpoint
            if not chat_endpoint.endswith("/api/chat"):
                chat_endpoint = chat_endpoint.replace("/api/generate", "/api/chat")
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt, "images": [b64]}
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 512},
            }
            actual_endpoint = chat_endpoint
        else:
            # Single-turn /api/generate
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 1024},
            }
            actual_endpoint = self.endpoint

        t0 = time.perf_counter()
        try:
            r = requests.post(actual_endpoint, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            # /api/chat trả {"message": {"content": "..."}}, /api/generate trả {"response": "..."}
            if "message" in data:
                raw = data["message"].get("content", "")
            else:
                raw = data.get("response", "")
        except Exception as exc:
            raise RuntimeError(f"Ollama request fail ({actual_endpoint}): {exc}") from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Parse JSON từ response (handle both {"points": [...]} và {ceiling: [...], walls: [...]} schemas)
        parsed = self._parse_points(raw, image_bgr.shape[:2], max_side=max_side)

        return VLMResponse(
            raw_text=raw,
            parsed_points=parsed,
            model=self.model,
            elapsed_ms=elapsed_ms,
        )

    def _parse_points(self, text: str, original_hw: tuple[int, int], max_side: int) -> dict:
        """Extract JSON points + rescale từ resized → original coords."""
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # Try extract first {...} block
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                log.warning("VLM response không có JSON: %s", text[:200])
                return {}
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                log.warning("Parse JSON fail: %s", exc)
                return {}

        h, w = original_hw
        # If image was resized for VLM, scale coords back
        scale = max(1.0, max(h, w) / max_side)

        def _rescale_point(pt):
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                return None
            try:
                x, y = float(pt[0]) * scale, float(pt[1]) * scale
            except (TypeError, ValueError):
                return None
            x = max(0, min(w - 1, int(x)))
            y = max(0, min(h - 1, int(y)))
            return [x, y]

        out: dict[str, list] = {}

        # Schema 1: bds-brain → {"points": [[x,y], ...]} — assume ceiling
        if "points" in obj and isinstance(obj["points"], list):
            pts = [_rescale_point(p) for p in obj["points"]]
            pts = [p for p in pts if p is not None]
            if pts:
                out["ceiling"] = pts
                return out

        # Schema 2: full schema từ DEFAULT_PROMPT
        for key, val in obj.items():
            if isinstance(val, list) and val and isinstance(val[0], (int, float)):
                rp = _rescale_point(val)
                if rp:
                    out[key] = rp
            elif isinstance(val, list):
                points = [_rescale_point(p) for p in val]
                points = [p for p in points if p is not None]
                if points:
                    out[key] = points
        return out


def check_ollama_available(endpoint: str = "http://localhost:11434") -> tuple[bool, list[str]]:
    """Check Ollama running + list available models. Returns (ok, model_names)."""
    try:
        r = requests.get(f"{endpoint}/api/tags", timeout=3)
        r.raise_for_status()
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        return True, models
    except Exception:
        return False, []
