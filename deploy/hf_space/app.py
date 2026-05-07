"""Gradio demo wrapping `pps_core.realestate.enhance_realestate_full`.

Deploy via `.github/workflows/hf-space-deploy.yml` (manual workflow_dispatch).
Free CPU tier OK; T4 small if `HF_SPACES_BUDGET_USD` allows.
"""

from __future__ import annotations

import logging
import os

import cv2
import gradio as gr
import numpy as np

from pps_core.realestate import enhance_realestate_full

log = logging.getLogger(__name__)

TITLE = "Pro Photo Studio — demo nâng cao ảnh BĐS"
DESCRIPTION = (
    "Tải lên 1 ảnh BĐS (interior / exterior). Pipeline áp WB, CLAHE, "
    "highlight recovery, shadow lift, vibrance, sharpening + sky/lawn nếu phát hiện. "
    "Đây là bản demo CPU — production pipeline đầy đủ chạy qua API."
)


def enhance(image: np.ndarray, sky_preset: str) -> np.ndarray:
    if image is None:
        return image
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out, _ = enhance_realestate_full(bgr, sky_preset=sky_preset)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"## {TITLE}\n\n{DESCRIPTION}")
        with gr.Row():
            inp = gr.Image(type="numpy", label="Ảnh gốc")
            out = gr.Image(type="numpy", label="Ảnh sau xử lý")
        sky = gr.Dropdown(
            choices=[
                "blue_clouds",
                "blue_clear",
                "sunset_warm",
                "golden_hour",
                "overcast_soft",
                "dramatic_storm",
            ],
            value="blue_clouds",
            label="Sky preset (chỉ áp khi phát hiện ngoại thất)",
        )
        btn = gr.Button("Nâng cao ảnh", variant="primary")
        btn.click(fn=enhance, inputs=[inp, sky], outputs=out)
    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
