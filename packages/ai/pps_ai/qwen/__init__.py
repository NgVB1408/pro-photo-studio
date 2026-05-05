"""Qwen-Image-Lightning instruction-based editing.

Wraps `lightx2v/Qwen-Image-Lightning` (LoRA on top of `Qwen/Qwen-Image`) for
fast natural-language image edits.

Usage:
    from pps_ai.qwen import QwenEditor

    editor = QwenEditor()  # lazy-loads on first call
    out = editor.edit(image_bgr, "brighten the kitchen, remove the photographer")
"""

from __future__ import annotations

from .editor import QwenEditor

__all__ = ["QwenEditor"]
