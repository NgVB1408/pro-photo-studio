# PyInstaller spec — bundle 4 CLI thành 1 wincei-stack.exe
#
# Usage:
#   uv pip install pyinstaller
#   cd C:/Users/kulam/pro-photo-studio
#   pyinstaller packages/wincei-api/wincei_stack.spec
#
# Output: dist/wincei-stack.exe (~2-3 GB do bundle SegFormer weights)
#
# NOTE: Bundle BIG vì kéo torch + transformers + SegFormer-B3 (~1.2GB).
#       Khách non-tech kéo lần đầu mất thời gian nhưng sau đó chạy offline.

import sys
from pathlib import Path

# Workspace root
WS = Path(SPECPATH).parent.parent  # packages/wincei-api/.. = pro-photo-studio/

# Bundle HuggingFace cache + sentencepiece
hf_cache = Path.home() / ".cache" / "huggingface"

block_cipher = None


a = Analysis(
    [str(WS / "packages" / "wincei-api" / "pps_wincei_api" / "__main__.py")],
    pathex=[str(WS)],
    binaries=[],
    datas=[
        # Bundle HF weights nếu đã download (skip nếu không tồn tại)
        *(
            [(str(hf_cache / "hub"), ".cache/huggingface/hub")]
            if (hf_cache / "hub").exists() else []
        ),
    ],
    hiddenimports=[
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "fastapi",
        "starlette",
        "pps_wincei",
        "pps_wincei_hdr",
        "pps_wincei_masks",
        "pps_wincei_masks.semantic",
        "pps_wincei_masks.refine",
        "pps_wincei_masks.molding",
        "pps_wincei_masks.evaluator",
        "pps_wincei_masks.regions_json",
        "pps_wincei_masks.precision",
        "pps_wincei_masks.tta",
        "pps_wincei_masks.crf",
        "pps_wincei_masks.vlm_client",
        "pps_wincei_masks.sam_engine",
        "pps_wincei_masks.vlm_sam_pipeline",
        "transformers.models.segformer",
        "transformers.models.segformer.modeling_segformer",
        "transformers.models.segformer.image_processing_segformer",
        "pymatting",
        "pymatting.alpha.estimate_alpha_cf",
        "tifffile",
        "PIL._tkinter_finder",
        "cv2",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "tkinter",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="wincei-stack",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
