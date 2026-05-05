from .config import Settings, load_settings
from .detect import auto_mask, detect_bright_logo, detect_edge_anomaly, detect_text_mser
from .dropbox_client import DropboxClient, DropboxError, DropboxFile
from .inpaint import (
    InpaintBackend,
    SUPPORTED_LAMA_MODELS,
    inpaint,
    inpaint_lama,
    inpaint_opencv,
    resolve_device,
)
from .mask import (
    build_mask_from_boxes,
    build_mask_from_color,
    build_mask_from_image,
    combine_masks,
    dilate_mask,
)
from .composite import CompositeReport, composite_from_original
from .enhance import EnhanceParams, PRESETS, enhance, enhance_preset, preset
from .quality import QualityReport, compare, compare_files, watermark_residual
from .realestate import (
    RealEstateReport,
    SceneReport,
    VerticalReport,
    classify_scene,
    correct_vertical,
    detect_blown_windows,
    detect_lawn_mask,
    detect_sky_mask,
    enhance_lawn,
    enhance_realestate_full,
    load_sky_from_path,
    replace_sky,
    window_pull,
)
from .unsplash import UnsplashClient

__all__ = [
    "Settings",
    "load_settings",
    "InpaintBackend",
    "SUPPORTED_LAMA_MODELS",
    "inpaint",
    "inpaint_lama",
    "inpaint_opencv",
    "resolve_device",
    "build_mask_from_boxes",
    "build_mask_from_color",
    "build_mask_from_image",
    "combine_masks",
    "dilate_mask",
    "auto_mask",
    "detect_bright_logo",
    "detect_edge_anomaly",
    "detect_text_mser",
    "CompositeReport",
    "composite_from_original",
    "EnhanceParams",
    "PRESETS",
    "enhance",
    "enhance_preset",
    "preset",
    "QualityReport",
    "compare",
    "compare_files",
    "watermark_residual",
    "RealEstateReport",
    "SceneReport",
    "VerticalReport",
    "classify_scene",
    "correct_vertical",
    "detect_blown_windows",
    "detect_lawn_mask",
    "detect_sky_mask",
    "enhance_lawn",
    "enhance_realestate_full",
    "load_sky_from_path",
    "replace_sky",
    "window_pull",
    "UnsplashClient",
    "DropboxClient",
    "DropboxFile",
    "DropboxError",
]

__version__ = "0.2.0"
