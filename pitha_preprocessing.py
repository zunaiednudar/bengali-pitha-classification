"""
Pitha Preprocessing & Curation Module (Pipeline V2)
---------------------------------------------------
Reusable image preprocessing functions used across:
  - Dataset splitting & standardization (split_and_augment.py)
  - Inference on new camera photos (classify_new_pitha.py)

Key Features:
  1. EXIF Auto-Orientation: Resolves phone camera orientation.
  2. Non-Destructive: Keeps authentic dough and crust food textures intact.
  3. Aspect-Preserving Reflection Padding: Scales and pads images to 224×224
     without stretching or distorting pitha geometry.
  4. Laplacian Blur Quality Audit: Measures edge variance to flag blurry images.
"""

from pathlib import Path
from PIL import Image, ImageOps
import cv2
import numpy as np

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_RESOLUTION = 140
BLUR_THRESHOLD = 20.0


def load_and_orient_image(img_path):
    """
    Loads image using PIL, resolves EXIF camera rotation,
    and converts color space to 3-channel RGB.
    Returns PIL Image or None on failure.
    """
    try:
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            return img
    except Exception:
        return None


def evaluate_quality(pil_img, min_res=MIN_RESOLUTION, blur_threshold=BLUR_THRESHOLD):
    """
    Checks resolution boundaries and computes Laplacian sharpness variance.
    Returns (passes_quality: bool, score: float, status: str).
    """
    w, h = pil_img.size
    if w < min_res or h < min_res:
        return False, 0.0, f"Resolution too low ({w}x{h})"

    np_img = np.array(pil_img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_blurry = variance < blur_threshold
    status = "Blurry" if is_blurry else "Sharp"
    return not is_blurry, variance, status


def resize_with_pad(img_np, size=224):
    """
    Resize an RGB or BGR numpy array preserving aspect ratio,
    then pad to square (size × size).
    Uses BORDER_REFLECT_101 to avoid harsh black borders.
    """
    h, w = img_np.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Cannot resize an empty image of shape {img_np.shape}")

    scale = size / max(h, w)
    # A very elongated photo (e.g. 5000x10) scales its short side down to 0,
    # which makes cv2.resize raise. Clamp to at least one pixel.
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=interp)

    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101
    )
    return padded


def preprocess_for_model(image_path, target_size=224):
    """
    Preprocesses a new single image file for EfficientNetV2:
      1. Loads with EXIF auto-orientation.
      2. Audits sharpness score.
      3. Resizes with aspect-preserving reflection padding to target_size.
    Returns (img_rgb_np: np.ndarray, quality_info: dict) or (None, None).
    """
    pil_img = load_and_orient_image(image_path)
    if pil_img is None:
        return None, None

    passes_quality, score, status = evaluate_quality(pil_img)
    np_rgb = np.array(pil_img)
    model_input = resize_with_pad(np_rgb, size=target_size)

    quality_info = {
        "sharpness_score": score,
        "status": status,
        "original_size": pil_img.size,
    }
    return model_input, quality_info
