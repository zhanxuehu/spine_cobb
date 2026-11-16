"""将输入 X 光对齐到 AASCE 训练分布：灰度、自动极性、对比度归一。"""

from __future__ import annotations

import cv2
import numpy as np


def read_raw(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray if gray.dtype == np.uint8 else _stretch_u8(gray)


def _stretch_u8(gray: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, (p_lo, p_hi))
    if hi <= lo + 1e-6:
        lo, hi = float(g.min()), float(g.max())
    if hi <= lo + 1e-6:
        return np.zeros(g.shape, dtype=np.uint8)
    out = (g - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _should_invert(gray_u8: np.ndarray) -> bool:
    h, w = gray_u8.shape[:2]
    m = max(8, min(h, w) // 10)
    corners = float(
        np.mean(
            [
                gray_u8[:m, :m].mean(),
                gray_u8[:m, -m:].mean(),
                gray_u8[-m:, :m].mean(),
                gray_u8[-m:, -m:].mean(),
            ]
        )
    )
    cy0, cy1 = max(0, h // 2 - m), min(h, h // 2 + m)
    cx0, cx1 = max(0, w // 2 - m), min(w, w // 2 + m)
    center = float(gray_u8[cy0:cy1, cx0:cx1].mean())
    return corners > center + 12.0


def preprocess_xray(
    img: np.ndarray,
    *,
    auto_invert: bool = True,
    force_invert: bool = False,
    stretch: bool = True,
    clahe: bool = True,
    clahe_clip: float = 2.0,
) -> tuple[np.ndarray, dict]:
    gray = _to_gray_u8(img)
    info = {"inverted": False, "clahe": False, "stretch": False}

    if force_invert or (auto_invert and _should_invert(gray)):
        gray = cv2.bitwise_not(gray)
        info["inverted"] = True

    if stretch:
        gray = _stretch_u8(gray, 1.0, 99.0)
        info["stretch"] = True

    if clahe:
        gray = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8)).apply(gray)
        info["clahe"] = True

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr, info


def load_xray_for_infer(
    path: str,
    *,
    auto_invert: bool = True,
    force_invert: bool = False,
    stretch: bool = True,
    clahe: bool = True,
) -> tuple[np.ndarray, dict]:
    raw = read_raw(path)
    return preprocess_xray(
        raw,
        auto_invert=auto_invert,
        force_invert=force_invert,
        stretch=stretch,
        clahe=clahe,
    )
