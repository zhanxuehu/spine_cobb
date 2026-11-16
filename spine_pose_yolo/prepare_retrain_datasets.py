"""
从头重训用的两套数据（不再混用旧权重逻辑）：

1) AASCE → 每张切成最多 17 个单椎 patch + 4 角点（角点老师）
   - vert_corner_patch：旧方块 patch（pad≈0.55）
   - vert_corner_ctx：邻椎上下文矩形 patch（高由邻距定、宽由椎尺度定）
   - vert_corner_ctx_cas：中心模型预测中心裁剪 + GT 四角 + 双椎条带
2) 中心点：
   - vert_center_spinal：仅 Spinal-AI2024
   - vert_center_mix_v2：Spinal + AASCE suitable 过采样

    python spine_pose_yolo/prepare_retrain_datasets.py --skip-patch --skip-ctx-patch --skip-center --build-center-mix
    python spine_pose_yolo/prepare_retrain_datasets.py --skip-patch --skip-ctx-patch --skip-center --build-ctx-cas
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prepare_vert_pose_dataset import (  # noqa: E402
    DEFAULT_AASCE,
    DEFAULT_SPINAL,
    IMG_EXTS,
    normalize_quad,
    parse_challenge_id,
    quad_ok,
    read_wh,
    resolve_spinal_image,
)

DEFAULT_OUT = _HERE / "datasets"

# 邻椎上下文矩形 patch（训练/推理共用）
CTX_W_FACTOR = 1.7
CTX_H_MIN_FACTOR = 2.2
CTX_H_MAX_FACTOR = 3.2
# 弯段定向版：更宽、高度上限略放宽
CTX_CURVE_W_BASE = 2.0
CTX_CURVE_W_MID = 2.2
CTX_CURVE_W_HIGH = 2.4
CTX_CURVE_H_MAX = 3.6
LUMBAR_VI = frozenset(range(12, 17))  # 约 L1–L5


def context_patch_size(
    vert_size: float,
    spacing_up: float,
    spacing_dn: float,
    w_factor: float = CTX_W_FACTOR,
    h_min_factor: float = CTX_H_MIN_FACTOR,
    h_max_factor: float = CTX_H_MAX_FACTOR,
) -> tuple[float, float]:
    """返回 (W, H)。高度由上下邻间距决定，宽度按椎体尺度收窄。"""
    vs = max(16.0, float(vert_size))
    su = max(0.0, float(spacing_up))
    sd = max(0.0, float(spacing_dn))
    if su < 1.0 and sd < 1.0:
        su = sd = vs
    elif su < 1.0:
        su = sd
    elif sd < 1.0:
        sd = su
    h = float(np.clip(su + sd, h_min_factor * vs, h_max_factor * vs))
    w = float(max(24.0, w_factor * vs))
    return w, h


def neighbor_spacings(centers: np.ndarray, i: int) -> tuple[float, float]:
    """到上下邻中心的欧氏距离；缺一侧时返回 0，由 context_patch_size 外推。"""
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    up = dn = 0.0
    if 0 <= i < len(c):
        if i - 1 >= 0:
            up = float(np.linalg.norm(c[i] - c[i - 1]))
        if i + 1 < len(c):
            dn = float(np.linalg.norm(c[i + 1] - c[i]))
    return up, dn


def local_spine_axis_angle_deg(centers: np.ndarray, i: int) -> float:
    """局部脊柱走向相对图像 +y 的夹角（度）；用于旋转对齐弯段。"""
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if len(c) == 0 or i < 0 or i >= len(c):
        return 0.0
    if i - 1 >= 0 and i + 1 < len(c):
        d = c[i + 1] - c[i - 1]
    elif i + 1 < len(c):
        d = c[i + 1] - c[i]
    elif i - 1 >= 0:
        d = c[i] - c[i - 1]
    else:
        return 0.0
    if float(np.linalg.norm(d)) < 1e-3:
        return 0.0
    return float(np.degrees(np.arctan2(float(d[0]), float(d[1]))))


def lateral_offset_ratio(centers: np.ndarray, i: int, vert_size: float) -> float:
    """邻中心横向跨度 / 椎体尺度，大弯处偏大。"""
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    vs = max(1.0, float(vert_size))
    if i - 1 >= 0 and i + 1 < len(c):
        return abs(float(c[i + 1, 0] - c[i - 1, 0])) / vs
    if i + 1 < len(c):
        return abs(float(c[i + 1, 0] - c[i, 0])) / vs
    if i - 1 >= 0:
        return abs(float(c[i, 0] - c[i - 1, 0])) / vs
    return 0.0


def curve_w_factor(centers: np.ndarray, i: int, vert_size: float) -> float:
    lat = lateral_offset_ratio(centers, i, vert_size)
    if lat > 0.35:
        return CTX_CURVE_W_HIGH
    if lat > 0.20:
        return CTX_CURVE_W_MID
    return CTX_CURVE_W_BASE


def approx_cobb_from_pts68(pts: np.ndarray) -> tuple[float, int] | None:
    """AASCE 68 点 → 近似 Cobb 角与顶椎下标（两极限倾角椎中点）。"""
    from infer_vert_pose import approx_main_cobb  # noqa: WPS433

    quads = [normalize_quad(pts[vi * 4 : (vi + 1) * 4]) for vi in range(17)]
    got = approx_main_cobb(quads)
    if got is None:
        return None
    ang, i0, i1 = got
    apex = int(round(0.5 * (i0 + i1)))
    return float(ang), apex


def crop_oriented_context_at_xy(
    bgr: np.ndarray,
    quad: np.ndarray | None,
    cx: float,
    cy: float,
    centers: np.ndarray,
    center_idx: int,
    vert_size: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, float, float] | None:
    """弯段定向上下文裁剪。返回 (crop, quad_in_crop|None, M_inv_2x3, W, H)。

    先绕中心旋转使局部脊柱轴对齐图像 +y，再按自适应宽高裁矩形。
    M_inv 把 crop 坐标系中的点映回原图像素（先 +offset 再仿射）。
    """
    q = None if quad is None else normalize_quad(quad)
    su, sd = neighbor_spacings(centers, center_idx)
    wf = curve_w_factor(centers, center_idx, vert_size)
    w, h = context_patch_size(
        vert_size, su, sd, w_factor=wf, h_min_factor=CTX_H_MIN_FACTOR, h_max_factor=CTX_CURVE_H_MAX
    )
    angle = local_spine_axis_angle_deg(centers, center_idx)
    ih, iw = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), -float(angle), 1.0)
    rotated = cv2.warpAffine(
        bgr,
        M,
        (iw, ih),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    got = crop_rect_at_center(rotated, float(cx), float(cy), w, h)
    if got is None:
        return None
    crop, xa, ya = got
    ch, cw = crop.shape[:2]
    # crop 内点 → 旋转图：(+xa,+ya)；再 Minv → 原图
    M_crop_to_rot = np.array([[1.0, 0.0, float(xa)], [0.0, 1.0, float(ya)]], dtype=np.float64)
    M_inv_rot = cv2.invertAffineTransform(M)
    # 合成：crop → image
    M_full = np.vstack([M_inv_rot, [0.0, 0.0, 1.0]]) @ np.vstack([M_crop_to_rot, [0.0, 0.0, 1.0]])
    M_inv = M_full[:2].astype(np.float64)

    q2 = None
    if q is not None:
        ones = np.ones((4, 1), dtype=np.float64)
        q_rot = (M @ np.hstack([q, ones]).T).T
        q2 = q_rot.copy()
        q2[:, 0] -= float(xa)
        q2[:, 1] -= float(ya)
        if (q2[:, 0] < -0.35 * cw).any() or (q2[:, 1] < -0.35 * ch).any():
            return None
        if (q2[:, 0] > cw + 0.35 * cw).any() or (q2[:, 1] > ch + 0.35 * ch).any():
            return None
        qc = q2.mean(axis=0)
        if qc[0] < -0.05 * cw or qc[0] > 1.05 * cw or qc[1] < -0.05 * ch or qc[1] > 1.05 * ch:
            return None
        q2[:, 0] = np.clip(q2[:, 0], 0, cw - 1)
        q2[:, 1] = np.clip(q2[:, 1], 0, ch - 1)
        if _area(q2) < 8.0:
            return None
    return crop, q2, M_inv, float(w), float(h)


def map_points_with_affine(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Nx2 ← 仿射 2x3。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(p), 1), dtype=np.float64)
    return (M @ np.hstack([p, ones]).T).T


def crop_rect_at_center(
    bgr: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    height: float,
) -> tuple[np.ndarray, int, int] | None:
    """以 (cx,cy) 为中心裁矩形；越界黑边补齐。返回 (crop, xa, ya)。"""
    ih, iw = bgr.shape[:2]
    w = int(max(16, round(float(width))))
    h = int(max(16, round(float(height))))
    xa = int(round(float(cx) - 0.5 * w))
    ya = int(round(float(cy) - 0.5 * h))
    xb = xa + w
    yb = ya + h
    pad_l = max(0, -xa)
    pad_t = max(0, -ya)
    pad_r = max(0, xb - iw)
    pad_b = max(0, yb - ih)
    x_a, y_a = max(0, xa), max(0, ya)
    x_b, y_b = min(iw, xb), min(ih, yb)
    if x_b - x_a < 8 or y_b - y_a < 8:
        return None
    crop = bgr[y_a:y_b, x_a:x_b].copy()
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    ch, cw = crop.shape[:2]
    if cw < 16 or ch < 16:
        return None
    return crop, xa, ya


def _imwrite(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix if path.suffix else ".jpg", bgr)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


def _imread(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return abs(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def crop_vert_patch(
    bgr: np.ndarray,
    quad: np.ndarray,
    pad: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """返回 (crop_bgr, quad_in_crop)。

    以椎体中心为方形 ROI 中心；越出原图边界时用黑边补齐，避免贴边裁切把椎体挤偏。
    """
    q = normalize_quad(quad)
    ih, iw = bgr.shape[:2]
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    bw, bh = max(8.0, x1 - x0), max(8.0, y1 - y0)
    side = int(max(16, round(max(bw, bh) * (1.0 + 2.0 * pad))))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    xa = int(round(cx - 0.5 * side))
    ya = int(round(cy - 0.5 * side))
    xb = xa + side
    yb = ya + side

    pad_l = max(0, -xa)
    pad_t = max(0, -ya)
    pad_r = max(0, xb - iw)
    pad_b = max(0, yb - ih)
    x_a = max(0, xa)
    y_a = max(0, ya)
    x_b = min(iw, xb)
    y_b = min(ih, yb)
    if x_b - x_a < 8 or y_b - y_a < 8:
        return None
    crop = bgr[y_a:y_b, x_a:x_b].copy()
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    q2 = q.copy()
    q2[:, 0] -= float(xa)
    q2[:, 1] -= float(ya)
    ch, cw = crop.shape[:2]
    if cw < 16 or ch < 16:
        return None
    if (q2[:, 0] < -2).any() or (q2[:, 1] < -2).any():
        return None
    if (q2[:, 0] > cw + 2).any() or (q2[:, 1] > ch + 2).any():
        return None
    q2[:, 0] = np.clip(q2[:, 0], 0, cw - 1)
    q2[:, 1] = np.clip(q2[:, 1], 0, ch - 1)
    if _area(q2) < 8.0:
        return None
    return crop, q2


def crop_vert_context_patch(
    bgr: np.ndarray,
    quads: list[np.ndarray],
    vi: int,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """邻椎上下文矩形裁剪。返回 (crop_bgr, quad_in_crop, W, H)。

    只监督当前椎四角；上下邻椎仅作为视觉上下文出现在 patch 中。
    """
    if vi < 0 or vi >= len(quads):
        return None
    q = normalize_quad(quads[vi])
    centers = np.asarray([normalize_quad(qq).mean(axis=0) for qq in quads], dtype=np.float64)
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    bw, bh = max(8.0, float(x1 - x0)), max(8.0, float(y1 - y0))
    vert_size = float(max(bw, bh))
    su, sd = neighbor_spacings(centers, vi)
    w, h = context_patch_size(vert_size, su, sd)
    cx, cy = float(centers[vi, 0]), float(centers[vi, 1])
    got = crop_rect_at_center(bgr, cx, cy, w, h)
    if got is None:
        return None
    crop, xa, ya = got
    q2 = q.copy()
    q2[:, 0] -= float(xa)
    q2[:, 1] -= float(ya)
    ch, cw = crop.shape[:2]
    if (q2[:, 0] < -2).any() or (q2[:, 1] < -2).any():
        return None
    if (q2[:, 0] > cw + 2).any() or (q2[:, 1] > ch + 2).any():
        return None
    q2[:, 0] = np.clip(q2[:, 0], 0, cw - 1)
    q2[:, 1] = np.clip(q2[:, 1], 0, ch - 1)
    if _area(q2) < 8.0:
        return None
    return crop, q2, float(w), float(h)


def patch_label_line(quad: np.ndarray, iw: int, ih: int, margin: float = 0.06) -> str:
    q = normalize_quad(quad)
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    x0 = max(0.0, x0 - bw * margin)
    y0 = max(0.0, y0 - bh * margin)
    x1 = min(iw - 1.0, x1 + bw * margin)
    y1 = min(ih - 1.0, y1 + bh * margin)
    cx = ((x0 + x1) * 0.5) / iw
    cy = ((y0 + y1) * 0.5) / ih
    w = (x1 - x0) / iw
    h = (y1 - y0) / ih
    parts = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]
    for x, y in q:
        parts.append(f"{np.clip(x / iw, 0, 1):.6f} {np.clip(y / ih, 0, 1):.6f} 2")
    return " ".join(parts)


def center_label_line(center_xy: np.ndarray, box_xyxy: np.ndarray, iw: int, ih: int) -> str:
    x0, y0, x1, y1 = box_xyxy
    cx = ((x0 + x1) * 0.5) / iw
    cy = ((y0 + y1) * 0.5) / ih
    w = max(1e-6, (x1 - x0) / iw)
    h = max(1e-6, (y1 - y0) / ih)
    kx = float(np.clip(center_xy[0] / iw, 0, 1))
    ky = float(np.clip(center_xy[1] / ih, 0, 1))
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kx:.6f} {ky:.6f} 2"


def write_yaml(out: Path, kpt_n: int, title: str) -> Path:
    flip = list(range(kpt_n))  # 中心点无左右互换；四角用 TL-TR-BL-BR
    if kpt_n == 4:
        flip = [1, 0, 3, 2]
    text = f"""# {title}
path: {out.as_posix()}
train: images/train
val: images/val

names:
  0: vertebra

kpt_shape: [{kpt_n}, 3]
flip_idx: {flip}
"""
    path = out / "data.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def load_suitable_stems(list_path: Path | None) -> set[str] | None:
    if list_path is None or not list_path.is_file():
        return None
    out: set[str] = set()
    for ln in list_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.add(Path(ln).stem)
    return out or None


def collect_aasce_items(
    aasce_dir: Path,
    extra_dirs: list[Path],
    extra_id_max: int,
    suitable_stems: set[str] | None = None,
) -> list[tuple[Path, np.ndarray]]:
    """返回 (img_path, pts68)。只用原图旁 ``*_corner.pts``。

    注意：AASCE challenge test 的 MICCAI pts 与 Kaggle 原图经常错位，
    默认不要把 test 目录塞进 extra_dirs。
    """
    items: list[tuple[Path, np.ndarray]] = []

    def add_from(src: Path, id_max: int | None, apply_suitable: bool) -> None:
        for img in sorted(src.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            if apply_suitable and suitable_stems is not None and img.stem not in suitable_stems:
                continue
            if id_max is not None:
                cid = parse_challenge_id(img.stem)
                if cid is None or cid < 1 or cid > id_max:
                    continue
            pts_path = img.with_name(img.stem + "_corner.pts")
            if not pts_path.is_file():
                continue
            arr = np.loadtxt(pts_path, dtype=np.float64)
            if arr.size != 136:
                continue
            items.append((img, arr.reshape(68, 2)))

    add_from(aasce_dir, None, apply_suitable=True)
    for d in extra_dirs:
        add_from(d, extra_id_max if extra_id_max > 0 else None, apply_suitable=False)
    return items


def build_aasce_patches(
    items: list[tuple[Path, np.ndarray]],
    out: Path,
    val_ratio: float,
    pad: float,
    seed: int,
    min_y_span: float = 0.40,
) -> None:
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    kept: list[tuple[Path, np.ndarray]] = []
    n_drop = 0
    for img_path, pts in items:
        try:
            iw0, ih0 = read_wh(img_path)
        except Exception:
            n_drop += 1
            continue
        if not aasce_pts_scale_ok(pts, iw0, ih0, min_y_span=min_y_span):
            n_drop += 1
            continue
        kept.append((img_path, pts))
    if n_drop:
        print(f"[patch] drop scale-mismatch AASCE: {n_drop}, keep {len(kept)}")

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_val = max(1, int(round(len(kept) * val_ratio))) if kept else 0
    if n_val >= len(kept) and len(kept) > 1:
        n_val = 1
    val_paths = {p for p, _ in kept[:n_val]} if kept else set()
    n_train = n_val_p = 0
    for img_path, pts in kept:
        split = "val" if img_path in val_paths else "train"
        try:
            bgr = _imread(img_path)
        except Exception:
            continue
        iw0, ih0 = read_wh(img_path)
        for vi in range(17):
            q = normalize_quad(pts[vi * 4 : (vi + 1) * 4])
            if not quad_ok(q, iw0, ih0, strict=False):
                continue
            got = crop_vert_patch(bgr, q, pad=pad)
            if got is None:
                continue
            crop, q2 = got
            # 拒掉近乎纯色裁切（高清图+错尺度 .pts 常裁到空白灰区）
            if float(crop.std()) < 6.0 or int(crop.max()) - int(crop.min()) < 12:
                continue
            ch, cw = crop.shape[:2]
            name = f"{img_path.stem}_v{vi:02d}.jpg"
            _imwrite(out / "images" / split / name, crop)
            (out / "labels" / split / f"{Path(name).stem}.txt").write_text(
                patch_label_line(q2, cw, ch) + "\n", encoding="utf-8"
            )
            if split == "train":
                n_train += 1
            else:
                n_val_p += 1

    write_yaml(out, 4, "AASCE vertebra patches, 4 corners (TL,TR,BL,BR)")
    print(f"[patch] train={n_train} val={n_val_p} -> {out}")


def build_aasce_context_patches(
    items: list[tuple[Path, np.ndarray]],
    out: Path,
    val_ratio: float,
    seed: int,
    min_y_span: float = 0.40,
    lumbar_oversample: bool = True,
) -> None:
    """邻椎上下文矩形 patch：高=邻距，宽=1.7×vert_size；腰段可过采样。"""
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    kept: list[tuple[Path, np.ndarray]] = []
    n_drop = 0
    for img_path, pts in items:
        try:
            iw0, ih0 = read_wh(img_path)
        except Exception:
            n_drop += 1
            continue
        if not aasce_pts_scale_ok(pts, iw0, ih0, min_y_span=min_y_span):
            n_drop += 1
            continue
        kept.append((img_path, pts))
    if n_drop:
        print(f"[ctx-patch] drop scale-mismatch AASCE: {n_drop}, keep {len(kept)}")

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_val = max(1, int(round(len(kept) * val_ratio))) if kept else 0
    if n_val >= len(kept) and len(kept) > 1:
        n_val = 1
    val_paths = {p for p, _ in kept[:n_val]} if kept else set()
    n_train = n_val_p = n_os = 0

    def _write_one(split: str, stem: str, crop: np.ndarray, q2: np.ndarray) -> None:
        nonlocal n_train, n_val_p
        ch, cw = crop.shape[:2]
        name = f"{stem}.jpg"
        _imwrite(out / "images" / split / name, crop)
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text(
            patch_label_line(q2, cw, ch) + "\n", encoding="utf-8"
        )
        if split == "train":
            n_train += 1
        else:
            n_val_p += 1

    for img_path, pts in kept:
        split = "val" if img_path in val_paths else "train"
        try:
            bgr = _imread(img_path)
        except Exception:
            continue
        iw0, ih0 = read_wh(img_path)
        quads: list[np.ndarray] = []
        ok_mask: list[bool] = []
        for vi in range(17):
            q = normalize_quad(pts[vi * 4 : (vi + 1) * 4])
            good = bool(quad_ok(q, iw0, ih0, strict=False))
            ok_mask.append(good)
            quads.append(q)
        if sum(ok_mask) < 10:
            continue
        for vi in range(17):
            if not ok_mask[vi]:
                continue
            got = crop_vert_context_patch(bgr, quads, vi)
            if got is None:
                continue
            crop, q2, _w, _h = got
            if float(crop.std()) < 6.0 or int(crop.max()) - int(crop.min()) < 12:
                continue
            stem = f"{img_path.stem}_v{vi:02d}"
            _write_one(split, stem, crop, q2)
            if split == "train" and lumbar_oversample and vi in LUMBAR_VI:
                _write_one(split, f"{stem}_os", crop, q2)
                n_os += 1

    write_yaml(
        out,
        4,
        "AASCE context patches (neighbor height, narrow width); 4 corners current vert only",
    )
    print(f"[ctx-patch] train={n_train} val={n_val_p} lumbar_os={n_os} -> {out}")


def build_aasce_context_patches_curve(
    items: list[tuple[Path, np.ndarray]],
    out: Path,
    val_ratio: float,
    seed: int,
    min_y_span: float = 0.40,
    high_cobb_quantile: float = 0.75,
    high_cobb_extra: int = 2,
    apex_extra: int = 1,
) -> None:
    """定向上下文 patch + 高弯/顶椎过采样，专攻大弯段角点。"""
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    kept: list[tuple[Path, np.ndarray, float, int]] = []
    n_drop = 0
    cobbs: list[float] = []
    for img_path, pts in items:
        try:
            iw0, ih0 = read_wh(img_path)
        except Exception:
            n_drop += 1
            continue
        if not aasce_pts_scale_ok(pts, iw0, ih0, min_y_span=min_y_span):
            n_drop += 1
            continue
        meta = approx_cobb_from_pts68(pts)
        ang, apex = (0.0, 8) if meta is None else meta
        kept.append((img_path, pts, float(ang), int(apex)))
        cobbs.append(float(ang))
    if n_drop:
        print(f"[ctx-curve] drop scale-mismatch AASCE: {n_drop}, keep {len(kept)}")
    thr = float(np.quantile(cobbs, high_cobb_quantile)) if cobbs else 1e9
    print(f"[ctx-curve] high-cobb thr(q{high_cobb_quantile:.2f})={thr:.1f} deg")

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_val = max(1, int(round(len(kept) * val_ratio))) if kept else 0
    if n_val >= len(kept) and len(kept) > 1:
        n_val = 1
    val_paths = {p for p, _, _, _ in kept[:n_val]} if kept else set()
    n_train = n_val_p = n_hi = n_apex = 0

    def _write_one(split: str, stem: str, crop: np.ndarray, q2: np.ndarray) -> None:
        nonlocal n_train, n_val_p
        ch, cw = crop.shape[:2]
        name = f"{stem}.jpg"
        _imwrite(out / "images" / split / name, crop)
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text(
            patch_label_line(q2, cw, ch) + "\n", encoding="utf-8"
        )
        if split == "train":
            n_train += 1
        else:
            n_val_p += 1

    for img_path, pts, ang, apex in kept:
        split = "val" if img_path in val_paths else "train"
        try:
            bgr = _imread(img_path)
        except Exception:
            continue
        iw0, ih0 = bgr.shape[1], bgr.shape[0]
        quads: list[np.ndarray] = []
        ok_mask: list[bool] = []
        for vi in range(17):
            q = normalize_quad(pts[vi * 4 : (vi + 1) * 4])
            good = bool(quad_ok(q, iw0, ih0, strict=False))
            ok_mask.append(good)
            quads.append(q)
        if sum(ok_mask) < 10:
            continue
        centers = np.asarray([q.mean(axis=0) for q in quads], dtype=np.float64)
        high = split == "train" and ang >= thr
        img_reps = 1 + (high_cobb_extra if high else 0)

        for vi in range(17):
            if not ok_mask[vi]:
                continue
            x0, y0 = quads[vi].min(axis=0)
            x1, y1 = quads[vi].max(axis=0)
            vs = float(max(x1 - x0, y1 - y0, 16.0))
            got = crop_oriented_context_at_xy(
                bgr,
                quads[vi],
                float(centers[vi, 0]),
                float(centers[vi, 1]),
                centers,
                vi,
                vs,
            )
            if got is None:
                continue
            crop, q2, _Minv, _w, _h = got
            if q2 is None:
                continue
            if float(crop.std()) < 6.0 or int(crop.max()) - int(crop.min()) < 12:
                continue
            near_apex = abs(vi - apex) <= 1
            vert_reps = img_reps
            if split == "train" and near_apex:
                vert_reps += apex_extra
            if split == "train" and vi in LUMBAR_VI:
                vert_reps += 1
            for r in range(vert_reps):
                stem = f"{img_path.stem}_v{vi:02d}" if r == 0 else f"{img_path.stem}_v{vi:02d}_r{r}"
                _write_one(split, stem, crop, q2)
                if high and r > 0:
                    n_hi += 1
                if near_apex and r > 0:
                    n_apex += 1

    write_yaml(
        out,
        4,
        "AASCE oriented context patches for curves; adaptive width; high-Cobb/apex OS",
    )
    print(
        f"[ctx-curve] train={n_train} val={n_val_p} high_extra~{n_hi} apex_extra~{n_apex} -> {out}"
    )


def match_pred_to_gt_centers(
    pred_c: np.ndarray,
    gt_c: np.ndarray,
    max_dist: float,
) -> dict[int, int]:
    """GT 下标 → 预测下标；贪心一对一，超出 max_dist 不匹配。"""
    pred = np.asarray(pred_c, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt_c, dtype=np.float64).reshape(-1, 2)
    used: set[int] = set()
    mapping: dict[int, int] = {}
    for gi, g in enumerate(gt):
        best, best_d = -1, 1e18
        for pi, p in enumerate(pred):
            if pi in used:
                continue
            d = float(np.linalg.norm(p - g))
            if d < best_d:
                best_d, best = d, pi
        if best >= 0 and best_d <= max_dist:
            mapping[gi] = best
            used.add(best)
    return mapping


def crop_context_at_xy(
    bgr: np.ndarray,
    quad: np.ndarray,
    cx: float,
    cy: float,
    centers_for_spacing: np.ndarray,
    center_idx: int,
    vert_size: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """用给定中心/尺度做上下文裁剪，监督四角仍为 GT quad。"""
    q = normalize_quad(quad)
    su, sd = neighbor_spacings(centers_for_spacing, center_idx)
    w, h = context_patch_size(vert_size, su, sd)
    got = crop_rect_at_center(bgr, float(cx), float(cy), w, h)
    if got is None:
        return None
    crop, xa, ya = got
    q2 = q.copy()
    q2[:, 0] -= float(xa)
    q2[:, 1] -= float(ya)
    ch, cw = crop.shape[:2]
    # 允许中心略偏：四角可稍出边界，再 clip；出太多则丢弃
    if (q2[:, 0] < -0.35 * cw).any() or (q2[:, 1] < -0.35 * ch).any():
        return None
    if (q2[:, 0] > cw + 0.35 * cw).any() or (q2[:, 1] > ch + 0.35 * ch).any():
        return None
    # 中心应大致在 patch 内
    qc = q2.mean(axis=0)
    if qc[0] < -0.05 * cw or qc[0] > 1.05 * cw or qc[1] < -0.05 * ch or qc[1] > 1.05 * ch:
        return None
    q2[:, 0] = np.clip(q2[:, 0], 0, cw - 1)
    q2[:, 1] = np.clip(q2[:, 1], 0, ch - 1)
    if _area(q2) < 8.0:
        return None
    return crop, q2


def build_aasce_cascade_context_patches(
    items: list[tuple[Path, np.ndarray]],
    out: Path,
    center_weights: Path,
    val_ratio: float,
    seed: int,
    min_y_span: float = 0.40,
    lumbar_oversample: bool = True,
    gt_mix_ratio: float = 0.30,
    center_conf: float = 0.15,
    center_imgsz: int = 1024,
    match_dist_frac: float = 0.55,
    pair_strip_prob: float = 0.35,
) -> None:
    """用中心模型预测中心裁上下文 patch，监督仍为 GT 四角（级联分布对齐）。

    gt_mix_ratio: 训练集约该比例额外写入 GT 中心裁剪，避免过拟合中心误差。
    pair_strip_prob: 训练集写入双椎条带（两套四角）的概率，用于学上下连续。
    """
    from ultralytics import YOLO  # noqa: WPS433

    # 延迟导入，避免与 infer_retrain_cascade 循环依赖在模块加载期炸掉
    from infer_retrain_cascade import _local_vert_size, detect_centers  # noqa: WPS433

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    kept: list[tuple[Path, np.ndarray]] = []
    n_drop = 0
    for img_path, pts in items:
        try:
            iw0, ih0 = read_wh(img_path)
        except Exception:
            n_drop += 1
            continue
        if not aasce_pts_scale_ok(pts, iw0, ih0, min_y_span=min_y_span):
            n_drop += 1
            continue
        kept.append((img_path, pts))
    if n_drop:
        print(f"[ctx-cas] drop scale-mismatch AASCE: {n_drop}, keep {len(kept)}")

    if not Path(center_weights).is_file():
        raise FileNotFoundError(f"缺少中心权重: {center_weights}")
    print(f"[ctx-cas] center weights: {center_weights}")
    center_model = YOLO(str(center_weights))

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_val = max(1, int(round(len(kept) * val_ratio))) if kept else 0
    if n_val >= len(kept) and len(kept) > 1:
        n_val = 1
    val_paths = {p for p, _ in kept[:n_val]} if kept else set()

    n_train = n_val_p = n_cas = n_gt = n_os = n_miss = n_pair = 0

    def _write_one(split: str, stem: str, crop: np.ndarray, q2: np.ndarray) -> None:
        nonlocal n_train, n_val_p
        ch, cw = crop.shape[:2]
        name = f"{stem}.jpg"
        _imwrite(out / "images" / split / name, crop)
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text(
            patch_label_line(q2, cw, ch) + "\n", encoding="utf-8"
        )
        if split == "train":
            n_train += 1
        else:
            n_val_p += 1

    def _write_pair(split: str, stem: str, crop: np.ndarray, q_a: np.ndarray, q_b: np.ndarray) -> None:
        nonlocal n_train, n_val_p, n_pair
        ch, cw = crop.shape[:2]
        name = f"{stem}.jpg"
        _imwrite(out / "images" / split / name, crop)
        lab = patch_label_line(q_a, cw, ch) + "\n" + patch_label_line(q_b, cw, ch) + "\n"
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text(lab, encoding="utf-8")
        if split == "train":
            n_train += 1
        else:
            n_val_p += 1
        n_pair += 1

    for img_path, pts in kept:
        split = "val" if img_path in val_paths else "train"
        try:
            bgr = _imread(img_path)
        except Exception:
            continue
        iw0, ih0 = bgr.shape[1], bgr.shape[0]
        quads: list[np.ndarray] = []
        ok_mask: list[bool] = []
        for vi in range(17):
            q = normalize_quad(pts[vi * 4 : (vi + 1) * 4])
            good = bool(quad_ok(q, iw0, ih0, strict=False))
            ok_mask.append(good)
            quads.append(q)
        if sum(ok_mask) < 10:
            continue
        gt_centers = np.asarray([q.mean(axis=0) for q in quads], dtype=np.float64)

        boxes, pred_centers = detect_centers(
            center_model, bgr, conf=center_conf, imgsz=center_imgsz, max_n=17
        )
        # 匹配阈值：约 0.55× 中位邻距，下限按图高
        gaps = [
            float(np.linalg.norm(gt_centers[i + 1] - gt_centers[i]))
            for i in range(16)
            if ok_mask[i] and ok_mask[i + 1]
        ]
        med_gap = float(np.median(gaps)) if gaps else 0.08 * float(ih0)
        max_dist = max(0.04 * float(ih0), match_dist_frac * med_gap)

        mapping: dict[int, int] = {}
        if len(pred_centers) >= 8:
            mapping = match_pred_to_gt_centers(pred_centers, gt_centers, max_dist=max_dist)

        for vi in range(17):
            if not ok_mask[vi]:
                continue
            wrote_cas = False
            if vi in mapping:
                pi = mapping[vi]
                vs = _local_vert_size(pred_centers, boxes, pi)
                got = crop_context_at_xy(
                    bgr,
                    quads[vi],
                    float(pred_centers[pi, 0]),
                    float(pred_centers[pi, 1]),
                    pred_centers,
                    pi,
                    vs,
                )
                if got is not None:
                    crop, q2 = got
                    if float(crop.std()) >= 6.0 and int(crop.max()) - int(crop.min()) >= 12:
                        stem = f"{img_path.stem}_v{vi:02d}_cas"
                        _write_one(split, stem, crop, q2)
                        n_cas += 1
                        wrote_cas = True
                        if split == "train" and lumbar_oversample and vi in LUMBAR_VI:
                            _write_one(split, f"{stem}_os", crop, q2)
                            n_os += 1
            if not wrote_cas:
                n_miss += 1

            # GT 混入：验证集全写一份 GT；训练集按比例
            want_gt = split == "val" or (split == "train" and rng.random() < float(gt_mix_ratio))
            if want_gt or not wrote_cas:
                got_gt = crop_vert_context_patch(bgr, quads, vi)
                if got_gt is None:
                    continue
                crop_g, q2g, _w, _h = got_gt
                if float(crop_g.std()) < 6.0 or int(crop_g.max()) - int(crop_g.min()) < 12:
                    continue
                stem_g = f"{img_path.stem}_v{vi:02d}_gt"
                _write_one(split, stem_g, crop_g, q2g)
                n_gt += 1

        # 双椎条带：学上下边相邻、少交叉
        if split == "train" and pair_strip_prob > 0:
            for vi in range(16):
                if not (ok_mask[vi] and ok_mask[vi + 1]):
                    continue
                if rng.random() > float(pair_strip_prob):
                    continue
                got_p = crop_dual_vert_strip(bgr, quads, vi)
                if got_p is None:
                    continue
                crop_p, qa, qb = got_p
                if float(crop_p.std()) < 6.0 or int(crop_p.max()) - int(crop_p.min()) < 12:
                    continue
                _write_pair(split, f"{img_path.stem}_pair{vi:02d}", crop_p, qa, qb)

    write_yaml(
        out,
        4,
        "AASCE cascade-aligned context patches + dual-vert strips; GT corners",
    )
    print(
        f"[ctx-cas] train={n_train} val={n_val_p} cas={n_cas} gt={n_gt} "
        f"lumbar_os={n_os} pair={n_pair} miss_cas={n_miss} -> {out}"
    )


def quads_to_center_label(quads: list[np.ndarray], iw: int, ih: int) -> str:
    lines = []
    for q in quads:
        c = q.mean(axis=0)
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        bw, bh = max(4.0, x1 - x0), max(4.0, y1 - y0)
        box = np.array(
            [
                max(0.0, x0 - 0.1 * bw),
                max(0.0, y0 - 0.1 * bh),
                min(iw - 1.0, x1 + 0.1 * bw),
                min(ih - 1.0, y1 + 0.1 * bh),
            ],
            dtype=np.float64,
        )
        lines.append(center_label_line(c, box, iw, ih))
    return "\n".join(lines) + "\n"


def aasce_pts_scale_ok(
    pts: np.ndarray,
    iw: int,
    ih: int,
    min_y_span: float = 0.40,
    min_y_max: float = 0.45,
) -> bool:
    """拒掉「高清大图 + 低分辨率 .pts」：点挤在图像顶部，归一化后像空标/错位。

    正常 AASCE T1–L5 的 y 跨度通常 >0.5；坏样本常见 yMax≈650、图高≈6000 → span≈0.1。
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 4 or iw <= 1 or ih <= 1:
        return False
    ys = p[:, 1] / float(ih)
    xs = p[:, 0] / float(iw)
    if float(ys.min()) < -0.02 or float(ys.max()) > 1.02:
        return False
    if float(xs.min()) < -0.02 or float(xs.max()) > 1.02:
        return False
    if float(ys.max() - ys.min()) < min_y_span:
        return False
    if float(ys.max()) < min_y_max:
        return False
    return True


def pts68_to_center_label(
    pts: np.ndarray,
    iw: int,
    ih: int,
    min_y_span: float = 0.40,
) -> str | None:
    if not aasce_pts_scale_ok(pts, iw, ih, min_y_span=min_y_span):
        return None
    quads = []
    for vi in range(17):
        q = normalize_quad(pts[vi * 4 : (vi + 1) * 4])
        if not quad_ok(q, iw, ih, strict=False):
            return None
        quads.append(q)
    return quads_to_center_label(quads, iw, ih)


def build_center_dataset(
    spinal_root: Path,
    out: Path,
    max_samples: int,
    seed: int,
    val_ratio: float,
) -> None:
    """仅 Spinal-AI2024：全身图 + 每椎 1 个中心点（不含 AASCE）。"""
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    ann_path = spinal_root / "Spinal_AI2024_train__annotation" / "Spinal_AI2024_train__annotation.json"
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)
    with open(ann_path, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_img: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        by_img[int(a["image_id"])].append(a)

    spinal_cands: list[tuple[Path, str]] = []
    for im in coco["images"]:
        file_name = im["file_name"]
        iw, ih = int(im["width"]), int(im["height"])
        path = resolve_spinal_image(spinal_root, file_name)
        if path is None:
            continue
        anns = sorted(by_img.get(int(im["id"]), []), key=lambda a: float(a.get("bbox", [0, 0, 0, 0])[1]))
        if len(anns) != 17:
            continue
        quads = []
        for a in anns:
            seg = a.get("segmentation")
            if not seg:
                continue
            poly = seg[0] if isinstance(seg[0], (list, tuple)) else seg
            if len(poly) < 8:
                continue
            q = normalize_quad(np.array(poly[:8], dtype=np.float64).reshape(4, 2))
            if quad_ok(q, iw, ih, strict=False):
                quads.append(q)
        if len(quads) != 17:
            continue
        ys = [float(q[:, 1].mean()) for q in quads]
        if sum(1 for i in range(16) if ys[i + 1] + 2.0 < ys[i]) > 2:
            continue
        quads = [q for _, q in sorted(zip(ys, quads), key=lambda t: t[0])]
        spinal_cands.append((path, quads_to_center_label(quads, iw, ih)))

    rng = random.Random(seed)
    rng.shuffle(spinal_cands)
    if max_samples > 0:
        spinal_cands = spinal_cands[:max_samples]

    n_val = max(1, int(round(len(spinal_cands) * val_ratio))) if spinal_cands else 0
    if n_val >= len(spinal_cands) and len(spinal_cands) > 1:
        n_val = max(1, len(spinal_cands) // 8)
    val_spinal = spinal_cands[:n_val]
    train_spinal = spinal_cands[n_val:]

    used: set[str] = set()

    def dump_one(img: Path, lab: str, split: str) -> None:
        stem = f"spinal_{img.stem}"
        name = f"{stem}{img.suffix.lower() if img.suffix else '.jpg'}"
        i = 2
        while name in used:
            name = f"{stem}__{i}{img.suffix.lower() if img.suffix else '.jpg'}"
            i += 1
        used.add(name)
        shutil.copy2(img, out / "images" / split / name)
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text(lab, encoding="utf-8")

    for img, lab in val_spinal:
        dump_one(img, lab, "val")
    for img, lab in train_spinal:
        dump_one(img, lab, "train")

    write_yaml(
        out,
        1,
        "centers: Spinal-AI2024 only (1 center/vert); no AASCE, no corner supervision",
    )
    print(
        f"[center] train={len(train_spinal)} val={len(val_spinal)} "
        f"(Spinal-AI2024 only) -> {out}"
    )


def _collect_spinal_center_cands(
    spinal_root: Path,
    max_samples: int,
    seed: int,
) -> list[tuple[Path, str]]:
    ann_path = spinal_root / "Spinal_AI2024_train__annotation" / "Spinal_AI2024_train__annotation.json"
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)
    with open(ann_path, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_img: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        by_img[int(a["image_id"])].append(a)

    spinal_cands: list[tuple[Path, str]] = []
    for im in coco["images"]:
        file_name = im["file_name"]
        iw, ih = int(im["width"]), int(im["height"])
        path = resolve_spinal_image(spinal_root, file_name)
        if path is None:
            continue
        anns = sorted(by_img.get(int(im["id"]), []), key=lambda a: float(a.get("bbox", [0, 0, 0, 0])[1]))
        if len(anns) != 17:
            continue
        quads = []
        for a in anns:
            seg = a.get("segmentation")
            if not seg:
                continue
            poly = seg[0] if isinstance(seg[0], (list, tuple)) else seg
            if len(poly) < 8:
                continue
            q = normalize_quad(np.array(poly[:8], dtype=np.float64).reshape(4, 2))
            if quad_ok(q, iw, ih, strict=False):
                quads.append(q)
        if len(quads) != 17:
            continue
        ys = [float(q[:, 1].mean()) for q in quads]
        if sum(1 for i in range(16) if ys[i + 1] + 2.0 < ys[i]) > 2:
            continue
        quads = [q for _, q in sorted(zip(ys, quads), key=lambda t: t[0])]
        spinal_cands.append((path, quads_to_center_label(quads, iw, ih)))

    rng = random.Random(seed)
    rng.shuffle(spinal_cands)
    if max_samples > 0:
        spinal_cands = spinal_cands[:max_samples]
    return spinal_cands


def build_center_mix_v2(
    spinal_root: Path,
    aasce_items: list[tuple[Path, np.ndarray]],
    out: Path,
    max_spinal: int,
    seed: int,
    val_ratio: float,
    aasce_repeat: int = 5,
    min_y_span: float = 0.40,
) -> None:
    """Spinal-AI2024 + AASCE suitable 中心点；AASCE 训练过采样以拉近域。"""
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    spinal_cands = _collect_spinal_center_cands(spinal_root, max_spinal, seed)
    rng = random.Random(seed)
    rng.shuffle(spinal_cands)
    n_val_s = max(1, int(round(len(spinal_cands) * val_ratio))) if spinal_cands else 0
    if n_val_s >= len(spinal_cands) and len(spinal_cands) > 1:
        n_val_s = max(1, len(spinal_cands) // 8)
    val_spinal = spinal_cands[:n_val_s]
    train_spinal = spinal_cands[n_val_s:]

    aasce_cands: list[tuple[Path, str]] = []
    n_drop = 0
    for img_path, pts in aasce_items:
        try:
            iw, ih = read_wh(img_path)
        except Exception:
            n_drop += 1
            continue
        lab = pts68_to_center_label(pts, iw, ih, min_y_span=min_y_span)
        if lab is None:
            n_drop += 1
            continue
        aasce_cands.append((img_path, lab))
    rng.shuffle(aasce_cands)
    n_val_a = max(1, int(round(len(aasce_cands) * val_ratio))) if aasce_cands else 0
    if n_val_a >= len(aasce_cands) and len(aasce_cands) > 1:
        n_val_a = 1
    val_aasce = aasce_cands[:n_val_a]
    train_aasce = aasce_cands[n_val_a:]

    used: set[str] = set()

    def dump_one(img: Path, lab: str, split: str, prefix: str, repeat: int = 1) -> int:
        n = 0
        for r in range(max(1, int(repeat))):
            stem = f"{prefix}_{img.stem}" if r == 0 else f"{prefix}_{img.stem}__r{r}"
            name = f"{stem}{img.suffix.lower() if img.suffix else '.jpg'}"
            i = 2
            while name in used:
                name = f"{stem}__{i}{img.suffix.lower() if img.suffix else '.jpg'}"
                i += 1
            used.add(name)
            shutil.copy2(img, out / "images" / split / name)
            (out / "labels" / split / f"{Path(name).stem}.txt").write_text(lab, encoding="utf-8")
            n += 1
        return n

    n_train = n_val = 0
    for img, lab in val_spinal:
        n_val += dump_one(img, lab, "val", "spinal", 1)
    for img, lab in train_spinal:
        n_train += dump_one(img, lab, "train", "spinal", 1)
    for img, lab in val_aasce:
        n_val += dump_one(img, lab, "val", "aasce", 1)
    for img, lab in train_aasce:
        n_train += dump_one(img, lab, "train", "aasce", aasce_repeat)

    write_yaml(
        out,
        1,
        "centers mix v2: Spinal-AI2024 + AASCE suitable (oversampled); 1 center/vert",
    )
    print(
        f"[center-mix-v2] train={n_train} val={n_val} "
        f"(spinal_train={len(train_spinal)} aasce_train={len(train_aasce)}x{aasce_repeat} "
        f"drop_aasce={n_drop}) -> {out}"
    )


def build_center_mix_v3(
    spinal_root: Path,
    aasce_items: list[tuple[Path, np.ndarray]],
    out: Path,
    max_spinal: int,
    seed: int,
    val_ratio: float,
    aasce_repeat: int = 5,
    high_cobb_quantile: float = 0.75,
    high_cobb_repeat: int = 8,
    min_y_span: float = 0.40,
) -> None:
    """mix_v2 + 高弯 AASCE 加重复，稳住检出并加强弯段中心链。"""
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    spinal_cands = _collect_spinal_center_cands(spinal_root, max_spinal, seed)
    rng = random.Random(seed)
    rng.shuffle(spinal_cands)
    n_val_s = max(1, int(round(len(spinal_cands) * val_ratio))) if spinal_cands else 0
    if n_val_s >= len(spinal_cands) and len(spinal_cands) > 1:
        n_val_s = max(1, len(spinal_cands) // 8)
    val_spinal = spinal_cands[:n_val_s]
    train_spinal = spinal_cands[n_val_s:]

    aasce_cands: list[tuple[Path, str, float]] = []
    n_drop = 0
    for img_path, pts in aasce_items:
        try:
            iw, ih = read_wh(img_path)
        except Exception:
            n_drop += 1
            continue
        lab = pts68_to_center_label(pts, iw, ih, min_y_span=min_y_span)
        if lab is None:
            n_drop += 1
            continue
        meta = approx_cobb_from_pts68(pts)
        ang = 0.0 if meta is None else float(meta[0])
        aasce_cands.append((img_path, lab, ang))
    rng.shuffle(aasce_cands)
    cobbs = [a[2] for a in aasce_cands]
    thr = float(np.quantile(cobbs, high_cobb_quantile)) if cobbs else 1e9
    print(f"[center-mix-v3] high-cobb thr={thr:.1f} deg")

    n_val_a = max(1, int(round(len(aasce_cands) * val_ratio))) if aasce_cands else 0
    if n_val_a >= len(aasce_cands) and len(aasce_cands) > 1:
        n_val_a = 1
    val_aasce = aasce_cands[:n_val_a]
    train_aasce = aasce_cands[n_val_a:]

    used: set[str] = set()

    def dump_one(img: Path, lab: str, split: str, prefix: str, repeat: int = 1) -> int:
        n = 0
        for r in range(max(1, int(repeat))):
            stem = f"{prefix}_{img.stem}" if r == 0 else f"{prefix}_{img.stem}__r{r}"
            name = f"{stem}{img.suffix.lower() if img.suffix else '.jpg'}"
            i = 2
            while name in used:
                name = f"{stem}__{i}{img.suffix.lower() if img.suffix else '.jpg'}"
                i += 1
            used.add(name)
            shutil.copy2(img, out / "images" / split / name)
            (out / "labels" / split / f"{Path(name).stem}.txt").write_text(lab, encoding="utf-8")
            n += 1
        return n

    n_train = n_val = n_hi = 0
    for img, lab in val_spinal:
        n_val += dump_one(img, lab, "val", "spinal", 1)
    for img, lab in train_spinal:
        n_train += dump_one(img, lab, "train", "spinal", 1)
    for img, lab, _ang in val_aasce:
        n_val += dump_one(img, lab, "val", "aasce", 1)
    for img, lab, ang in train_aasce:
        rep = high_cobb_repeat if ang >= thr else aasce_repeat
        if ang >= thr:
            n_hi += 1
        n_train += dump_one(img, lab, "train", "aasce", rep)

    write_yaml(
        out,
        1,
        "centers mix v3: Spinal + AASCE suitable; high-Cobb AASCE extra oversample",
    )
    print(
        f"[center-mix-v3] train={n_train} val={n_val} high_cobb_imgs={n_hi} "
        f"(aasce_base_x{aasce_repeat} high_x{high_cobb_repeat} drop={n_drop}) -> {out}"
    )



def build_aasce_context_patches_adj(
    items: list[tuple[Path, np.ndarray]],
    out: Path,
    val_ratio: float,
    seed: int,
    min_y_span: float = 0.40,
    pair_prob_mid: float = 0.22,
    pair_prob_hard: float = 0.55,
) -> None:
    """轴对齐邻椎上下文 + 双椎条带（难段更高概率），不写磁盘重复副本。

    难段：T1–T3 (0..2) 与 L1–L5 (12..16) 及跨这些节段的邻接对。
    """
    HARD_VI = frozenset(list(range(0, 3)) + list(range(12, 17)))
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    kept: list[tuple[Path, np.ndarray]] = []
    n_drop = 0
    for img_path, pts in items:
        try:
            iw0, ih0 = read_wh(img_path)
        except Exception:
            n_drop += 1
            continue
        if not aasce_pts_scale_ok(pts, iw0, ih0, min_y_span=min_y_span):
            n_drop += 1
            continue
        kept.append((img_path, pts))
    if n_drop:
        print(f"[ctx-adj] drop scale-mismatch AASCE: {n_drop}, keep {len(kept)}")

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_val = max(1, int(round(len(kept) * val_ratio))) if kept else 0
    if n_val >= len(kept) and len(kept) > 1:
        n_val = 1
    val_paths = {p for p, _ in kept[:n_val]} if kept else set()
    n_train = n_val_p = n_single = n_pair = 0

    def _write_one(split: str, stem: str, crop: np.ndarray, q2: np.ndarray) -> None:
        nonlocal n_train, n_val_p, n_single
        ch, cw = crop.shape[:2]
        name = f"{stem}.jpg"
        _imwrite(out / "images" / split / name, crop)
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text(
            patch_label_line(q2, cw, ch) + "\n", encoding="utf-8"
        )
        if split == "train":
            n_train += 1
        else:
            n_val_p += 1
        n_single += 1

    def _write_pair(split: str, stem: str, crop: np.ndarray, qa: np.ndarray, qb: np.ndarray) -> None:
        nonlocal n_train, n_val_p, n_pair
        ch, cw = crop.shape[:2]
        name = f"{stem}.jpg"
        _imwrite(out / "images" / split / name, crop)
        lab = patch_label_line(qa, cw, ch) + "\n" + patch_label_line(qb, cw, ch) + "\n"
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text(lab, encoding="utf-8")
        if split == "train":
            n_train += 1
        else:
            n_val_p += 1
        n_pair += 1

    for img_path, pts in kept:
        split = "val" if img_path in val_paths else "train"
        try:
            bgr = _imread(img_path)
        except Exception:
            continue
        iw0, ih0 = bgr.shape[1], bgr.shape[0]
        quads: list[np.ndarray] = []
        ok_mask: list[bool] = []
        for vi in range(17):
            q = normalize_quad(pts[vi * 4 : (vi + 1) * 4])
            good = bool(quad_ok(q, iw0, ih0, strict=False))
            ok_mask.append(good)
            quads.append(q)
        if sum(ok_mask) < 10:
            continue

        for vi in range(17):
            if not ok_mask[vi]:
                continue
            got = crop_vert_context_patch(bgr, quads, vi)
            if got is None:
                continue
            crop, q2, _w, _h = got
            if float(crop.std()) < 6.0 or int(crop.max()) - int(crop.min()) < 12:
                continue
            _write_one(split, f"{img_path.stem}_v{vi:02d}", crop, q2)

        # 双椎条带：难段邻接更高概率；每对最多写一份
        for vi in range(16):
            if not (ok_mask[vi] and ok_mask[vi + 1]):
                continue
            hard = (vi in HARD_VI) or ((vi + 1) in HARD_VI)
            prob = pair_prob_hard if hard else pair_prob_mid
            if split == "val":
                # 验证集少写一点双椎，保持分布即可
                if not hard and rng.random() > 0.15:
                    continue
                if hard and rng.random() > 0.35:
                    continue
            elif rng.random() > prob:
                continue
            got_p = crop_dual_vert_strip(bgr, quads, vi)
            if got_p is None:
                continue
            crop_p, qa, qb = got_p
            if float(crop_p.std()) < 6.0 or int(crop_p.max()) - int(crop_p.min()) < 12:
                continue
            _write_pair(split, f"{img_path.stem}_pair{vi:02d}", crop_p, qa, qb)

    write_yaml(
        out,
        4,
        "AASCE axis-aligned context + dual-vert strips (hard-region higher pair rate); no file dup",
    )
    print(
        f"[ctx-adj] train={n_train} val={n_val_p} single~{n_single} pair={n_pair} -> {out}"
    )


def crop_dual_vert_strip(
    bgr: np.ndarray,
    quads: list[np.ndarray],
    vi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """裁相邻两椎竖向 strip，返回 (crop, q0_in_crop, q1_in_crop)。"""
    if vi < 0 or vi + 1 >= len(quads):
        return None
    q0 = normalize_quad(quads[vi])
    q1 = normalize_quad(quads[vi + 1])
    all_pts = np.vstack([q0, q1])
    x0, y0 = all_pts.min(axis=0)
    x1, y1 = all_pts.max(axis=0)
    bw = max(16.0, float(x1 - x0))
    bh = max(16.0, float(y1 - y0))
    w = 1.25 * bw
    h = 1.20 * bh
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    got = crop_rect_at_center(bgr, cx, cy, w, h)
    if got is None:
        return None
    crop, xa, ya = got
    q0c = q0.copy()
    q1c = q1.copy()
    q0c[:, 0] -= float(xa)
    q0c[:, 1] -= float(ya)
    q1c[:, 0] -= float(xa)
    q1c[:, 1] -= float(ya)
    ch, cw = crop.shape[:2]
    for q2 in (q0c, q1c):
        if (q2[:, 0] < -4).any() or (q2[:, 1] < -4).any():
            return None
        if (q2[:, 0] > cw + 4).any() or (q2[:, 1] > ch + 4).any():
            return None
        q2[:, 0] = np.clip(q2[:, 0], 0, cw - 1)
        q2[:, 1] = np.clip(q2[:, 1], 0, ch - 1)
        if _area(q2) < 8.0:
            return None
    return crop, q0c, q1c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aasce-dir", default=DEFAULT_AASCE)
    ap.add_argument(
        "--aasce-extra-dir",
        action="append",
        default=None,
        help="额外 AASCE 图目录。默认不加 challenge test（其 pts 与原图常错位）",
    )
    ap.add_argument("--extra-id-max", type=int, default=30)
    ap.add_argument(
        "--include-aasce-test",
        action="store_true",
        help="显式加入 AASCE_integrated/images/test（不推荐：MICCAI pts 易错位）",
    )
    ap.add_argument(
        "--suitable-list",
        default=r"E:\脊柱侧弯\AASCE_integrated\lists\train_suitable.txt",
        help="只保留 suitable 名单中的 train 图；空字符串表示不过滤",
    )
    ap.add_argument("--spinal-root", default=DEFAULT_SPINAL)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--spinal-max", type=int, default=4000)
    ap.add_argument("--patch-pad", type=float, default=0.55)
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-patch", action="store_true", help="跳过旧方块 vert_corner_patch")
    ap.add_argument("--skip-ctx-patch", action="store_true", help="跳过邻椎上下文 vert_corner_ctx")
    ap.add_argument(
        "--build-ctx-cas",
        action="store_true",
        help="构建 vert_corner_ctx_cas（中心模型裁剪 + GT 角点 + 双椎条带）",
    )
    ap.add_argument(
        "--build-center-mix",
        action="store_true",
        help="构建 vert_center_mix_v2（Spinal + AASCE suitable 过采样）",
    )
    ap.add_argument(
        "--build-center-mix-v3",
        action="store_true",
        help="构建 vert_center_mix_v3（高弯 AASCE 加强过采样）",
    )
    ap.add_argument(
        "--build-ctx-curve",
        action="store_true",
        help="构建 vert_corner_ctx_curve（定向裁剪 + 高弯/顶椎过采样）",
    )
    ap.add_argument(
        "--build-ctx-adj",
        action="store_true",
        help="构建 vert_corner_ctx_adj（轴对齐ctx+双椎条带，无文件复制）",
    )
    ap.add_argument("--aasce-repeat", type=int, default=5, help="mix_v2/v3 中 AASCE 训练重复次数")
    ap.add_argument("--skip-center", action="store_true")
    ap.add_argument(
        "--min-aasce-yspan",
        type=float,
        default=0.40,
        help="角点 patch 用：AASCE 角点归一化 y 跨度下限，低于则丢弃",
    )
    ap.add_argument(
        "--no-lumbar-os",
        action="store_true",
        help="关闭腰段 (v12–v16) 训练过采样",
    )
    ap.add_argument(
        "--center-weights",
        default=str(_ROOT / "best_weights" / "vert_center_spinal_best.pt"),
        help="级联对齐裁剪用的中心点权重",
    )
    ap.add_argument(
        "--gt-mix-ratio",
        type=float,
        default=0.30,
        help="ctx_cas 训练集中额外混入 GT 裁剪的比例",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)

    need_aasce = (
        (not args.skip_patch)
        or (not args.skip_ctx_patch)
        or args.build_ctx_cas
        or args.build_center_mix
        or args.build_center_mix_v3
        or args.build_ctx_curve
        or args.build_ctx_adj
    )
    items: list[tuple[Path, np.ndarray]] = []
    if need_aasce:
        extras = [Path(d) for d in (args.aasce_extra_dir or [])]
        if args.include_aasce_test:
            test_dir = Path(r"E:\脊柱侧弯\AASCE_integrated\images\test")
            if test_dir.is_dir():
                extras.append(test_dir)
        suitable = load_suitable_stems(Path(args.suitable_list) if args.suitable_list else None)
        items = collect_aasce_items(
            Path(args.aasce_dir),
            extras,
            args.extra_id_max,
            suitable_stems=suitable,
        )
        print(
            f"AASCE source images: {len(items)}"
            + (f" (suitable filter={len(suitable)})" if suitable else "")
            + (f" extras={len(extras)}" if extras else " (train only, no challenge test)")
        )

    if not args.skip_patch:
        build_aasce_patches(
            items,
            out_root / "vert_corner_patch",
            args.val_ratio,
            args.patch_pad,
            args.seed,
            min_y_span=args.min_aasce_yspan,
        )

    if not args.skip_ctx_patch:
        build_aasce_context_patches(
            items,
            out_root / "vert_corner_ctx",
            args.val_ratio,
            args.seed,
            min_y_span=args.min_aasce_yspan,
            lumbar_oversample=not args.no_lumbar_os,
        )

    if args.build_ctx_cas:
        cw = Path(args.center_weights)
        if not cw.is_file():
            alt = _HERE / "runs" / "vert_center_spinal" / "weights" / "best.pt"
            if alt.is_file():
                cw = alt
        build_aasce_cascade_context_patches(
            items,
            out_root / "vert_corner_ctx_cas",
            cw,
            args.val_ratio,
            args.seed,
            min_y_span=args.min_aasce_yspan,
            lumbar_oversample=not args.no_lumbar_os,
            gt_mix_ratio=args.gt_mix_ratio,
        )

    if args.build_center_mix:
        build_center_mix_v2(
            Path(args.spinal_root),
            items,
            out_root / "vert_center_mix_v2",
            args.spinal_max,
            args.seed,
            args.val_ratio,
            aasce_repeat=args.aasce_repeat,
            min_y_span=args.min_aasce_yspan,
        )

    if args.build_center_mix_v3:
        build_center_mix_v3(
            Path(args.spinal_root),
            items,
            out_root / "vert_center_mix_v3",
            args.spinal_max,
            args.seed,
            args.val_ratio,
            aasce_repeat=args.aasce_repeat,
            min_y_span=args.min_aasce_yspan,
        )

    if args.build_ctx_curve:
        build_aasce_context_patches_curve(
            items,
            out_root / "vert_corner_ctx_curve",
            args.val_ratio,
            args.seed,
            min_y_span=args.min_aasce_yspan,
        )

    if args.build_ctx_adj:
        build_aasce_context_patches_adj(
            items,
            out_root / "vert_corner_ctx_adj",
            args.val_ratio,
            args.seed,
            min_y_span=args.min_aasce_yspan,
        )

    if not args.skip_center:
        build_center_dataset(
            Path(args.spinal_root),
            out_root / "vert_center_spinal",
            args.spinal_max,
            args.seed,
            args.val_ratio,
        )


if __name__ == "__main__":
    main()
