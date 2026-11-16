"""
新重训级联：Spinal 中心模型找位置 → AASCE 上下文 patch 四角模型精修。

    python spine_pose_yolo/infer_retrain_cascade.py ^
      --center-weights spine_pose_yolo/runs/vert_center_spinal/weights/best.pt ^
      --corner-weights spine_pose_yolo/runs/vert_corner_ctx/weights/best.pt ^
      --image-dir E:/脊柱侧弯/AASCE_integrated/images/test --limit 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from infer_vert_pose import (  # noqa: E402
    IMG_EXTS,
    TARGET_N,
    VERT_NAMES,
    approx_main_cobb,
    draw,
    filter_spine_column,
    nms_indices,
    normalize_quad,
    read_bgr,
    select_vertebra_chain,
)
from prepare_retrain_datasets import (  # noqa: E402
    context_patch_size,
    crop_oriented_context_at_xy,
    crop_rect_at_center,
    map_points_with_affine,
    neighbor_spacings,
)


from corner_regressor import (  # noqa: E402
    is_corner_regressor_weights,
    load_corner_regressor,
    predict_corners_norm,
)
from train_corner_adj import AdjacentPoseModel  # noqa: E402,F401

DEFAULT_CENTER = os.path.join(_ROOT, "best_weights", "vert_center_mix_v2_best.pt")
if not os.path.isfile(DEFAULT_CENTER):
    DEFAULT_CENTER = os.path.join(_ROOT, "best_weights", "vert_center_mix_v3_best.pt")
DEFAULT_CORNER = os.path.join(_ROOT, "best_weights", "vert_corner_reg_best.pt")
if not os.path.isfile(DEFAULT_CORNER):
    DEFAULT_CORNER = os.path.join(_ROOT, "best_weights", "vert_corner_ctx_best.pt")
if not os.path.isfile(DEFAULT_CORNER):
    DEFAULT_CORNER = os.path.join(_ROOT, "best_weights", "vert_corner_ctx_adj_best.pt")
if not os.path.isfile(DEFAULT_CORNER):
    DEFAULT_CORNER = os.path.join(_ROOT, "best_weights", "vert_corner_ctx_curve_best.pt")


def load_corner_model(weights: str, device: str | None = None):
    """加载角点模型：CornerRegressor 或 YOLO-Pose。"""
    if is_corner_regressor_weights(weights):
        dev = device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
        return load_corner_regressor(weights, device=dev)
    return YOLO(weights)


def detect_centers(
    model: YOLO,
    bgr: np.ndarray,
    conf: float,
    imgsz: int,
    max_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 boxes(xyxy), centers(N,2)，已按 y 排序且至多 max_n。"""
    # 略降 conf、多留候选，靠链选择去假阳，减轻缺检
    res = model.predict(bgr, conf=min(conf, 0.12), imgsz=imgsz, max_det=100, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0, 2))

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    if res.keypoints is not None and len(res.keypoints) == len(boxes):
        kxy = res.keypoints.xy.cpu().numpy()
        centers = []
        for i in range(len(boxes)):
            if kxy[i].shape[0] >= 1 and np.isfinite(kxy[i][0]).all() and (kxy[i][0] > 1).any():
                centers.append(kxy[i][0])
            else:
                b = boxes[i]
                centers.append([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])])
        centers = np.asarray(centers, dtype=np.float64)
    else:
        centers = np.stack(
            [0.5 * (boxes[:, 0] + boxes[:, 2]), 0.5 * (boxes[:, 1] + boxes[:, 3])],
            axis=1,
        )

    keep = nms_indices(boxes, confs, 0.45)
    boxes, confs, centers = boxes[keep], confs[keep], centers[keep]
    # 伪造 kpts 走列过滤
    dummy = np.repeat(centers[:, None, :], 4, axis=1)
    boxes, confs, dummy = filter_spine_column(boxes, confs, dummy, int(bgr.shape[1]))
    centers = dummy[:, 0, :]
    order = np.argsort(centers[:, 1])
    boxes, confs, centers = boxes[order], confs[order], centers[order]
    idxs = select_vertebra_chain(boxes, confs, max_n=max_n)
    return boxes[idxs], centers[idxs]


def fill_centers_to_n(
    centers: np.ndarray,
    boxes: np.ndarray,
    n: int = TARGET_N,
) -> tuple[np.ndarray, np.ndarray]:
    """按邻距插值/外推，将中心链补到 n（缺检保底）。"""
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4) if len(boxes) else np.zeros((0, 4))
    if len(c) == 0 or len(c) >= n:
        return c[:n], b[: min(len(b), n)] if len(b) else np.zeros((0, 4))
    # 中位间距
    if len(c) >= 2:
        gaps = np.linalg.norm(np.diff(c, axis=0), axis=1)
        med = float(np.median(gaps))
        direction = c[-1] - c[0]
        direction = direction / (np.linalg.norm(direction) + 1e-6)
    else:
        med = 48.0
        direction = np.array([0.0, 1.0], dtype=np.float64)
    # 在最大 gap 处插值，直到凑满 n；再两端外推
    while len(c) < n:
        if len(c) >= 2:
            gaps = np.linalg.norm(np.diff(c, axis=0), axis=1)
            i = int(np.argmax(gaps))
            if gaps[i] > 1.35 * med:
                mid = 0.5 * (c[i] + c[i + 1])
                c = np.insert(c, i + 1, mid, axis=0)
                if len(b) >= 2:
                    bm = 0.5 * (b[min(i, len(b) - 1)] + b[min(i + 1, len(b) - 1)])
                    b = np.insert(b, min(i + 1, len(b)), bm, axis=0)
                continue
        # 外推：缺在两端则上下延伸；否则在最大 gap 硬插
        need = n - len(c)
        if need <= 0:
            break
        # 优先两端外推各一半
        n_up = need // 2
        n_dn = need - n_up
        extras_up = [c[0] - (k + 1) * med * direction for k in range(n_up)][::-1]
        extras_dn = [c[-1] + (k + 1) * med * direction for k in range(n_dn)]
        c = np.vstack([extras_up, c, extras_dn]) if extras_up or extras_dn else c
        if len(b):
            bw = float(np.median(b[:, 2] - b[:, 0])) if len(b) else 40.0
            bh = float(np.median(b[:, 3] - b[:, 1])) if len(b) else 40.0
            new_b = []
            for p in extras_up:
                new_b.append([p[0] - 0.5 * bw, p[1] - 0.5 * bh, p[0] + 0.5 * bw, p[1] + 0.5 * bh])
            new_b.extend(b.tolist())
            for p in extras_dn:
                new_b.append([p[0] - 0.5 * bw, p[1] - 0.5 * bh, p[0] + 0.5 * bw, p[1] + 0.5 * bh])
            b = np.asarray(new_b, dtype=np.float64)
        else:
            break
        break  # 一轮外推后退出，避免死循环
    # 若仍不足（仅插值路径），继续在最大 gap 插
    while len(c) < n and len(c) >= 2:
        gaps = np.linalg.norm(np.diff(c, axis=0), axis=1)
        i = int(np.argmax(gaps))
        mid = 0.5 * (c[i] + c[i + 1])
        c = np.insert(c, i + 1, mid, axis=0)
        if len(b) >= 2:
            bm = 0.5 * (b[min(i, len(b) - 1)] + b[min(i + 1, len(b) - 1)])
            b = np.insert(b, min(i + 1, len(b)), bm, axis=0)
    order = np.argsort(c[:, 1])
    c = c[order][:n]
    if len(b) >= len(order):
        b = b[order][:n]
    elif len(b):
        b = b[: min(len(b), n)]
    return c, b


def recenter_centers_x(
    bgr: np.ndarray,
    centers: np.ndarray,
    boxes: np.ndarray,
    blend: float = 0.35,
    max_shift_frac: float = 0.35,
) -> np.ndarray:
    """局部竖条边缘能量把中心 x 轻量吸向骨性中轴。"""
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 2).copy()
    if len(c) == 0:
        return c
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape[:2]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    out = c.copy()
    for i in range(len(c)):
        vs = _local_vert_size(c, boxes, i)
        cx, cy = float(c[i, 0]), float(c[i, 1])
        half_w = max(12.0, 1.15 * vs)
        half_h = max(8.0, 0.45 * vs)
        x0 = int(max(0, np.floor(cx - half_w)))
        x1 = int(min(iw, np.ceil(cx + half_w)))
        y0 = int(max(0, np.floor(cy - half_h)))
        y1 = int(min(ih, np.ceil(cy + half_h)))
        if x1 - x0 < 8 or y1 - y0 < 4:
            continue
        col = mag[y0:y1, x0:x1].mean(axis=0)
        if col.size < 5 or float(col.max()) < 1e-3:
            continue
        # 平滑后取峰
        k = max(3, int(round(0.08 * col.size)) | 1)
        ker = np.ones(k, dtype=np.float64) / k
        sm = np.convolve(col, ker, mode="same")
        j = int(np.argmax(sm))
        new_x = float(x0 + j)
        max_shift = max_shift_frac * vs
        new_x = float(np.clip(new_x, cx - max_shift, cx + max_shift))
        out[i, 0] = (1.0 - blend) * cx + blend * new_x
    return out


def stitch_adjacent_quads(
    quads: list[np.ndarray],
    centers: np.ndarray,
    angle_blend: float = 0.25,
    edge_blend: float = 0.40,
) -> list[np.ndarray]:
    """轻量邻接缝合：倾角平滑 + 上下边靠近 + 过大重叠收缩。"""
    if len(quads) < 2:
        return quads
    qs = [normalize_quad(q) for q in quads]
    cents = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    # 1) 倾角一阶平滑（相对水平）
    angs = []
    for q in qs:
        tl, tr = q[0], q[1]
        angs.append(float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))))
    angs = np.asarray(angs, dtype=np.float64)
    sm = angs.copy()
    for i in range(len(angs)):
        parts = [angs[i]]
        wts = [1.0]
        if i > 0:
            parts.append(angs[i - 1])
            wts.append(angle_blend)
        if i + 1 < len(angs):
            parts.append(angs[i + 1])
            wts.append(angle_blend)
        sm[i] = float(np.average(parts, weights=wts))
    # 绕质心旋转到平滑角
    for i, q in enumerate(qs):
        d = float(sm[i] - angs[i])
        if abs(d) < 0.5:
            continue
        rad = np.radians(d)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        cxy = q.mean(axis=0)
        r = q - cxy
        q2 = np.stack(
            [cxy[0] + r[:, 0] * cos_a - r[:, 1] * sin_a, cxy[1] + r[:, 0] * sin_a + r[:, 1] * cos_a],
            axis=1,
        )
        qs[i] = normalize_quad(q2)
    # 2) 上下边靠近：i 下边与 i+1 上边中点对齐到中点连线
    for i in range(len(qs) - 1):
        q0, q1 = qs[i], qs[i + 1]
        bot = 0.5 * (q0[2] + q0[3])
        top = 0.5 * (q1[0] + q1[1])
        mid = 0.5 * (bot + top)
        # 目标：bot / top 沿中心连线方向各移向 mid
        delta0 = (mid - bot) * edge_blend
        delta1 = (mid - top) * edge_blend
        q0 = q0.copy()
        q1 = q1.copy()
        q0[2] += delta0
        q0[3] += delta0
        q1[0] += delta1
        q1[1] += delta1
        qs[i], qs[i + 1] = normalize_quad(q0), normalize_quad(q1)
    # 3) 尺度封顶：跨度不超过邻中心距 * 1.35
    for i, q in enumerate(qs):
        vs = _local_vert_size(cents, np.zeros((len(cents), 4)), i) if len(cents) else 48.0
        span = float(max(q[:, 0].ptp(), q[:, 1].ptp()))
        lim = 1.35 * vs
        if span > lim and span > 1e-3:
            cxy = q.mean(axis=0)
            q = cxy + (q - cxy) * (lim / span)
            qs[i] = normalize_quad(q)
    return qs


def _local_vert_size(centers: np.ndarray, boxes: np.ndarray, i: int) -> float:
    """估计单椎尺度（像素）。优先邻椎中心距（≈椎高），再用检测框兜底。"""
    gaps: list[float] = []
    if i + 1 < len(centers):
        gaps.append(float(np.linalg.norm(centers[i + 1] - centers[i])))
    if i - 1 >= 0:
        gaps.append(float(np.linalg.norm(centers[i] - centers[i - 1])))
    gap = float(np.median(gaps)) if gaps else 0.0
    box_s = 0.0
    if i < len(boxes):
        bw = max(1.0, float(boxes[i, 2] - boxes[i, 0]))
        bh = max(1.0, float(boxes[i, 3] - boxes[i, 1]))
        # 斜椎 AABB 往往偏宽，取较短边更接近椎体厚度/高度量级
        box_s = float(min(bw, bh))
        box_long = float(max(bw, bh))
    else:
        box_long = 0.0
    # 邻距可靠时为主；框只作辅助，避免松框把尺度撑爆
    if gap >= 8.0:
        size = gap
        if box_s >= 8.0:
            size = 0.75 * gap + 0.25 * box_s
        # 防止异常大框把 size 拉飞
        if box_long >= 8.0:
            size = min(size, 1.15 * box_long)
    else:
        size = box_long if box_long >= 8.0 else 48.0
    return float(max(16.0, size))


def refine_corners_with_regressor(
    corner_model,
    bgr: np.ndarray,
    centers: np.ndarray,
    boxes: np.ndarray,
) -> tuple[list[np.ndarray], int]:
    """轴对齐 ctx 裁剪 + CNN 回归四角（每椎必出，无 fill 滤镜）。"""
    quads: list[np.ndarray] = []
    n_ok = 0
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    device = next(corner_model.parameters()).device
    for i, c in enumerate(centers):
        vert_size = _local_vert_size(centers, boxes, i)
        cx, cy = float(c[0]), float(c[1])
        su, sd = neighbor_spacings(centers, i)
        width, height = context_patch_size(vert_size, su, sd)
        got = crop_rect_at_center(bgr, cx, cy, width, height)
        if got is None:
            s = 0.45 * vert_size
            quads.append(
                np.array(
                    [[cx - s, cy - s], [cx + s, cy - s], [cx - s, cy + s], [cx + s, cy + s]],
                    dtype=np.float64,
                )
            )
            continue
        crop, xa, ya = got
        qn = predict_corners_norm(corner_model, crop, device=device)
        ch, cw = crop.shape[:2]
        q = qn.copy()
        q[:, 0] = q[:, 0] * float(cw) + float(xa)
        q[:, 1] = q[:, 1] * float(ch) + float(ya)
        q = normalize_quad(q)
        n_ok += 1
        quads.append(q)
    return quads, n_ok


def refine_corners_at_centers(
    corner_model,
    bgr: np.ndarray,
    centers: np.ndarray,
    boxes: np.ndarray,
    conf: float = 0.1,
    imgsz: int = 384,
    oriented: bool = True,
) -> tuple[list[np.ndarray], int]:
    """邻椎上下文裁 patch 后跑四角。

    CornerRegressor：直接回归。YOLO-Pose：oriented 可选；含历史 fill 滤镜。
    """
    if hasattr(corner_model, "backbone") and hasattr(corner_model, "head"):
        return refine_corners_with_regressor(corner_model, bgr, centers, boxes)

    quads: list[np.ndarray] = []
    n_ok = 0
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    for i, c in enumerate(centers):
        vert_size = _local_vert_size(centers, boxes, i)
        cx, cy = float(c[0]), float(c[1])
        M_inv = None
        width = height = 0.0
        crop = None
        xa = ya = 0.0
        if oriented:
            got_o = crop_oriented_context_at_xy(
                bgr, None, cx, cy, centers, i, vert_size
            )
            if got_o is not None:
                crop, _q2, M_inv, width, height = got_o
        if crop is None:
            su, sd = neighbor_spacings(centers, i)
            width, height = context_patch_size(vert_size, su, sd)
            got = crop_rect_at_center(bgr, cx, cy, width, height)
            if got is None:
                s = 0.45 * vert_size
                quads.append(
                    np.array(
                        [[cx - s, cy - s], [cx + s, cy - s], [cx - s, cy + s], [cx + s, cy + s]],
                        dtype=np.float64,
                    )
                )
                continue
            crop, xa, ya = got
            M_inv = None
        res = corner_model.predict(crop, conf=conf, imgsz=imgsz, max_det=10, verbose=False)[0]
        q = None
        if res.keypoints is not None and res.boxes is not None and len(res.boxes):
            kxy = res.keypoints.xy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            best, best_s = -1, -1e18
            ctx, cty = 0.5 * (crop.shape[1] - 1), 0.5 * (crop.shape[0] - 1)
            for j in range(len(kxy)):
                if kxy[j].shape[0] < 4:
                    continue
                pj = normalize_quad(kxy[j][:4])
                pc = pj.mean(axis=0)
                span_w = float(pj[:, 0].ptp())
                span_h = float(pj[:, 1].ptp())
                span = float(max(span_w, span_h))
                fill_w = span_w / max(float(width), 1.0)
                fill_h = span_h / max(float(height), 1.0)
                if fill_w < 0.18 or fill_w > 0.85:
                    continue
                if fill_h < 0.10 or fill_h > 0.70:
                    continue
                size_err = abs(span - vert_size) / max(vert_size, 1.0)
                if size_err > 0.85:
                    continue
                dist = float(np.hypot(pc[0] - ctx, pc[1] - cty))
                score = float(confs[j]) - 0.002 * dist - 0.35 * size_err
                if score > best_s:
                    best_s = score
                    best = j
            if best >= 0:
                q_local = normalize_quad(kxy[best][:4])
                if M_inv is not None:
                    q = normalize_quad(map_points_with_affine(q_local, M_inv))
                else:
                    q = q_local.copy()
                    q[:, 0] += float(xa)
                    q[:, 1] += float(ya)
                span_img = float(max(q[:, 0].ptp(), q[:, 1].ptp()))
                if span_img > 1.55 * vert_size:
                    q = None
        if q is None:
            s = 0.42 * vert_size
            q = np.array(
                [[cx - s, cy - s], [cx + s, cy - s], [cx - s, cy + s], [cx + s, cy + s]],
                dtype=np.float64,
            )
        else:
            n_ok += 1
        quads.append(q)
    return quads, n_ok


def predict_retrain_cascade(
    center_model: YOLO,
    corner_model,
    image: str | np.ndarray,
    conf: float = 0.15,
    imgsz: int = 1024,
    max_n: int = TARGET_N,
    corner_conf: float = 0.1,
    corner_imgsz: int = 384,
    oriented_corners: bool | None = None,
    corner_weights_name: str = "",
):
    """返回 (bgr, quads, names, cobb, diag)。

    diag 字段：
      stage: ok | no_center | no_corner | partial_corner
      n_center / n_corner_ok / n_quad
      tip: 给人看的中文原因
    """
    bgr = read_bgr(image) if isinstance(image, str) else np.asarray(image)
    boxes, centers = detect_centers(center_model, bgr, conf, imgsz, max_n)
    n_raw = int(len(centers))
    if n_raw == 0:
        diag = {
            "stage": "no_center",
            "n_center": 0,
            "n_corner_ok": 0,
            "n_quad": 0,
            "tip": "中心点模型未检出椎体（角点未运行）",
            "centers": [],
        }
        return bgr, [], [], None, diag
    # 缺检补全 + 横向轻纠偏
    filled = False
    if n_raw < max_n:
        centers, boxes = fill_centers_to_n(centers, boxes, n=max_n)
        filled = len(centers) > n_raw
    centers = recenter_centers_x(bgr, centers, boxes)
    # 纠偏后同步 box 中心（保持宽高）
    if len(boxes) == len(centers):
        for i in range(len(boxes)):
            bw = float(boxes[i, 2] - boxes[i, 0])
            bh = float(boxes[i, 3] - boxes[i, 1])
            cx, cy = float(centers[i, 0]), float(centers[i, 1])
            boxes[i] = [cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh]
    n_center = int(len(centers))
    use_reg = hasattr(corner_model, "backbone") and hasattr(corner_model, "head")
    if oriented_corners is None:
        oriented_corners = (not use_reg) and ("curve" in (corner_weights_name or "").lower())
        if not corner_weights_name and not use_reg:
            oriented_corners = "curve" in os.path.basename(DEFAULT_CORNER).lower()
    quads, n_corner_ok = refine_corners_at_centers(
        corner_model,
        bgr,
        centers,
        boxes,
        conf=corner_conf,
        imgsz=corner_imgsz,
        oriented=bool(oriented_corners) and not use_reg,
    )
    quads = stitch_adjacent_quads(quads, centers)
    names = list(VERT_NAMES[: len(quads)])
    cobb = approx_main_cobb(quads)
    fill_tip = f"，已补全 {n_raw}→{n_center}" if filled else ""
    if use_reg:
        stage = "ok" if n_corner_ok == n_center else ("partial_corner" if n_corner_ok else "no_corner")
        tip = f"中心 {n_raw} 个{fill_tip}，角点回归 {n_corner_ok}/{n_center}"
    elif n_corner_ok == 0:
        stage = "no_corner"
        tip = f"中心检出 {n_raw} 个{fill_tip}，但角点模型全部未检出（已用中心退化框）"
    elif n_corner_ok < n_center:
        stage = "partial_corner"
        tip = f"中心 {n_raw} 个{fill_tip}，角点成功 {n_corner_ok}/{n_center}"
    else:
        stage = "ok"
        tip = f"中心 {n_raw} 个{fill_tip}，角点成功 {n_corner_ok}/{n_center}"
    diag = {
        "stage": stage,
        "n_center": n_center,
        "n_center_raw": n_raw,
        "n_corner_ok": int(n_corner_ok),
        "n_quad": len(quads),
        "tip": tip,
        "centers": np.asarray(centers, dtype=np.float64).reshape(-1, 2).tolist(),
    }
    return bgr, quads, names, cobb, diag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--center-weights", default=DEFAULT_CENTER)
    ap.add_argument("--corner-weights", default=DEFAULT_CORNER)
    ap.add_argument("--image", default=None)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "runs", "infer_retrain_cascade"))
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max-n", type=int, default=TARGET_N)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    center_w = args.center_weights
    corner_w = args.corner_weights
    if not os.path.isfile(center_w):
        center_w = os.path.join(_HERE, "runs", "vert_center_mix_v2", "weights", "best.pt")
    if not os.path.isfile(center_w):
        center_w = os.path.join(_ROOT, "best_weights", "vert_center_spinal_best.pt")
    if not os.path.isfile(center_w):
        center_w = os.path.join(_HERE, "runs", "vert_center_spinal", "weights", "best.pt")
    if not os.path.isfile(corner_w):
        corner_w = os.path.join(_HERE, "runs", "vert_corner_reg", "best.pt")
    if not os.path.isfile(corner_w):
        corner_w = os.path.join(_HERE, "runs", "vert_corner_ctx", "weights", "best.pt")
    if not os.path.isfile(corner_w):
        corner_w = os.path.join(_HERE, "runs", "vert_corner_ctx_cas", "weights", "best.pt")
    if not os.path.isfile(corner_w):
        corner_w = os.path.join(_HERE, "runs", "vert_corner_patch", "weights", "best.pt")
    if not os.path.isfile(center_w) or not os.path.isfile(corner_w):
        raise FileNotFoundError(f"缺少权重 center={center_w} corner={corner_w}")

    print(f"center: {center_w}")
    print(f"corner: {corner_w}")

    paths: list[Path] = []
    if args.image:
        paths = [Path(args.image)]
    elif args.image_dir:
        root = Path(args.image_dir)
        paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in IMG_EXTS]
        if args.limit > 0:
            paths = paths[: args.limit]
    else:
        raise SystemExit("请指定 --image 或 --image-dir")

    os.makedirs(args.out_dir, exist_ok=True)
    center_model = YOLO(center_w)
    corner_model = load_corner_model(corner_w)
    for p in paths:
        bgr, quads, names, cobb, diag = predict_retrain_cascade(
            center_model,
            corner_model,
            str(p),
            conf=args.conf,
            imgsz=args.imgsz,
            max_n=args.max_n,
            corner_weights_name=os.path.basename(corner_w),
        )
        print(f"  {p.name}: {diag.get('tip', '')}")
        vis = draw(bgr, quads, names, cobb=cobb)
        out = os.path.join(args.out_dir, p.stem + "_retrain.png")
        ok, buf = cv2.imencode(".png", vis)
        if ok:
            buf.tofile(out)
        msg = f"n={len(quads)}/<={args.max_n}"
        if cobb and names:
            msg += f" cobb~{cobb[0]:.1f} {names[cobb[1]]}-{names[cobb[2]]}"
        print(f"[ok] {p.name}  {msg}")
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
