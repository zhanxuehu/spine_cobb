"""
椎体 Pose 推理：单类检出 → 至多 17（可更少）→ 按 y 序赋 T1..L5。

    python spine_pose_yolo/infer_vert_pose.py --weights spine_pose_yolo/runs/vert_pose1_aasce_aug/weights/best.pt --image-dir E:/脊柱侧弯/AASCE_integrated/images/test --limit 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "runs", "vert_pose1_aasce_only", "weights", "best.pt"
)
FALLBACK_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "runs", "vert_pose1_yolo11s_b", "weights", "best.pt"
)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VERT_NAMES = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]
# 参考样式：统一橙黄半透明框（BGR）
ACCENT = (60, 130, 255)
THORACIC = ACCENT
LUMBAR = ACCENT
TARGET_N = 17


def read_bgr(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def normalize_quad(pts4: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts4, dtype=np.float64).reshape(4, 2)
    order = np.argsort(pts[:, 1])
    top, bot = pts[order[:2]], pts[order[2:]]
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]
    return np.stack([top[0], top[1], bot[0], bot[1]], axis=0)


def endplate_tilt_deg(quad: np.ndarray) -> float:
    tl, tr, _, _ = normalize_quad(quad)
    ang = float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))
    while ang <= -90:
        ang += 180
    while ang > 90:
        ang -= 180
    return ang


def approx_main_cobb(quads: list[np.ndarray]) -> tuple[float, int, int] | None:
    if len(quads) < 2:
        return None
    tilts = [endplate_tilt_deg(q) for q in quads]
    best = 0.0
    i0 = i1 = 0
    for i in range(len(tilts)):
        for j in range(i + 1, len(tilts)):
            d = abs(tilts[i] - tilts[j])
            if d > 90:
                d = 180 - d
            if d > best:
                best, i0, i1 = d, i, j
    return float(best), i0, i1


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list[int]:
    """Greedy NMS on xyxy boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def filter_spine_column(
    boxes: np.ndarray,
    confs: np.ndarray,
    kpts: np.ndarray,
    img_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """去掉侧方标尺/金属件：只保留靠近主脊柱列（x 中心）的检测。"""
    if len(boxes) == 0:
        return boxes, confs, kpts
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    widths = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
    # 先用置信度较高的一半估计脊柱中线，避免标尺把 median 拉偏
    order = np.argsort(confs)[::-1]
    seed = order[: max(8, min(len(order), max(17, len(order) // 2)))]
    mid = float(np.median(cx[seed]))
    med_w = float(np.median(widths[seed]))
    # 允许侧弯导致的横向漂移，但拒绝贴边标尺
    x_tol = max(0.12 * img_w, 2.2 * med_w)
    w_lo, w_hi = 0.45 * med_w, 2.4 * med_w
    keep = []
    for i in range(len(boxes)):
        if abs(cx[i] - mid) > x_tol:
            continue
        if not (w_lo <= widths[i] <= w_hi):
            continue
        keep.append(i)
    if len(keep) < 8:
        # 放宽一次：只按 x 距离
        keep = [i for i in range(len(boxes)) if abs(cx[i] - mid) <= 1.35 * x_tol]
    if not keep:
        return boxes, confs, kpts
    idx = np.asarray(keep, dtype=np.int64)
    return boxes[idx], confs[idx], kpts[idx]


def select_vertebra_chain(
    boxes: np.ndarray,
    confs: np.ndarray,
    max_n: int = TARGET_N,
) -> list[int]:
    """Assume boxes already NMS'd and sorted by center-y.

    - 检出 <= max_n：全部保留（允许少于 17）
    - 检出 > max_n：只取一条最优连续链，长度封顶 max_n（绝不超过 17）
    """
    n = len(boxes)
    if n == 0:
        return []
    if n <= max_n:
        return list(range(n))
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    best_i = 0
    best_score = -1e18
    for i in range(n - max_n + 1):
        w_conf = confs[i : i + max_n]
        w_cy = cy[i : i + max_n]
        w_cx = cx[i : i + max_n]
        gaps = np.diff(w_cy)
        if gaps.min() <= 1.0:
            continue
        med = float(np.median(gaps))
        if med < 1e-3:
            continue
        if gaps.max() > 3.5 * med:
            continue
        gap_pen = float(np.std(gaps) / (med + 1e-6))
        # 标尺假阳常在侧方：窗口内 x 离散度大则重罚
        x_span = float(np.ptp(w_cx))
        x_std = float(np.std(w_cx))
        med_w = float(np.median(boxes[i : i + max_n, 2] - boxes[i : i + max_n, 0]))
        x_pen = (x_std / (med_w + 1e-6)) + 0.5 * (x_span / (med_w + 1e-6))
        score = float(w_conf.sum()) - 0.35 * gap_pen - 0.55 * x_pen
        if score > best_score:
            best_score = score
            best_i = i
    if best_score < -1e17:
        # fallback: prefer windows with smallest x-spread among top conf sums
        cands = []
        for i in range(n - max_n + 1):
            conf_sum = float(confs[i : i + max_n].sum())
            x_std = float(np.std(cx[i : i + max_n]))
            cands.append((conf_sum - 0.01 * x_std, i))
        best_i = max(cands)[1]
    return list(range(best_i, best_i + max_n))


def draw(bgr, quads: list[np.ndarray], names: list[str], cobb=None, alpha: float = 0.32):
    """参考样式：半透明填充四边形 + 细描边 + 角点 + 居中标签（图上不写 Cobb 文字）。"""
    canvas = bgr.copy()
    if not quads:
        return canvas

    # 相对椎体高度自适应，全身/局部观感更接近参考图
    mean_h = float(np.mean([float(np.ptp(normalize_quad(q)[:, 1])) for q in quads]))
    scale = float(np.clip(mean_h / 28.0, 0.7, 2.8))

    thick = max(1, int(round(1.2 * scale)))
    radius = max(2, int(round(3.2 * scale)))
    font_scale = float(np.clip(0.42 * scale, 0.32, 1.15))
    font_thick = max(1, int(round(1.4 * scale)))
    color = ACCENT

    overlay = canvas.copy()
    for q in quads:
        qn = normalize_quad(q)
        # TL-TR-BR-BL，与参考倾斜框一致
        poly = np.array([qn[0], qn[1], qn[3], qn[2]], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], color, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)

    for i, q in enumerate(quads):
        qn = normalize_quad(q)
        poly = np.array([qn[0], qn[1], qn[3], qn[2]], dtype=np.int32)
        cv2.polylines(canvas, [poly], True, color, thick, cv2.LINE_AA)
        for p in qn:
            cv2.circle(canvas, (int(p[0]), int(p[1])), radius, color, -1, cv2.LINE_AA)

        name = names[i] if i < len(names) else str(i + 1)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        cx = int(qn[:, 0].mean())
        cy = int(qn[:, 1].mean())
        org = (cx - tw // 2, cy + th // 2)
        cv2.putText(
            canvas, name, org,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA,
        )
    return canvas


def predict_one(
    model: YOLO,
    image: str | np.ndarray,
    conf: float,
    imgsz: int,
    max_n: int = TARGET_N,
    target_n: int | None = None,
):
    """``image`` 可为文件路径或 BGR ndarray（如 ROI 裁剪图）。

    椎体数：允许 < max_n，禁止 > max_n（默认 17）。
    """
    if target_n is not None:
        max_n = int(target_n)
    max_n = max(1, int(max_n))
    bgr = read_bgr(image) if isinstance(image, str) else np.asarray(image)
    # low conf to recall; max_n chain select will prune extras
    res = model.predict(bgr, conf=conf, imgsz=imgsz, max_det=80, verbose=False)[0]
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return bgr, [], [], None

    kxy = res.keypoints.xy.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()

    quads_all = []
    valid_boxes = []
    valid_confs = []
    for i in range(len(boxes)):
        if kxy[i].shape[0] < 4:
            continue
        quads_all.append(normalize_quad(kxy[i][:4]))
        valid_boxes.append(boxes[i])
        valid_confs.append(confs[i])
    if not quads_all:
        return bgr, [], [], None

    boxes_a = np.asarray(valid_boxes, dtype=np.float64)
    confs_a = np.asarray(valid_confs, dtype=np.float64)
    kpts_a = np.asarray(quads_all, dtype=np.float64)

    # NMS → 去掉侧方标尺 → 按 y 排序 → 至多 max_n（可更少）
    keep_nms = nms_indices(boxes_a, confs_a, 0.45)
    boxes_a, confs_a, kpts_a = boxes_a[keep_nms], confs_a[keep_nms], kpts_a[keep_nms]
    img_w = int(bgr.shape[1])
    boxes_a, confs_a, kpts_a = filter_spine_column(boxes_a, confs_a, kpts_a, img_w)
    cy = 0.5 * (boxes_a[:, 1] + boxes_a[:, 3])
    order = np.argsort(cy)
    boxes_a, confs_a, kpts_a = boxes_a[order], confs_a[order], kpts_a[order]

    idxs = select_vertebra_chain(boxes_a, confs_a, max_n=max_n)
    quads = [kpts_a[i] for i in idxs]
    # 自上而下赋名；不足 17 时只用前 n 个名称（不强制凑满）
    names = list(VERT_NAMES[: len(quads)])
    cobb = approx_main_cobb(quads)
    return bgr, quads, names, cobb


def resolve_weights(path: str) -> str:
    if path and os.path.isfile(path):
        return path
    if os.path.isfile(FALLBACK_WEIGHTS):
        return FALLBACK_WEIGHTS
    raise FileNotFoundError(path or DEFAULT_WEIGHTS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--image", default=None)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "runs", "infer_vert_pose1"))
    ap.add_argument("--conf", type=float, default=0.15, help="偏低以提高召回，靠 max_n 封顶砍假阳")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--max-n",
        "--target-n",
        dest="max_n",
        type=int,
        default=TARGET_N,
        help="椎体数上限（可更少，不可更多）；默认 17",
    )
    args = ap.parse_args()

    weights = resolve_weights(args.weights)
    print(f"weights: {weights}")

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
    model = YOLO(weights)
    for p in paths:
        bgr, quads, names, cobb = predict_one(
            model, str(p), args.conf, args.imgsz, max_n=args.max_n
        )
        vis = draw(bgr, quads, names, cobb=cobb)
        out = os.path.join(args.out_dir, p.stem + "_vert.png")
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
