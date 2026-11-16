"""
级联推理：Spinal 混训权重负责「找椎体」；AASCE 权重在每椎 ROI 内精修 4 角。

    python spine_pose_yolo/cascade_infer.py ^
      --detect-weights best_weights/vert_pose1_yolo11s_b_best.pt ^
      --refine-weights best_weights/vert_pose1_aasce_only_best.pt ^
      --image-dir E:/脊柱侧弯/AASCE_integrated/images/test --limit 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
for _p in (_HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from spine_pose_yolo.infer_vert_pose import (  # noqa: E402
        IMG_EXTS,
        TARGET_N,
        VERT_NAMES,
        approx_main_cobb,
        draw,
        filter_spine_column,
        nms_indices,
        normalize_quad,
        read_bgr,
        resolve_weights,
        select_vertebra_chain,
    )
except ImportError:
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
        resolve_weights,
        select_vertebra_chain,
    )

DEFAULT_DETECT = os.path.join(ROOT, "best_weights", "vert_pose1_yolo11s_b_best.pt")
DEFAULT_REFINE = os.path.join(ROOT, "best_weights", "vert_pose1_aasce_only_best.pt")


def _resolve_best(path: str, *fallbacks: str) -> str:
    if path and os.path.isfile(path):
        return path
    for fb in fallbacks:
        if fb and os.path.isfile(fb):
            return fb
    return resolve_weights(path)


def detect_chain(
    model: YOLO,
    image: str | np.ndarray,
    conf: float,
    imgsz: int,
    max_n: int = TARGET_N,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[str]]:
    """检出 + 列过滤 + 至多 max_n。返回 bgr, boxes(xyxy), quads, names。"""
    bgr = read_bgr(image) if isinstance(image, str) else np.asarray(image)
    res = model.predict(bgr, conf=conf, imgsz=imgsz, max_det=80, verbose=False)[0]
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return bgr, np.zeros((0, 4), dtype=np.float64), [], []

    kxy = res.keypoints.xy.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()

    quads_all, valid_boxes, valid_confs = [], [], []
    for i in range(len(boxes)):
        if kxy[i].shape[0] < 4:
            continue
        quads_all.append(normalize_quad(kxy[i][:4]))
        valid_boxes.append(boxes[i])
        valid_confs.append(float(confs[i]))
    if not quads_all:
        return bgr, np.zeros((0, 4), dtype=np.float64), [], []

    boxes_a = np.asarray(valid_boxes, dtype=np.float64)
    confs_a = np.asarray(valid_confs, dtype=np.float64)
    kpts_a = np.asarray(quads_all, dtype=np.float64)

    keep = nms_indices(boxes_a, confs_a, 0.45)
    boxes_a, confs_a, kpts_a = boxes_a[keep], confs_a[keep], kpts_a[keep]
    boxes_a, confs_a, kpts_a = filter_spine_column(boxes_a, confs_a, kpts_a, int(bgr.shape[1]))
    order = np.argsort(0.5 * (boxes_a[:, 1] + boxes_a[:, 3]))
    boxes_a, confs_a, kpts_a = boxes_a[order], confs_a[order], kpts_a[order]

    idxs = select_vertebra_chain(boxes_a, confs_a, max_n=max_n)
    boxes_out = boxes_a[idxs]
    quads = [kpts_a[i] for i in idxs]
    names = list(VERT_NAMES[: len(quads)])
    return bgr, boxes_out, quads, names


def _expand_box(box: np.ndarray, iw: int, ih: int, pad: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = map(float, box)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    x0 = max(0.0, x0 - bw * pad)
    y0 = max(0.0, y0 - bh * pad)
    x1 = min(float(iw - 1), x1 + bw * pad)
    y1 = min(float(ih - 1), y1 + bh * pad)
    return int(x0), int(y0), int(x1), int(y1)


def refine_one_quad(
    refine_model: YOLO,
    bgr: np.ndarray,
    box: np.ndarray,
    fallback_quad: np.ndarray,
    conf: float = 0.10,
    imgsz: int = 640,
    pad: float = 0.45,
) -> np.ndarray:
    """在检测框 ROI 内用 AASCE 模型精修 4 角；失败则回退 detect 角点。"""
    ih, iw = bgr.shape[:2]
    x0, y0, x1, y1 = _expand_box(box, iw, ih, pad)
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return normalize_quad(fallback_quad)

    crop = bgr[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return normalize_quad(fallback_quad)

    res = refine_model.predict(crop, conf=conf, imgsz=imgsz, max_det=20, verbose=False)[0]
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return normalize_quad(fallback_quad)

    kxy = res.keypoints.xy.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    # ROI 中心优先：选中心最靠近 crop 中心的实例
    cx_t = 0.5 * (crop.shape[1] - 1)
    cy_t = 0.5 * (crop.shape[0] - 1)
    best_i, best_score = -1, -1e18
    for i in range(len(boxes)):
        if kxy[i].shape[0] < 4:
            continue
        bx = boxes[i]
        cx = 0.5 * (bx[0] + bx[2])
        cy = 0.5 * (bx[1] + bx[3])
        dist = ((cx - cx_t) ** 2 + (cy - cy_t) ** 2) ** 0.5
        # 归一化距离 + 置信度
        diag = float(np.hypot(crop.shape[1], crop.shape[0]) + 1e-6)
        score = float(confs[i]) - 1.2 * (dist / diag)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i < 0:
        return normalize_quad(fallback_quad)

    q = normalize_quad(kxy[best_i][:4])
    q[:, 0] += x0
    q[:, 1] += y0
    fb = normalize_quad(fallback_quad)
    if not _quad_plausible(q, fb, box):
        return fb
    return q


def _quad_plausible(q: np.ndarray, fallback: np.ndarray, box: np.ndarray) -> bool:
    """精修结果若面积/中心离谱，则视为失败。"""
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    fb = np.asarray(fallback, dtype=np.float64).reshape(4, 2)

    def _area(pts: np.ndarray) -> float:
        x, y = pts[:, 0], pts[:, 1]
        return abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    aq, af = _area(q), _area(fb)
    if aq < 4.0:
        return False
    if af > 1.0 and (aq > 3.5 * af or aq < 0.25 * af):
        return False
    qc = q.mean(axis=0)
    bc = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])], dtype=np.float64)
    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    if abs(qc[0] - bc[0]) > 0.85 * bw or abs(qc[1] - bc[1]) > 0.85 * bh:
        return False
    # 交叉四边形：对角线中点应接近
    mid_ac = 0.5 * (q[0] + q[3])
    mid_bd = 0.5 * (q[1] + q[2])
    if np.linalg.norm(mid_ac - mid_bd) > 0.35 * max(bw, bh):
        return False
    return True


def predict_cascade(
    detect_model: YOLO,
    refine_model: YOLO | None,
    image: str | np.ndarray,
    conf: float = 0.15,
    imgsz: int = 1024,
    max_n: int = TARGET_N,
    refine_conf: float = 0.10,
    refine_imgsz: int = 640,
    refine_pad: float = 0.45,
):
    """返回 bgr, quads, names, cobb。"""
    bgr, boxes, quads0, names = detect_chain(detect_model, image, conf, imgsz, max_n=max_n)
    if not quads0:
        return bgr, [], [], None

    if refine_model is None:
        cobb = approx_main_cobb(quads0)
        return bgr, quads0, names, cobb

    quads = [
        refine_one_quad(
            refine_model,
            bgr,
            boxes[i],
            quads0[i],
            conf=refine_conf,
            imgsz=refine_imgsz,
            pad=refine_pad,
        )
        for i in range(len(quads0))
    ]
    cobb = approx_main_cobb(quads)
    return bgr, quads, names, cobb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detect-weights", default=DEFAULT_DETECT)
    ap.add_argument("--refine-weights", default=DEFAULT_REFINE)
    ap.add_argument("--no-refine", action="store_true", help="只跑检测，方便对照")
    ap.add_argument("--image", default=None)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "runs", "infer_cascade"))
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--refine-conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--refine-imgsz", type=int, default=640)
    ap.add_argument("--max-n", type=int, default=TARGET_N)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    detect_w = _resolve_best(
        args.detect_weights,
        DEFAULT_DETECT,
        os.path.join(ROOT, "spine_pose_yolo", "runs", "vert_pose1_yolo11s_b", "weights", "best.pt"),
    )
    refine_w = None if args.no_refine else _resolve_best(
        args.refine_weights,
        DEFAULT_REFINE,
        os.path.join(ROOT, "spine_pose_yolo", "runs", "vert_pose1_aasce_only", "weights", "best.pt"),
    )
    print(f"detect: {detect_w}")
    print(f"refine: {refine_w or '(off)'}")

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
    detect_model = YOLO(detect_w)
    refine_model = YOLO(refine_w) if refine_w else None

    for p in paths:
        bgr, quads, names, cobb = predict_cascade(
            detect_model,
            refine_model,
            str(p),
            conf=args.conf,
            imgsz=args.imgsz,
            max_n=args.max_n,
            refine_conf=args.refine_conf,
            refine_imgsz=args.refine_imgsz,
        )
        vis = draw(bgr, quads, names, cobb=cobb)
        out = os.path.join(args.out_dir, p.stem + "_cascade.png")
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
