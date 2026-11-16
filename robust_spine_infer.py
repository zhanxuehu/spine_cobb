

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from spine_pose_yolo.infer_vert_pose import TARGET_N, predict_one
from spine_roi_yolo.infer_roi import expand_xyxy


def center_strip_xyxy(
    iw: int,
    ih: int,
    width_ratio: float = 0.48,
    top_ratio: float = 0.02,
    bottom_ratio: float = 0.02,
) -> tuple[int, int, int, int]:
    bw = max(32, int(round(iw * width_ratio)))
    cx = iw // 2
    x0 = max(0, cx - bw // 2)
    x1 = min(iw - 1, x0 + bw)
    y0 = max(0, int(round(ih * top_ratio)))
    y1 = min(ih - 1, int(round(ih * (1.0 - bottom_ratio))))
    return x0, y0, x1, y1


def detect_roi_xyxy(
    roi_model: YOLO | None,
    bgr: np.ndarray,
    conf: float = 0.20,
    imgsz: int = 640,
) -> tuple[float, float, float, float] | None:
    if roi_model is None:
        return None
    pred = roi_model.predict(bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    if pred.boxes is None or len(pred.boxes) == 0:
        return None
    confs = pred.boxes.conf.cpu().numpy()
    xyxys = pred.boxes.xyxy.cpu().numpy()
    i = int(np.argmax(confs))
    return tuple(map(float, xyxys[i]))


def propose_crop_boxes(
    bgr: np.ndarray,
    roi_model: YOLO | None = None,
    roi_conf: float = 0.20,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    ih, iw = bgr.shape[:2]
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    tall = (ih / max(iw, 1)) >= 1.35

    def add_roi_candidates() -> None:
        raw_roi = detect_roi_xyxy(roi_model, bgr, conf=roi_conf)
        if raw_roi is None:
            return
        boxes.append(
            (
                "roi",
                expand_xyxy(
                    raw_roi, iw, ih,
                    margin_x=0.35, margin_y=0.10,
                    min_width_ratio=0.36, min_height_ratio=0.55,
                ),
            )
        )
        boxes.append(
            (
                "roi_wide",
                expand_xyxy(
                    raw_roi, iw, ih,
                    margin_x=0.55, margin_y=0.12,
                    min_width_ratio=0.48, min_height_ratio=0.60,
                ),
            )
        )

    if tall:
    
        for name, wr in (("center48", 0.48), ("center58", 0.58), ("center38", 0.38)):
            boxes.append((name, center_strip_xyxy(iw, ih, width_ratio=wr)))
        add_roi_candidates()
        boxes.append(("full", (0, 0, iw - 1, ih - 1)))
    else:
       
        boxes.append(("full", (0, 0, iw - 1, ih - 1)))
        add_roi_candidates()

    uniq: list[tuple[str, tuple[int, int, int, int]]] = []
    seen = set()
    for name, box in boxes:
        key = tuple(box)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, box))
    return uniq


def _score_result(n: int, max_n: int = TARGET_N) -> float:
    if n <= 0:
        return -1e6
    if n > max_n:
        return -100.0 * (n - max_n)
    return float(n) * 10.0 - float(max_n - n) * 2.0


def robust_pose_infer(
    pose_model: YOLO,
    bgr: np.ndarray,
    conf: float = 0.15,
    imgsz: int = 1024,
    roi_model: YOLO | None = None,
    roi_conf: float = 0.20,
    max_n: int = TARGET_N,
    target_n: int | None = None,
    early_stop: bool = True,
):

    if target_n is not None:
        max_n = int(target_n)
    best = None
    best_score = -1e18
    tried = []

    for name, box in propose_crop_boxes(bgr, roi_model=roi_model, roi_conf=roi_conf):
        x0, y0, x1, y1 = box
        if x1 <= x0 + 8 or y1 <= y0 + 8:
            continue
        crop = bgr[y0 : y1 + 1, x0 : x1 + 1]
        if crop.size == 0:
            continue
        _, quads, names, cobb = predict_one(
            pose_model, crop, conf, imgsz, max_n=max_n
        )
        offset = np.array([x0, y0], dtype=np.float64)
        quads_full = [q + offset for q in quads]
        n = len(quads_full)
        score = _score_result(n, max_n)
        tried.append((name, n, score))
        if score > best_score:
            best_score = score
            best = (quads_full, names, cobb, name, box)
        if early_stop and n >= max(12, max_n - 2):
            break

    if best is None:
        return [], [], None, "none", (0, 0, 0, 0), tried
    quads_full, names, cobb, name, box = best
    return quads_full, names, cobb, name, box, tried
