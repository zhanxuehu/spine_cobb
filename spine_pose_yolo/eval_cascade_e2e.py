"""
整图级联 vs AASCE GT 分节段验收（中心偏移 + 角点误差 + 邻接间隙）。

    python spine_pose_yolo/eval_cascade_e2e.py --limit 40
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from prepare_retrain_datasets import (  # noqa: E402
    aasce_pts_scale_ok,
    collect_aasce_items,
    load_suitable_stems,
)
from prepare_vert_pose_dataset import normalize_quad, quad_ok, read_wh  # noqa: E402
from infer_retrain_cascade import (  # noqa: E402
    DEFAULT_CENTER,
    DEFAULT_CORNER,
    load_corner_model,
    predict_retrain_cascade,
)
from train_corner_adj import AdjacentPoseModel  # noqa: E402,F401  — unpickle trainer class
from ultralytics import YOLO  # noqa: E402

SEGMENTS = {
    "T1-T3": range(0, 3),
    "T4-T12": range(3, 12),
    "L1-L5": range(12, 17),
}


def _centers_from_quads(quads: list[np.ndarray]) -> np.ndarray:
    return np.asarray([np.asarray(q).reshape(-1, 2).mean(axis=0) for q in quads], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--center-weights", default=DEFAULT_CENTER)
    ap.add_argument("--corner-weights", default=DEFAULT_CORNER)
    ap.add_argument("--aasce-dir", default=r"E:\脊柱侧弯\AASCE_integrated\images\train")
    ap.add_argument("--suitable-list", default=r"E:\脊柱侧弯\AASCE_integrated\lists\train_suitable.txt")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    suitable = load_suitable_stems(Path(args.suitable_list) if args.suitable_list else None)
    items = collect_aasce_items(Path(args.aasce_dir), [], 0, suitable_stems=suitable)
    rng = np.random.default_rng(args.seed)
    idxs = np.arange(len(items))
    rng.shuffle(idxs)
    if args.limit > 0:
        idxs = idxs[: args.limit]

    center_m = YOLO(args.center_weights)
    corner_m = load_corner_model(args.corner_weights)
    print(f"center={args.center_weights}")
    print(f"corner={args.corner_weights}")
    print(f"eval n={len(idxs)}")

    c_err: list[float] = []
    hit = 0
    n_img = 0
    kpt_err = {k: [] for k in SEGMENTS}
    gap_ratios: list[float] = []

    for i in idxs:
        img_path, pts = items[int(i)]
        try:
            iw, ih = read_wh(img_path)
        except Exception:
            continue
        if not aasce_pts_scale_ok(pts, iw, ih):
            continue
        gt_quads = []
        ok = []
        for vi in range(17):
            q = normalize_quad(pts[vi * 4 : (vi + 1) * 4])
            good = bool(quad_ok(q, iw, ih, strict=False))
            ok.append(good)
            gt_quads.append(q)
        if sum(ok) < 14:
            continue
        gt_c = _centers_from_quads(gt_quads)
        _, quads, _names, _cobb, diag = predict_retrain_cascade(
            center_m,
            corner_m,
            str(img_path),
            conf=args.conf,
            imgsz=1024,
            corner_weights_name=os.path.basename(args.corner_weights),
        )
        if len(quads) < 10:
            continue
        n_img += 1
        pred_c = np.asarray(diag.get("centers") or _centers_from_quads(quads), dtype=np.float64).reshape(-1, 2)
        # match by index assuming ordered T1.. (cascade names ordered)
        m = min(len(pred_c), 17)
        for vi in range(m):
            if not ok[vi]:
                continue
            d = float(np.linalg.norm(pred_c[vi] - gt_c[vi]))
            c_err.append(d)
            # hit if within 0.55 * local gap
            gaps = []
            if vi > 0 and ok[vi - 1]:
                gaps.append(float(np.linalg.norm(gt_c[vi] - gt_c[vi - 1])))
            if vi + 1 < 17 and ok[vi + 1]:
                gaps.append(float(np.linalg.norm(gt_c[vi + 1] - gt_c[vi])))
            thr = 0.55 * float(np.median(gaps)) if gaps else 0.06 * ih
            if d <= thr:
                hit += 1
        pq = [np.asarray(q, dtype=np.float64).reshape(-1, 2) for q in quads[:17]]
        for name, r in SEGMENTS.items():
            for vi in r:
                if vi >= len(pq) or not ok[vi]:
                    continue
                # mean corner L2 to GT
                e = float(np.linalg.norm(pq[vi] - gt_quads[vi], axis=1).mean())
                kpt_err[name].append(e)
        for vi in range(min(len(pq) - 1, 16)):
            if not (ok[vi] and ok[vi + 1]):
                continue
            bot = 0.5 * (pq[vi][2] + pq[vi][3])
            top = 0.5 * (pq[vi + 1][0] + pq[vi + 1][1])
            gap_ratios.append(float(np.linalg.norm(bot - top)))

    def _summ(xs: list[float]) -> str:
        if not xs:
            return "n=0"
        a = np.asarray(xs, dtype=np.float64)
        return f"n={len(a)} mean={a.mean():.2f} med={np.median(a):.2f} p90={np.quantile(a,0.9):.2f}"

    total_c = max(1, len(c_err))
    print(f"images={n_img}")
    print(f"center_err_px: {_summ(c_err)}  hit_rate={hit/total_c:.3f}")
    for name, xs in kpt_err.items():
        print(f"corner_err[{name}]: {_summ(xs)}")
    print(f"adj_gap_px: {_summ(gap_ratios)}")


if __name__ == "__main__":
    main()
