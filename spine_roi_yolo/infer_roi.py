"""
用训练好的 YOLO 在全身片上预测脊柱框，可选裁剪导出。

    python spine_roi_yolo/infer_roi.py --weights spine_roi_yolo/runs/spine_roi/weights/best.pt --source <图或目录>
    python spine_roi_yolo/infer_roi.py --weights ... --source ... --save-crop --out-dir runs/spine_crops
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def imread_bgr(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(path)


def list_images(source: str) -> list[str]:
    p = Path(source)
    if p.is_file():
        return [str(p)]
    return sorted(
        str(f) for f in p.iterdir()
        if f.suffix.lower() in IMG_EXTS and not f.name.startswith("._")
    )


def expand_xyxy(
    xyxy, iw, ih,
    margin_x: float = 0.28,
    margin_y: float = 0.08,
    min_width_ratio: float = 0.42,
    min_height_ratio: float = 0.55,
):
    x0, y0, x1, y1 = map(float, xyxy)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    bw2 = max(bw * (1.0 + 2.0 * margin_x), min_width_ratio * iw)
    bh2 = max(bh * (1.0 + 2.0 * margin_y), min_height_ratio * ih)
    x0, x1 = cx - 0.5 * bw2, cx + 0.5 * bw2
    y0, y1 = cy - 0.5 * bh2, cy + 0.5 * bh2
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(iw - 1, int(x1)), min(ih - 1, int(y1))
    return x0, y0, x1, y1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="图片或目录")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "runs", "infer"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--margin-x", type=float, default=0.28, help="水平外扩（严重侧弯）")
    ap.add_argument("--margin-y", type=float, default=0.08, help="竖直外扩")
    ap.add_argument("--save-crop", action="store_true", help="另存裁剪图")
    ap.add_argument("--save-viz", action="store_true", default=True, help="保存画框图")
    args = ap.parse_args()

    images = list_images(args.source)
    if not images:
        raise SystemExit(f"无图片: {args.source}")

    os.makedirs(args.out_dir, exist_ok=True)
    crop_dir = os.path.join(args.out_dir, "crops")
    viz_dir = os.path.join(args.out_dir, "viz")
    if args.save_crop:
        os.makedirs(crop_dir, exist_ok=True)
    if args.save_viz:
        os.makedirs(viz_dir, exist_ok=True)

    model = YOLO(args.weights)
    results_meta = []

    for path in images:
        img = imread_bgr(path)
        ih, iw = img.shape[:2]
        pred = model.predict(img, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        stem = Path(path).stem
        entry = {"image": path, "boxes": []}

        if pred.boxes is None or len(pred.boxes) == 0:
            print(f"[miss] {stem}")
            results_meta.append(entry)
            continue

        # 取置信度最高的一个 spine 框
        boxes = pred.boxes
        confs = boxes.conf.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy()
        i = int(np.argmax(confs))
        xyxy = expand_xyxy(xyxys[i], iw, ih, margin_x=args.margin_x, margin_y=args.margin_y)
        conf = float(confs[i])
        entry["boxes"].append({"xyxy": list(xyxy), "conf": conf})
        results_meta.append(entry)
        print(f"[ok] {stem}  conf={conf:.3f}  box={xyxy}")

        if args.save_viz:
            viz = img.copy()
            x0, y0, x1, y1 = xyxy
            cv2.rectangle(viz, (x0, y0), (x1, y1), (0, 255, 128), 3)
            cv2.putText(viz, f"spine {conf:.2f}", (x0, max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
            imwrite(os.path.join(viz_dir, f"{stem}_roi.jpg"), viz)

        if args.save_crop:
            x0, y0, x1, y1 = xyxy
            crop = img[y0:y1, x0:x1]
            if crop.size > 0:
                imwrite(os.path.join(crop_dir, f"{stem}_spine.jpg"), crop)

    meta_path = os.path.join(args.out_dir, "predictions.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(results_meta, fh, indent=2, ensure_ascii=False)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
