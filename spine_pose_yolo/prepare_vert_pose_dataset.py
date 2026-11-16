"""
制作「椎体实例 × 4 角点」YOLO-Pose 数据集（比 1×68 点更稳）。

主数据：AASCE_integrated（高质量，box + 角点）
辅数据：Spinal-AI2024（只进 train；默认角点 visibility=0，仅监督检测框）
分源加权：``--aasce-repeat`` 过采样 AASCE，避免 pose 被粗标淹没。

    python spine_pose_yolo/prepare_vert_pose_dataset.py --single-class
    python spine_pose_yolo/prepare_vert_pose_dataset.py --single-class --aasce-repeat 5 --spinal-kpt-vis 0 --out-dir spine_pose_yolo/datasets/vert_pose1_balanced
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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_AASCE = r"E:\脊柱侧弯\AASCE_integrated\images\train"
DEFAULT_SPINAL = r"E:\脊柱侧弯\Spinal-AI2024"
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "datasets", "vert_pose4")
DEFAULT_ALIGNED_PTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs",
    "test98_aligned_pts",
)
# 17 classes: T1..T12, L1..L5（自上而下）
VERT_NAMES = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]
assert len(VERT_NAMES) == 17


def read_wh(path: Path) -> tuple[int, int]:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    return w, h


def normalize_quad(pts4: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts4, dtype=np.float64).reshape(4, 2)
    order = np.argsort(pts[:, 1])
    top, bot = pts[order[:2]], pts[order[2:]]
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]
    return np.stack([top[0], top[1], bot[0], bot[1]], axis=0)  # TL,TR,BL,BR


def quad_ok(pts: np.ndarray, iw: int, ih: int, strict: bool = True) -> bool:
    """``strict=True`` for AASCE GT; ``False`` for Spinal-AI rough quads."""
    if not np.isfinite(pts).all():
        return False
    pad = 4.0 if not strict else 2.0
    if (pts[:, 0] < -pad).any() or (pts[:, 0] > iw + pad).any():
        return False
    if (pts[:, 1] < -pad).any() or (pts[:, 1] > ih + pad).any():
        return False
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    min_area = 4.0 if not strict else 6.0
    if area < min_area:
        return False
    d = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    if (d < (1.0 if not strict else 1.5)).any():
        return False
    if strict:
        if not (pts[0, 0] <= pts[1, 0] + 1e-3 and pts[2, 0] <= pts[3, 0] + 1e-3):
            return False
        if not (pts[0, 1] <= pts[2, 1] + 1e-3 and pts[1, 1] <= pts[3, 1] + 1e-3):
            return False
    return True


def instance_line(quad: np.ndarray, iw: int, ih: int, margin: float, class_id: int) -> str:
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
    parts = [f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]
    for x, y in q:
        parts.append(f"{np.clip(x / iw, 0, 1):.6f} {np.clip(y / ih, 0, 1):.6f} 2")
    return " ".join(parts)


def unique_name(path: Path, used: set[str], prefix: str = "") -> str:
    name = f"{prefix}{path.name}" if prefix else path.name
    if name not in used:
        used.add(name)
        return name
    stem, suf = Path(name).stem, Path(name).suffix
    i = 2
    while True:
        cand = f"{stem}__{i}{suf}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


def set_kpt_visibility(label_text: str, vis: int) -> str:
    """Rewrite YOLO-Pose keypoint visibility (x y v) while keeping boxes."""
    vis = int(vis)
    lines_out: list[str] = []
    for line in label_text.strip().splitlines():
        parts = line.split()
        # class cx cy w h + 4 * (x y v)
        if len(parts) < 5 + 4 * 3:
            lines_out.append(line)
            continue
        out = parts[:5]
        for i in range(4):
            base = 5 + i * 3
            out.extend([parts[base], parts[base + 1], str(vis)])
        lines_out.append(" ".join(out))
    return ("\n".join(lines_out) + "\n") if lines_out else ""


def parse_challenge_id(stem: str) -> int | None:
    """``01-July-2019-27`` -> 27."""
    tail = stem.split("-")[-1]
    return int(tail) if tail.isdigit() else None


def load_aligned_pts_original(img: Path, aligned_dir: Path) -> np.ndarray | None:
    """Aligned pts are in letterbox space → map back to original image pixels."""
    here = os.path.dirname(os.path.abspath(__file__))
    yolo_root = os.path.dirname(here)
    pix2point = os.path.join(os.path.dirname(yolo_root), "Pix2Point")
    for p in (pix2point, yolo_root):
        if p and p not in sys.path and os.path.isdir(p):
            sys.path.insert(0, p)
    from align_test98_gui import load_gray, load_pts, pts_letterbox_to_original, to_letterbox_gray

    pts_path = aligned_dir / f"{img.stem}_corner.pts"
    if not pts_path.is_file():
        return None
    gray = load_gray(str(img))
    _, meta = to_letterbox_gray(gray)
    pts_lb = load_pts(str(pts_path))
    return pts_letterbox_to_original(pts_lb, meta)


def load_saved_extra_stems(transforms_json: Path | None) -> set[str] | None:
    """If present, only keep stems marked saved=True in align GUI export."""
    if transforms_json is None or not transforms_json.is_file():
        return None
    raw = json.loads(transforms_json.read_text(encoding="utf-8"))
    per = raw.get("per_image", raw) if isinstance(raw, dict) else {}
    saved = {k for k, v in per.items() if isinstance(v, dict) and v.get("saved")}
    return saved


def samples_from_pts(
    img: Path,
    pts: np.ndarray,
    margin: float,
    loose: bool,
    single_class: bool = False,
) -> str | None:
    pts = np.asarray(pts, dtype=np.float64).reshape(68, 2)
    iw, ih = read_wh(img)
    lines = []
    bad = 0
    for vi in range(17):
        qn = normalize_quad(pts[vi * 4 : (vi + 1) * 4])
        if not quad_ok(qn, iw, ih, strict=not loose):
            bad += 1
            if not loose:
                return None
            if not np.isfinite(qn).all():
                return None
        cid = 0 if single_class else vi
        lines.append(instance_line(qn, iw, ih, margin, class_id=cid))
    if loose:
        keep = len(lines) == 17 and bad <= 3
    else:
        keep = len(lines) == 17 and bad == 0
    return ("\n".join(lines) + "\n") if keep else None


def collect_aasce(
    src: Path,
    margin: float,
    loose: bool = True,
    id_min: int | None = None,
    id_max: int | None = None,
    aligned_pts_dir: Path | None = None,
    allow_stems: set[str] | None = None,
    single_class: bool = False,
) -> list[tuple[Path, str]]:
    """Return list of (image_path, label_text).

    AASCE 全身片只标 T1–L5：头颅/骨盆区域无框 = 自然负样本。
    对 test 挑战赛图：务必限制 id 范围，并用对齐后的 pts。
    """
    out = []
    skipped = 0
    for img in sorted(src.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        cid = parse_challenge_id(img.stem)
        if id_min is not None or id_max is not None:
            if cid is None:
                skipped += 1
                continue
            if id_min is not None and cid < id_min:
                skipped += 1
                continue
            if id_max is not None and cid > id_max:
                skipped += 1
                continue
        if allow_stems is not None and img.stem not in allow_stems:
            skipped += 1
            continue

        pts = None
        if aligned_pts_dir is not None:
            pts = load_aligned_pts_original(img, aligned_pts_dir)
        if pts is None:
            pts_path = img.with_name(img.stem + "_corner.pts")
            if not pts_path.is_file():
                skipped += 1
                continue
            arr = np.loadtxt(pts_path, dtype=np.float64)
            if arr.size != 136:
                skipped += 1
                continue
            pts = arr.reshape(68, 2)

        lab = samples_from_pts(img, pts, margin, loose, single_class=single_class)
        if lab is None:
            skipped += 1
        else:
            out.append((img, lab))
    print(f"  AASCE {src}: kept={len(out)} skipped={skipped}")
    return out


def resolve_spinal_image(root: Path, file_name: str) -> Path | None:
    try:
        idx = int(Path(file_name).stem)
    except ValueError:
        return None
    subset = (idx - 1) // 4000 + 1
    if subset < 1 or subset > 5:
        return None
    path = root / f"Spinal-AI2024-subset{subset}" / file_name
    return path if path.is_file() else None


def collect_spinal(
    root: Path,
    margin: float,
    max_samples: int,
    seed: int,
    single_class: bool = False,
) -> list[tuple[Path, str]]:
    """Spinal：正好 17 个椎体。多类时按 y 映射 T1..L5；单类时 class=0。"""
    ann_path = root / "Spinal_AI2024_train__annotation" / "Spinal_AI2024_train__annotation.json"
    if not ann_path.is_file():
        print(f"[warn] Spinal-AI2024 ann missing: {ann_path}")
        return []
    with open(ann_path, encoding="utf-8") as fh:
        coco = json.load(fh)
    by_img: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        by_img[int(a["image_id"])].append(a)

    candidates: list[tuple[Path, str]] = []
    n_seen = 0
    for im in coco["images"]:
        n_seen += 1
        file_name = im["file_name"]
        iw, ih = int(im["width"]), int(im["height"])
        path = resolve_spinal_image(root, file_name)
        if path is None:
            continue
        anns = sorted(by_img.get(int(im["id"]), []), key=lambda a: float(a.get("bbox", [0, 0, 0, 0])[1]))
        # 17 类需要稳定一一映射，只要正好 17
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
        inversions = sum(1 for i in range(16) if ys[i + 1] + 2.0 < ys[i])
        if inversions > 2:
            continue
        quads = [q for _, q in sorted(zip(ys, quads), key=lambda t: t[0])]
        lines = [
            instance_line(q, iw, ih, margin, class_id=(0 if single_class else i))
            for i, q in enumerate(quads)
        ]
        candidates.append((path, "\n".join(lines) + "\n"))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    print(f"  Spinal candidates (exact 17, soft quad): {len(candidates)} / images={n_seen}")
    if max_samples > 0 and len(candidates) > max_samples:
        candidates = candidates[:max_samples]
    return candidates


def write_yaml(out: Path, single_class: bool = False) -> Path:
    # TL<->TR, BL<->BR
    flip_idx = [1, 0, 3, 2]
    if single_class:
        names_block = "  0: vertebra"
        header = "# single class vertebra, 4 corners (TL,TR,BL,BR); assign T1..L5 by y-order at infer"
    else:
        names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(VERT_NAMES))
        header = "# 17 vertebra classes (T1..L5), each with 4 corners (TL,TR,BL,BR)"
    text = f"""{header}
path: {out.as_posix()}
train: images/train
val: images/val

names:
{names_block}

kpt_shape: [4, 3]
flip_idx: {flip_idx}
"""
    path = out / "data.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aasce-dir", default=DEFAULT_AASCE)
    ap.add_argument(
        "--aasce-extra-dir",
        action="append",
        default=None,
        help="额外 AASCE 全身图目录（可重复；仅标 T1-L5，头颅/骨盆作负样本）",
    )
    ap.add_argument(
        "--extra-id-max",
        type=int,
        default=30,
        help="额外 test 图只收编号 <= 此值（你校对过的前 30 张）；0=不限制",
    )
    ap.add_argument(
        "--aligned-pts-dir",
        default=DEFAULT_ALIGNED_PTS,
        help="对齐后的 letterbox pts 目录（会映射回原图像素）",
    )
    ap.add_argument(
        "--extra-only-saved",
        action="store_true",
        help="额外图仅用 align GUI 里 saved=True 的样本（默认关：用编号<=extra-id-max）",
    )
    ap.add_argument("--spinal-root", default=DEFAULT_SPINAL)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--val-ratio", type=float, default=0.12, help="仅从主 AASCE 划 val")
    ap.add_argument("--spinal-max", type=int, default=6000, help="最多掺入多少张 Spinal-AI（0=不用）")
    ap.add_argument("--box-margin", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-spinal", action="store_true")
    ap.add_argument(
        "--aasce-repeat",
        type=int,
        default=1,
        help="AASCE(+extra) train 过采样倍数；分源加权建议 5",
    )
    ap.add_argument(
        "--spinal-kpt-vis",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="Spinal 角点 visibility；0=不监督 pose（仅学框），2=与 AASCE 相同",
    )
    ap.add_argument(
        "--single-class",
        action="store_true",
        help="单类 vertebra；推理时再按 y 序赋 T1..L5 并约束数量=17",
    )
    args = ap.parse_args()
    sc = bool(args.single_class)
    aasce_repeat = max(1, int(args.aasce_repeat))
    spinal_kpt_vis = int(args.spinal_kpt_vis)

    aasce = collect_aasce(Path(args.aasce_dir), args.box_margin, loose=True, single_class=sc)
    print(f"AASCE primary: {len(aasce)}  single_class={sc}")
    if len(aasce) < 10:
        raise SystemExit("AASCE 可用样本太少")

    # 额外全身图：默认只收已校对编号 <=30，并用 aligned pts
    aligned_dir = Path(args.aligned_pts_dir) if args.aligned_pts_dir else None
    allow = None
    if args.extra_only_saved and aligned_dir is not None:
        allow = load_saved_extra_stems(aligned_dir / "per_image_transforms.json")
        if allow is not None:
            print(f"  align saved stems: {len(allow)}")
    extra_aasce: list[tuple[Path, str]] = []
    for d in args.aasce_extra_dir or []:
        extra_aasce.extend(
            collect_aasce(
                Path(d),
                args.box_margin,
                loose=True,
                id_min=1 if args.extra_id_max > 0 else None,
                id_max=args.extra_id_max if args.extra_id_max > 0 else None,
                aligned_pts_dir=aligned_dir,
                allow_stems=allow,
                single_class=sc,
            )
        )
    print(f"AASCE extra(full-body, id<= {args.extra_id_max}): {len(extra_aasce)}")

    random.seed(args.seed)
    random.shuffle(aasce)
    n_val = max(1, int(round(len(aasce) * args.val_ratio)))
    if n_val >= len(aasce):
        n_val = 1
    val_items = aasce[:n_val]
    train_items = aasce[n_val:] + extra_aasce

    spinal_items: list[tuple[Path, str]] = []
    if not args.no_spinal and args.spinal_max > 0:
        spinal_items = collect_spinal(
            Path(args.spinal_root), args.box_margin, args.spinal_max, args.seed,
            single_class=sc,
        )
        print(f"Spinal-AI2024 used: {len(spinal_items)} (cap={args.spinal_max})")
        train_items = train_items + spinal_items
    else:
        print("Spinal-AI2024: skipped")

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    def dump(items: list[tuple[Path, str]], split: str, prefix: str, used: set[str] | None = None) -> int:
        if used is None:
            used = set()
        n = 0
        for img, lab in items:
            name = unique_name(img, used, prefix=prefix)
            shutil.copy2(img, out / "images" / split / name)
            (out / "labels" / split / (Path(name).stem + ".txt")).write_text(lab, encoding="utf-8")
            n += 1
        return n

    used_train: set[str] = set()
    aasce_train_base = aasce[n_val:]
    n_aasce_written = 0
    for r in range(aasce_repeat):
        pref = "aasce_" if r == 0 else f"aasce_r{r}_"
        n_aasce_written += dump(aasce_train_base, "train", prefix=pref, used=used_train)
    n_extra_written = 0
    for r in range(aasce_repeat):
        pref = "aasceX_" if r == 0 else f"aasceX_r{r}_"
        n_extra_written += dump(extra_aasce, "train", prefix=pref, used=used_train)

    spinal_labs = spinal_items
    if spinal_items and spinal_kpt_vis != 2:
        spinal_labs = [(img, set_kpt_visibility(lab, spinal_kpt_vis)) for img, lab in spinal_items]
    n_spinal_written = dump(spinal_labs, "train", prefix="spinal_", used=used_train)
    n_val_files = dump(val_items, "val", prefix="aasce_")

    yaml_path = write_yaml(out, single_class=sc)
    n_train = len(list((out / "images" / "train").glob("*")))
    print(
        f"train images: {n_train}  "
        f"(AASCE {len(aasce_train_base)}×{aasce_repeat}={n_aasce_written} + "
        f"AASCE_extra {len(extra_aasce)}×{aasce_repeat}={n_extra_written} + "
        f"Spinal {n_spinal_written}, kpt_vis={spinal_kpt_vis})"
    )
    print(f"val images: {n_val_files}  (AASCE only)")
    print(f"data.yaml: {yaml_path}  mode={'single' if sc else '17-class'}")


if __name__ == "__main__":
    main()
