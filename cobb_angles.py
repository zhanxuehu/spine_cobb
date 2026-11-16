"""
Cobb 角计算与可视化。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from spine_pose_yolo.infer_vert_pose import VERT_NAMES, normalize_quad


# 参考图配色：上弯青、中弯品红、下弯绿（BGR）
CURVE_COLORS = (
    (255, 200, 0),   # cyan-ish
    (255, 0, 255),   # magenta
    (0, 220, 80),    # green
)


@dataclass
class CobbCurve:
    angle: float
    i_upper: int
    i_lower: int
    color: tuple[int, int, int]
    name: str = ""

    def label(self, names: list[str] | None = None) -> str:
        names = names or VERT_NAMES
        u = names[self.i_upper] if 0 <= self.i_upper < len(names) else str(self.i_upper)
        v = names[self.i_lower] if 0 <= self.i_lower < len(names) else str(self.i_lower)
        tag = f"{self.name} " if self.name else ""
        return f"{tag}{u}-{v}: {self.angle:.1f}°"


def quads_to_pts68(quads: list[np.ndarray]) -> np.ndarray:
    """每椎 TL,TR,BL,BR → 68×2（与 AASCE 角点顺序一致）。"""
    pts = []
    for q in quads:
        qn = normalize_quad(q)
        pts.extend([qn[0], qn[1], qn[2], qn[3]])
    return np.asarray(pts, dtype=np.float32)


def _is_s_curve(mid_p_v: np.ndarray) -> bool:
    num = mid_p_v.shape[0]
    if num < 3:
        return False
    ll = []
    for i in range(num - 2):
        den_y = mid_p_v[0, 1] - mid_p_v[num - 1, 1]
        den_x = mid_p_v[0, 0] - mid_p_v[num - 1, 0]
        if abs(den_y) < 1e-6 or abs(den_x) < 1e-6:
            ll.append(0.0)
            continue
        term1 = (mid_p_v[i, 1] - mid_p_v[num - 1, 1]) / den_y
        term2 = (mid_p_v[i, 0] - mid_p_v[num - 1, 0]) / den_x
        ll.append(term1 - term2)
    ll = np.asarray(ll, np.float32)[:, None]
    ll_pair = ll @ ll.T
    a = float(ll_pair.sum())
    b = float(np.abs(ll_pair).sum())
    return abs(a - b) >= 1e-4


def compute_cobb_curves(
    quads: list[np.ndarray],
    image_height: int | None = None,
) -> list[CobbCurve]:
    """
    基于 AASCE 官方算法（CurvNet/MICCAI 移植）求最多 3 个弯。
    需要接近 17 个椎体四边形；少于 2 个返回空。
    """
    if len(quads) < 2:
        return []

    pts = quads_to_pts68(quads)
    num_pts = pts.shape[0]
    vnum = num_pts // 4 - 1  # 16 when 17 verts
    h = int(image_height) if image_height is not None else int(np.ceil(float(pts[:, 1].max()) + 1))

    mid_p_v = (pts[0::2, :] + pts[1::2, :]) / 2
    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i, :] + pts[i + 2, :]) / 2
        pt2 = (pts[i + 1, :] + pts[i + 3, :]) / 2
        mid_p.append(pt1)
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)

    vec_m = mid_p[1::2, :] - mid_p[0::2, :]
    dot_v = vec_m @ vec_m.T
    mod_v = np.sqrt(np.sum(vec_m ** 2, axis=1))[:, None]
    mod_v = mod_v @ mod_v.T
    cosine = np.zeros_like(dot_v)
    mask = mod_v != 0.0
    cosine[mask] = np.clip(dot_v[mask] / mod_v[mask], 0.0, 1.0)
    angles = np.arccos(cosine)

    pos1 = np.argmax(angles, axis=1)
    maxt = np.amax(angles, axis=1)
    pos2 = int(np.argmax(maxt))
    cobb1 = float(np.amax(maxt) / np.pi * 180.0)

    flag_s = _is_s_curve(mid_p_v)
    pairs: list[tuple[int, int, float]] = []

    if not flag_s:
        cobb2 = float(angles[0, pos2] / np.pi * 180.0)
        cobb3 = float(angles[vnum, pos1[pos2]] / np.pi * 180.0)
        pairs = [
            (0, pos2, cobb2),
            (pos2, int(pos1[pos2]), cobb1),
            (int(pos1[pos2]), vnum, cobb3),
        ]
    else:
        # 主弯是否偏上：两端 y 之和 < 图像高
        if (mid_p_v[pos2 * 2, 1] + mid_p_v[pos1[pos2] * 2, 1]) < h:
            angle2 = angles[pos2, : (pos2 + 1)]
            cobb2 = float(np.max(angle2) / np.pi * 180.0)
            pos1_1 = int(np.argmax(angle2))
            angle3 = angles[pos1[pos2], pos1[pos2] : (vnum + 1)]
            cobb3 = float(np.max(angle3) / np.pi * 180.0)
            pos1_2 = int(np.argmax(angle3)) + int(pos1[pos2]) - 1
            pos1_2 = int(np.clip(pos1_2, 0, vnum))
            pairs = [
                (pos1_1, pos2, cobb2),
                (pos2, int(pos1[pos2]), cobb1),
                (int(pos1[pos2]), pos1_2, cobb3),
            ]
        else:
            angle2 = angles[pos2, : (pos2 + 1)]
            cobb2 = float(np.max(angle2) / np.pi * 180.0)
            pos1_1 = int(np.argmax(angle2))
            angle3 = angles[pos1_1, : (pos1_1 + 1)]
            cobb3 = float(np.max(angle3) / np.pi * 180.0)
            pos1_2 = int(np.argmax(angle3))
            pairs = [
                (pos1_2, pos1_1, cobb3),
                (pos1_1, pos2, cobb2),
                (pos2, int(pos1[pos2]), cobb1),
            ]

    # 按上→下排序，命名 PT / MT / TL
    cleaned: list[tuple[int, int, float]] = []
    for a, b, ang in pairs:
        i0, i1 = (a, b) if a <= b else (b, a)
        if i0 == i1:
            continue
        if ang < 0.5:  # 忽略几乎共线
            continue
        cleaned.append((i0, i1, float(round(ang, 1))))
    cleaned.sort(key=lambda t: (t[0] + t[1]) / 2.0)

    names = ("上弯", "主弯", "下弯")
    curves: list[CobbCurve] = []
    for k, (i0, i1, ang) in enumerate(cleaned[:3]):
        curves.append(
            CobbCurve(
                angle=ang,
                i_upper=i0,
                i_lower=i1,
                color=CURVE_COLORS[k % len(CURVE_COLORS)],
                name=names[k] if k < len(names) else f"弯{k+1}",
            )
        )
    # 主弯标成角度最大的那条
    if curves:
        main_i = int(np.argmax([c.angle for c in curves]))
        for i, c in enumerate(curves):
            if i == main_i:
                c.name = "主弯"
            elif c.name == "主弯":
                c.name = "次弯"
    return curves


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _extend_line(p0: np.ndarray, p1: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    d = _unit(p1 - p0)
    mid = 0.5 * (p0 + p1)
    return mid - d * length, mid + d * length


def _half_endplate_line(
    p0: np.ndarray,
    p1: np.ndarray,
    side: float,
    extend: float,
) -> tuple[np.ndarray, np.ndarray]:
    """终板线只向量角那一侧延长（另一侧停在椎体边缘）。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = _unit(p1 - p0)
    side_vec = np.array([side, 0.0], dtype=np.float64)
    # 朝向量角侧的那一端再向外延伸
    if np.dot(d, side_vec) >= 0:
        inner, outer = p0, p1
        out_dir = d
    else:
        inner, outer = p1, p0
        out_dir = -d
    return inner, outer + out_dir * extend


def _endplate_foot_on_side(
    p0: np.ndarray,
    p1: np.ndarray,
    side: float,
    t: float = 0.72,
) -> np.ndarray:
    """终板垂足：从中点沿终板移到量角侧（与交点同侧）。

    t=0 中点，t=1 量角侧端点；默认偏外侧，避免交角落在椎体正中。
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    mid = 0.5 * (p0 + p1)
    d = _unit(p1 - p0)
    if float(np.dot(d, np.array([side, 0.0], dtype=np.float64))) < 0:
        d = -d
    half = 0.5 * float(np.linalg.norm(p1 - p0))
    return mid + d * (half * float(np.clip(t, 0.0, 1.0)))


def _dashed_line(img, p0, p1, color, thick, dash=12, gap=8):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    dist = float(np.linalg.norm(p1 - p0))
    if dist < 1:
        return
    d = (p1 - p0) / dist
    n_seg = int(dist // (dash + gap)) + 1
    for i in range(n_seg):
        a = p0 + d * (i * (dash + gap))
        b = p0 + d * min(dist, i * (dash + gap) + dash)
        cv2.line(
            img,
            (int(a[0]), int(a[1])),
            (int(b[0]), int(b[1])),
            color,
            thick,
            cv2.LINE_AA,
        )


def _line_intersection(p1, d1, p2, d2):
    """p + t d 直线交点。"""
    A = np.array([d1, -d2], dtype=np.float64).T
    if abs(np.linalg.det(A)) < 1e-8:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


END_VERTEBRA_COLOR = (40, 40, 230)  # 参考图：终椎半透明红（BGR）


def _put_outlined_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thick: int,
) -> None:
    """彩色字 + 白色描边，贴近参考图可读性。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    outline = max(thick + 2, int(round(thick * 2.2)))
    cv2.putText(img, text, org, font, font_scale, (255, 255, 255), outline, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, thick, cv2.LINE_AA)


def _highlight_end_vertebra(canvas: np.ndarray, quad: np.ndarray, alpha: float = 0.35) -> None:
    qn = normalize_quad(quad)
    poly = np.array([qn[0], qn[1], qn[3], qn[2]], dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [poly], END_VERTEBRA_COLOR, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)
    mean_h = float(np.ptp(qn[:, 1]))
    r = max(2, int(round(mean_h / 9.0)))
    t = max(1, int(round(mean_h / 22.0)))
    cv2.polylines(canvas, [poly], True, END_VERTEBRA_COLOR, t, cv2.LINE_AA)
    for p in qn:
        cv2.circle(canvas, (int(p[0]), int(p[1])), r, END_VERTEBRA_COLOR, -1, cv2.LINE_AA)


def _curve_concave_side(quads: list[np.ndarray], i_upper: int, i_lower: int) -> float:
    """按该弯顶椎相对上下终椎连线的凸向，返回凹侧（量角侧）：-1 左 / +1 右。"""
    i0, i1 = (i_upper, i_lower) if i_upper <= i_lower else (i_lower, i_upper)
    centers = np.array(
        [normalize_quad(quads[i]).mean(axis=0) for i in range(i0, i1 + 1)],
        dtype=np.float64,
    )
    if len(centers) < 2:
        return -1.0

    p0, p1 = centers[0], centers[-1]
    chord = p1 - p0
    chord2 = float(np.dot(chord, chord))
    best_dev = 0.0
    mids = centers[1:-1] if len(centers) > 2 else centers
    for c in mids:
        if chord2 < 1e-6:
            dev = float(c[0] - p0[0])
        else:
            t = float(np.clip(np.dot(c - p0, chord) / chord2, 0.0, 1.0))
            proj = p0 + t * chord
            # 水平偏离：>0 顶椎偏右（右凸）
            dev = float(c[0] - proj[0])
        if abs(dev) > abs(best_dev):
            best_dev = dev

    if abs(best_dev) < 1e-3:
        # 几乎直：用段中点相对整段中点
        best_dev = float(centers[len(centers) // 2, 0] - 0.5 * (p0[0] + p1[0]))

    # 右凸 → 凹侧在左(-1)；左凸 → 凹侧在右(+1)
    return -1.0 if best_dev > 0 else 1.0


def _margin_on_side(width: int, ref_x: float, side: float) -> float:
    """量角侧到图像边缘的可用宽度。"""
    return float(ref_x) if side < 0 else float(width - ref_x)


def _pick_draw_side(width: int, ref_x: float, preferred: float, mean_w: float) -> tuple[float, float]:
    """优先凹侧；若该侧太窄则换到另一侧，并返回该侧可用边距。"""
    need = max(mean_w * 2.2, 36.0)
    m0 = _margin_on_side(width, ref_x, preferred)
    m1 = _margin_on_side(width, ref_x, -preferred)
    if m0 >= need or m0 >= m1:
        return preferred, m0
    return -preferred, m1


def _clip_pt(p: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.array(
        [float(np.clip(p[0], 0, w - 1)), float(np.clip(p[1], 0, h - 1))],
        dtype=np.float64,
    )


def _endplate_foot(p0: np.ndarray, p1: np.ndarray, side: float, ref_x: float) -> np.ndarray:
    """垂线脚点取终板偏向量角侧的位置，让交角离开椎体。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    mid = 0.5 * (p0 + p1)
    outer = p0 if (p0[0] - ref_x) * side >= (p1[0] - ref_x) * side else p1
    return 0.72 * outer + 0.28 * mid


def _best_perp_intersection(
    fu: np.ndarray,
    fl: np.ndarray,
    nu: np.ndarray,
    nl: np.ndarray,
    side: float,
    ref_x: float,
    w: int,
    h: int,
    min_lateral: float,
):
    """
    在 ±法向上选真实交点：必须在量角侧、离开脊柱足够远，形成图2那种侧方 X 交角。
    返回 (hit, nu_used, nl_used) 或 None。
    """
    y0, y1 = float(min(fu[1], fl[1])), float(max(fu[1], fl[1]))
    span = max(y1 - y0, 1.0)
    cands = []
    for su in (1.0, -1.0):
        for sl in (1.0, -1.0):
            nu_u = su * nu
            nl_u = sl * nl
            hit = _line_intersection(fu, nu_u, fl, nl_u)
            if hit is None:
                continue
            # 垂线应朝交点延伸（t>=0）
            if np.dot(hit - fu, nu_u) < -1e-3 or np.dot(hit - fl, nl_u) < -1e-3:
                continue
            lat = (hit[0] - ref_x) * side  # >0 在量角侧
            if lat < min_lateral * 0.35:
                continue  # 交点压在椎体上/凸侧，直接丢弃
            side_score = 80.0 + min(lat, min_lateral * 2.0) * 0.4
            y_ok = (y0 - 0.2 * span) <= hit[1] <= (y1 + 0.2 * span)
            y_score = 30.0 if y_ok else -25.0
            dist = 0.5 * (float(np.linalg.norm(hit - fu)) + float(np.linalg.norm(hit - fl)))
            dist_score = -abs(dist - span * 0.65) * 0.025
            dist_score -= max(0.0, dist - span * 2.0) * 0.06
            in_img = (8 <= hit[0] < w - 8) and (8 <= hit[1] < h - 8)
            img_score = 30.0 if in_img else -150.0
            edge_pen = 0.0
            if hit[0] < 0.05 * w or hit[0] > 0.95 * w:
                edge_pen -= 40.0
            score = side_score + y_score + dist_score + img_score + edge_pen
            cands.append((score, hit, nu_u, nl_u, dist))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0][1], cands[0][2], cands[0][3]


def _place_angle_at_corner(
    canvas: np.ndarray,
    angle: float,
    hit: np.ndarray,
    fu: np.ndarray,
    fl: np.ndarray,
    side: float,
    ref_x: float,
    font_scale: float,
    font_thick: int,
    color: tuple[int, int, int],
) -> None:
    """度数画在两终板垂足连线的正中（角两边中间）；OpenCV 无 °，用数字 + 右上小圆。

    小角度时真交点很远、可见 apex 常贴在一端，不能再沿「交点→角平分线」放字，
    否则会贴到上/下终板。改为始终取两垂足中点，再略向量角侧/交点挪一点。
    """
    h, w = canvas.shape[:2]
    scale = float(np.clip(min(font_scale, w / 240.0, h / 900.0), 0.4, 1.5))
    thick = max(1, int(round(scale * 2.0)))
    outline = max(thick + 2, int(round(thick * 2.2)))
    text = f"{angle:.1f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = max(4, thick + 2)
    deg_r = max(2, int(round(scale * 3.2)))
    deg_gap = max(1, outline // 2)

    hit = np.asarray(hit, dtype=np.float64)
    fu = np.asarray(fu, dtype=np.float64)
    fl = np.asarray(fl, dtype=np.float64)

    # 角两边中间 = 上下垂足的中点（竖直/水平都居中）
    mid = 0.5 * (fu + fl)
    span = float(np.linalg.norm(fl - fu))
    to_hit = hit - mid
    n_hit = float(np.linalg.norm(to_hit))
    # 略离开中点：朝交点（量角开口内），幅度跟两终板间距挂钩，避免贴虚线交点或椎体
    if n_hit > 1e-3:
        nudge = float(np.clip(0.18 * span, 8.0, max(8.0, 0.28 * span)))
        # 交点若几乎贴在某一垂足，to_hit 不可靠，改为量角侧水平推开
        foot_dist = min(float(np.linalg.norm(hit - fu)), float(np.linalg.norm(hit - fl)))
        if foot_dist < 0.25 * max(span, 1.0):
            dir_n = np.array([side, 0.0], dtype=np.float64)
        else:
            dir_n = to_hit / n_hit
        anchor = mid + dir_n * nudge
    else:
        anchor = mid + np.array([side * max(10.0, 0.15 * span), 0.0], dtype=np.float64)

    block_w = tw + outline + deg_gap + deg_r * 2 + 2
    # 文字块中心对齐 anchor（OpenCV org 是左下基线）
    tx = int(round(anchor[0] - 0.5 * tw))
    ty = int(round(anchor[1] + 0.35 * th))
    tx = int(np.clip(tx, pad, max(pad, w - block_w - pad)))
    ty = int(np.clip(ty, th + pad, h - pad))
    # 若压到脊柱中线，向量角侧再推
    if side > 0 and tx < ref_x + 6:
        tx = int(min(w - block_w - pad, max(tx, int(ref_x + 10))))
    if side < 0 and tx + tw > ref_x - 6:
        tx = int(max(pad, min(tx, int(ref_x - tw - 14))))

    _put_outlined_text(canvas, text, (tx, ty), scale, color, thick)
    cx = tx + tw + deg_gap + deg_r
    cy = ty - th + deg_r + max(0, baseline)
    cx = int(np.clip(cx, deg_r + 1, w - deg_r - 1))
    cy = int(np.clip(cy, deg_r + 1, h - deg_r - 1))
    cv2.circle(canvas, (cx, cy), deg_r, (255, 255, 255), max(thick + 1, 2), cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), deg_r, color, max(1, thick), cv2.LINE_AA)


def _spine_x_bounds(quads: list[np.ndarray]) -> tuple[float, float]:
    xs = np.concatenate([normalize_quad(q)[:, 0] for q in quads])
    return float(xs.min()), float(xs.max())


def _orient_normal_to_side(n: np.ndarray, side: float) -> np.ndarray:
    n = _unit(n)
    if n[0] * side < 0:
        n = -n
    # 终板接近水平时法线接近竖直，补一点侧向分量，保证能量角侧离开脊柱
    if abs(n[0]) < 0.20:
        n = _unit(n + np.array([side * 0.55, 0.0], dtype=np.float64))
    return n


def _ray_hit_x(p: np.ndarray, d: np.ndarray, x_line: float) -> np.ndarray:
    """从 p 沿 d 走到竖直线 x=x_line（若反向则翻转 d）。"""
    d = _unit(d)
    if abs(d[0]) < 1e-6:
        return np.array([x_line, float(p[1])], dtype=np.float64)
    if d[0] * (x_line - p[0]) < 0:
        d = -d
    t = (x_line - p[0]) / d[0]
    return p + d * max(t, 0.0)


def _visible_apex(fu: np.ndarray, fl: np.ndarray, hit: np.ndarray, w: int, h: int) -> np.ndarray:
    """交点若在图外，沿「脚点中点→交点」缩到画布内，避免独立 clamp x/y 把圆点挪歪。"""
    if 0 <= hit[0] < w and 0 <= hit[1] < h:
        return np.asarray(hit, dtype=np.float64)
    mid = 0.5 * (fu + fl)
    to = hit - mid
    if float(np.linalg.norm(to)) < 1e-6:
        return _clip_pt(hit, w, h)
    lo, hi = 0.0, 1.0
    best = _clip_pt(mid, w, h)
    for _ in range(24):
        t = 0.5 * (lo + hi)
        p = mid + to * t
        if 1 <= p[0] < w - 2 and 1 <= p[1] < h - 2:
            best = p
            lo = t
        else:
            hi = t
    return np.asarray(best, dtype=np.float64)


def draw_cobb_curves(
    bgr: np.ndarray,
    quads: list[np.ndarray],
    curves: list[CobbCurve],
    show_text: bool = True,
) -> np.ndarray:
    """终板实线 + 垂线交角；垂足放在量角侧（与交点同侧），避免交角落在椎体正中。"""
    if not curves or not quads:
        return bgr
    canvas = bgr
    h, w = canvas.shape[:2]
    mean_w = float(np.mean([float(np.ptp(normalize_quad(q)[:, 0])) for q in quads]))
    thick = max(2, int(round(mean_w / 16.0)))
    dash = max(5, int(round(mean_w / 3.5)))
    gap = max(4, int(round(dash * 0.75)))
    font_scale = float(np.clip(min(mean_w / 26.0, w / 200.0), 0.45, 1.5))
    spine_xmin, spine_xmax = _spine_x_bounds(quads)
    clear = max(mean_w * 0.85, 22.0)

    for curve in curves:
        if not (0 <= curve.i_upper < len(quads) and 0 <= curve.i_lower < len(quads)):
            continue
        qu = normalize_quad(quads[curve.i_upper])
        ql = normalize_quad(quads[curve.i_lower])
        color = curve.color

        preferred = _curve_concave_side(quads, curve.i_upper, curve.i_lower)
        ref_x = 0.5 * (qu[:, 0].mean() + ql[:, 0].mean())
        side, margin = _pick_draw_side(w, ref_x, preferred, mean_w)
        if side > 0 and (w - spine_xmax) < clear + 8:
            side = -1.0
            margin = spine_xmin
        elif side < 0 and spine_xmin < clear + 8:
            side = 1.0
            margin = w - spine_xmax

        extend = float(
            np.clip(min(mean_w * 2.0, margin * 0.5), mean_w * 0.7, max(mean_w, margin * 0.65))
        )

        # 终板实线（只向量角侧）
        u0, u1 = _half_endplate_line(qu[0], qu[1], side, extend)
        l0, l1 = _half_endplate_line(ql[2], ql[3], side, extend)
        cv2.line(
            canvas,
            (int(u0[0]), int(u0[1])),
            (int(u1[0]), int(u1[1])),
            color,
            thick,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (int(l0[0]), int(l0[1])),
            (int(l1[0]), int(l1[1])),
            color,
            thick,
            cv2.LINE_AA,
        )

        # 垂足：落在量角侧终板线上（偏外侧），与交点同侧，不再用中点
        fu = _endplate_foot_on_side(qu[0], qu[1], side, t=0.78)
        fl = _endplate_foot_on_side(ql[2], ql[3], side, t=0.78)
        # 再沿已画终板往外贴一点，确保垂足在可见线段上
        fu = 0.4 * fu + 0.6 * np.asarray(u1, dtype=np.float64)
        fl = 0.4 * fl + 0.6 * np.asarray(l1, dtype=np.float64)
        du = _unit(np.asarray(qu[1], dtype=np.float64) - np.asarray(qu[0], dtype=np.float64))
        dl = _unit(np.asarray(ql[3], dtype=np.float64) - np.asarray(ql[2], dtype=np.float64))
        # 投影回终板直线，保证垂线几何正确
        mid_u = 0.5 * (np.asarray(qu[0], dtype=np.float64) + np.asarray(qu[1], dtype=np.float64))
        mid_l = 0.5 * (np.asarray(ql[2], dtype=np.float64) + np.asarray(ql[3], dtype=np.float64))
        fu = mid_u + du * float(np.dot(fu - mid_u, du))
        fl = mid_l + dl * float(np.dot(fl - mid_l, dl))

        nu0 = np.array([-du[1], du[0]], dtype=np.float64)
        nl0 = np.array([-dl[1], dl[0]], dtype=np.float64)

        hit = None
        best_sc = -1e18
        for su in (1.0, -1.0):
            for sl in (1.0, -1.0):
                nu_u, nl_u = su * nu0, sl * nl0
                pt = _line_intersection(fu, nu_u, fl, nl_u)
                if pt is None:
                    continue
                if np.dot(pt - fu, nu_u) < -1e-6 or np.dot(pt - fl, nl_u) < -1e-6:
                    continue
                # 交点必须在量角侧（与垂足同侧）
                lat = float((pt[0] - ref_x) * side)
                foot_lat = 0.5 * (
                    float((fu[0] - ref_x) * side) + float((fl[0] - ref_x) * side)
                )
                sc = lat + (800.0 if lat >= foot_lat * 0.5 else -800.0)
                y0, y1 = float(min(fu[1], fl[1])), float(max(fu[1], fl[1]))
                span = y1 - y0 + 1.0
                if y0 - 0.35 * span <= pt[1] <= y1 + 0.35 * span:
                    sc += 80.0
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    sc += 120.0
                sc -= 0.02 * float(np.linalg.norm(pt - 0.5 * (fu + fl)))
                if sc > best_sc:
                    best_sc = sc
                    hit = np.asarray(pt, dtype=np.float64)

        if hit is None:
            nu = _orient_normal_to_side(nu0, side)
            nl = _orient_normal_to_side(nl0, side)
            tip_u = fu + nu * max(mean_w * 1.5, 30.0)
            tip_l = fl + nl * max(mean_w * 1.5, 30.0)
            hit = 0.5 * (tip_u + tip_l)
            _dashed_line(canvas, fu, tip_u, color, max(1, thick - 1), dash, gap)
            _dashed_line(canvas, fl, tip_l, color, max(1, thick - 1), dash, gap)
        else:
            _dashed_line(canvas, fu, hit, color, max(1, thick - 1), dash, gap)
            _dashed_line(canvas, fl, hit, color, max(1, thick - 1), dash, gap)

        apex = _visible_apex(fu, fl, hit, w, h)
        cv2.circle(
            canvas,
            (int(round(apex[0])), int(round(apex[1]))),
            max(2, thick),
            color,
            -1,
            cv2.LINE_AA,
        )

        if show_text:
            # 用统一布局：数字贴在交角外侧，° 紧跟数字右上，避免压到椎体名
            _place_angle_at_corner(
                canvas,
                curve.angle,
                apex,
                fu,
                fl,
                side,
                ref_x,
                font_scale,
                thick,
                color,
            )
    return canvas


def format_cobb_summary(curves: list[CobbCurve], names: list[str] | None = None) -> str:
    if not curves:
        return "Cobb: 无"
    return "  |  ".join(c.label(names) for c in curves)
