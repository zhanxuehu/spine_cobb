from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from cobb_angles import (  # noqa: E402
    compute_cobb_curves,
    draw_cobb_curves,
    format_cobb_summary,
)
from robust_spine_infer import robust_pose_infer  # noqa: E402
from spine_pose_yolo.cascade_infer import predict_cascade  # noqa: E402
from spine_pose_yolo.infer_retrain_cascade import (  # noqa: E402
    load_corner_model,
    predict_retrain_cascade,
)
from spine_pose_yolo.infer_vert_pose import draw, predict_one, resolve_weights  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from xray_preprocess import preprocess_xray, read_raw  # noqa: E402

# 自定义 Trainer 权重 unpickle 兼容
from spine_pose_yolo.train_corner_adj import AdjacentPoseModel  # noqa: E402,F401

# 默认：mix_v2 中心 + 局部四角回归；未就绪则回退旧 ctx
DEFAULT_DETECT_WEIGHTS = os.path.join(ROOT, "best_weights", "vert_center_mix_v2_best.pt")
if not os.path.isfile(DEFAULT_DETECT_WEIGHTS):
    DEFAULT_DETECT_WEIGHTS = os.path.join(
        ROOT, "spine_pose_yolo", "runs", "vert_center_mix_v2", "weights", "best.pt"
    )
if not os.path.isfile(DEFAULT_DETECT_WEIGHTS):
    DEFAULT_DETECT_WEIGHTS = os.path.join(ROOT, "best_weights", "vert_center_spinal_best.pt")
if not os.path.isfile(DEFAULT_DETECT_WEIGHTS):
    DEFAULT_DETECT_WEIGHTS = os.path.join(
        ROOT, "spine_pose_yolo", "runs", "vert_center_spinal", "weights", "best.pt"
    )
DEFAULT_REFINE_WEIGHTS = os.path.join(ROOT, "best_weights", "vert_corner_reg_best.pt")
if not os.path.isfile(DEFAULT_REFINE_WEIGHTS):
    DEFAULT_REFINE_WEIGHTS = os.path.join(
        ROOT, "spine_pose_yolo", "runs", "vert_corner_reg", "best.pt"
    )
if not os.path.isfile(DEFAULT_REFINE_WEIGHTS):
    DEFAULT_REFINE_WEIGHTS = os.path.join(ROOT, "best_weights", "vert_corner_ctx_best.pt")
if not os.path.isfile(DEFAULT_REFINE_WEIGHTS):
    DEFAULT_REFINE_WEIGHTS = os.path.join(
        ROOT, "spine_pose_yolo", "runs", "vert_corner_ctx", "weights", "best.pt"
    )
if not os.path.isfile(DEFAULT_REFINE_WEIGHTS):
    DEFAULT_REFINE_WEIGHTS = os.path.join(ROOT, "best_weights", "vert_corner_patch_best.pt")
DEFAULT_POSE_WEIGHTS = DEFAULT_DETECT_WEIGHTS

DEFAULT_ROI_WEIGHTS = os.path.join(ROOT, "best_weights", "spine_roi_best.pt")
if not os.path.isfile(DEFAULT_ROI_WEIGHTS):
    DEFAULT_ROI_WEIGHTS = os.path.join(
        ROOT, "spine_roi_yolo", "runs", "spine_roi_v2", "weights", "best.pt"
    )
    if not os.path.isfile(DEFAULT_ROI_WEIGHTS):
        DEFAULT_ROI_WEIGHTS = os.path.join(
            ROOT, "spine_roi_yolo", "runs", "spine_roi", "weights", "best.pt"
        )


def raw_to_bgr_u8(raw: np.ndarray) -> np.ndarray:
    """仅做显示用转换，不改变外观风格（16bit 线性缩放到 8bit）。"""
    if raw.ndim == 2:
        if raw.dtype != np.uint8:
            g = raw.astype(np.float32)
            lo, hi = float(g.min()), float(g.max())
            gray = (
                np.zeros(g.shape, dtype=np.uint8)
                if hi <= lo
                else np.clip((g - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            )
        else:
            gray = raw
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if raw.dtype != np.uint8:
        g = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = float(g.min()), float(g.max())
        gray = (
            np.zeros(g.shape, dtype=np.uint8)
            if hi <= lo
            else np.clip((g - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return raw


def _bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def centers_from_quads(quads) -> np.ndarray:
    """四角质心，用作非级联路径的中心回退。"""
    if not quads:
        return np.zeros((0, 2), dtype=np.float64)
    pts = []
    for q in quads:
        arr = np.asarray(q, dtype=np.float64).reshape(-1, 2)
        pts.append(arr.mean(axis=0))
    return np.asarray(pts, dtype=np.float64)


def smooth_center_curve(centers: np.ndarray, samples_per_seg: int = 10) -> list[tuple[float, float]]:
    """中心点序列 → 平滑折线（Catmull-Rom）。"""
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n == 0:
        return []
    if n <= 2:
        return [(float(pts[i, 0]), float(pts[i, 1])) for i in range(n)]
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    d = np.maximum(d, 1e-6)
    t = np.concatenate([[0.0], np.cumsum(d)])
    t = t / t[-1]
    out: list[tuple[float, float]] = []
    steps = max(2, (n - 1) * samples_per_seg)
    for k in range(steps + 1):
        u = k / steps
        j = int(np.searchsorted(t, u, side="right") - 1)
        j = int(np.clip(j, 0, n - 2))
        t0, t1 = t[j], t[j + 1]
        local = 0.0 if t1 <= t0 else (u - t0) / (t1 - t0)
        p0 = pts[max(j - 1, 0)]
        p1 = pts[j]
        p2 = pts[j + 1]
        p3 = pts[min(j + 2, n - 1)]
        local2 = local * local
        local3 = local2 * local
        p = 0.5 * (
            (2 * p1)
            + (-p0 + p2) * local
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * local2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * local3
        )
        out.append((float(p[0]), float(p[1])))
    return out


def draw_center_curve(bgr: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """画中心点十字 + 编号 + 青色中心曲线。"""
    vis = bgr
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return vis
    curve = smooth_center_curve(pts)
    if len(curve) >= 2:
        poly = np.array([[[int(round(x)), int(round(y))]] for x, y in curve], dtype=np.int32)
        cv2.polylines(vis, [poly], False, (0, 0, 0), 4, lineType=cv2.LINE_AA)
        cv2.polylines(vis, [poly], False, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    for i, (cx, cy) in enumerate(pts):
        x, y = int(round(cx)), int(round(cy))
        cv2.line(vis, (x - 6, y), (x + 6, y), (0, 220, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(vis, (x, y - 6), (x, y + 6), (0, 220, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(vis, (x, y), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            str(i + 1),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


class PillToggle(tk.Canvas):
    """可点击药丸开关：开=彩色实心，关=灰底描边。"""

    def __init__(
        self,
        master,
        text: str,
        accent_bgr: tuple[int, int, int],
        variable: tk.BooleanVar,
        command=None,
        **kwargs,
    ):
        self._text = text
        self._accent = _bgr_to_hex(accent_bgr)
        self._var = variable
        self._command = command
        self._pad_x = 14
        self._pad_y = 6
        kwargs.setdefault("highlightthickness", 0)
        kwargs.setdefault("bd", 0)
        kwargs.setdefault("cursor", "hand2")
        kwargs.setdefault("background", "#f0f0f0")
        super().__init__(master, height=28, **kwargs)
        self.bind("<Button-1>", self._on_click)
        self._var.trace_add("write", lambda *_: self._paint())
        self._paint()

    def _on_click(self, _event=None) -> None:
        self._var.set(not bool(self._var.get()))
        if self._command is not None:
            self._command()

    def _paint(self) -> None:
        on = bool(self._var.get())
        fg = "#ffffff" if on else "#555555"
        bg = self._accent if on else "#ececec"
        outline = self._accent if on else "#bdbdbd"
        self.delete("all")
        tmp = self.create_text(0, 0, text=self._text, font=("Segoe UI", 9, "bold"), anchor="nw")
        x0, y0, x1, y1 = self.bbox(tmp)
        self.delete(tmp)
        tw, th = x1 - x0, y1 - y0
        w = tw + self._pad_x * 2
        h = max(26, th + self._pad_y * 2)
        self.configure(width=w, height=h)
        # 胶囊外形：左右圆 + 中间矩形
        self.create_oval(0, 0, h, h, fill=bg, outline=outline, width=1)
        self.create_oval(w - h, 0, w, h, fill=bg, outline=outline, width=1)
        self.create_rectangle(h // 2, 1, w - h // 2, h - 1, fill=bg, outline=bg)
        self.create_line(h // 2, 0, w - h // 2, 0, fill=outline)
        self.create_line(h // 2, h - 1, w - h // 2, h - 1, fill=outline)
        self.create_text(w / 2, h / 2, text=self._text, fill=fg, font=("Segoe UI", 9, "bold"))


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("spine_cobb · 脊柱椎体定位与 Cobb 角")
        self.root.geometry("980x820")
        self.pose_weights = tk.StringVar(value=os.path.abspath(DEFAULT_DETECT_WEIGHTS))
        self.refine_weights = tk.StringVar(
            value=os.path.abspath(DEFAULT_REFINE_WEIGHTS) if os.path.isfile(DEFAULT_REFINE_WEIGHTS) else ""
        )
        self.image_path = tk.StringVar(value="")
        self.conf = tk.DoubleVar(value=0.15)
        self.use_cascade = tk.BooleanVar(value=True)
        self.use_multi_crop = tk.BooleanVar(value=False)
        self.show_boxes = tk.BooleanVar(value=True)
        self.show_centers = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="正在自动加载模型…")
        self.pose_model = None
        self.refine_model = None
        self.roi_model = None  # 仅后台裁剪候选，界面不暴露
        self._loaded_pose = None
        self._loaded_refine = None
        self._orig_bgr = None
        self._proc_bgr = None
        self._view_bgr = None
        self._last_quads = None
        self._last_names = None
        self._last_curves = None
        self._last_centers = None  # (N,2) 中心点模型预测 / 四角质心回退
        self._cobb_vars: list[tk.BooleanVar] = []
        self._cobb_pills: list[PillToggle] = []
        self._photo = None
        self._pil_rgb = None  # 缓存 RGB PIL，避免每次缩放都 BGR→RGB
        self._busy = False
        self._resize_job = None
        self._zoom_job = None
        self._prep_token = 0
        self._zoom = 1.0  # 滚轮缩放，1=适应窗口
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_last = None
        self._build()
        self.root.after(100, self.load_model)

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="检测权重").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.pose_weights).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(top, text="浏览…", command=self.browse_weights).grid(row=0, column=2)

        ttk.Label(top, text="角点权重").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(top, textvariable=self.refine_weights).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(top, text="浏览…", command=self.browse_refine_weights).grid(row=1, column=2)

        ttk.Label(top, text="图片").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(top, textvariable=self.image_path).grid(row=2, column=1, sticky="we", padx=4)
        ttk.Button(top, text="选择图片…", command=self.browse_image).grid(row=2, column=2)

        ttk.Label(top, text="conf").grid(row=3, column=0, sticky="w")
        conf_row = ttk.Frame(top)
        conf_row.grid(row=3, column=1, columnspan=2, sticky="w", padx=4)
        ttk.Entry(conf_row, textvariable=self.conf, width=8).pack(side=tk.LEFT)
        ttk.Checkbutton(
            conf_row,
            text="级联精修四角",
            variable=self.use_cascade,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(
            conf_row,
            text="多候选裁剪",
            variable=self.use_multi_crop,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(
            conf_row,
            text="显示推理框",
            variable=self.show_boxes,
            command=self._refresh_overlay,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(
            conf_row,
            text="显示中心线",
            variable=self.show_centers,
            command=self._refresh_overlay,
        ).pack(side=tk.LEFT, padx=(12, 0))

        btns = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="重新加载模型", command=self.load_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="开始推理", command=self.run_infer).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="保存结果…", command=self.save_result).pack(side=tk.LEFT, padx=4)

        cobb_bar = ttk.LabelFrame(self.root, text="Cobb 角（点击药丸开关绘制）", padding=(8, 4))
        cobb_bar.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._cobb_pills_row = ttk.Frame(cobb_bar)
        self._cobb_pills_row.pack(fill=tk.X, anchor="w")
        self._cobb_empty = ttk.Label(self._cobb_pills_row, text="推理后显示各弯开关")
        self._cobb_empty.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.root, background="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Enter>", lambda _e: self.canvas.focus_set())
        # 滚轮缩放（相对鼠标位置）；中键拖拽平移；中键双击复位
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom)  # Linux up
        self.canvas.bind("<Button-5>", self._on_zoom)  # Linux down
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<Double-Button-2>", self._reset_view)
        ttk.Label(
            self.root,
            textvariable=self.status,
            padding=8,
        ).pack(fill=tk.X)

    def browse_weights(self) -> None:
        path = filedialog.askopenfilename(
            title="选择检测权重 (Spinal/找椎体)",
            initialdir=os.path.dirname(self.pose_weights.get() or DEFAULT_DETECT_WEIGHTS),
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")],
        )
        if path:
            self.pose_weights.set(path)
            self.pose_model = None
            self._loaded_pose = None

    def browse_refine_weights(self) -> None:
        path = filedialog.askopenfilename(
            title="选择角点精修权重 (AASCE)",
            initialdir=os.path.dirname(self.refine_weights.get() or DEFAULT_REFINE_WEIGHTS or ROOT),
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")],
        )
        if path:
            self.refine_weights.set(path)
            self.refine_model = None
            self._loaded_refine = None

    def browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择脊柱 X 光片",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return
        self.image_path.set(path)
        try:
            raw = read_raw(path)
            orig = raw_to_bgr_u8(raw)
            self._orig_bgr = orig
            self._set_view_bgr(orig)
            self._proc_bgr = None
            self._last_quads = None
            self._last_names = None
            self._last_curves = None
            self._last_centers = None
            self._rebuild_cobb_pills([])
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            self.status.set(f"已加载 {os.path.basename(path)} | 滚轮缩放，中键拖拽，中键双击复位")
            self._redraw(fast=False)
            self._start_background_preprocess(raw)
        except Exception as exc:
            messagebox.showerror("错误", f"读图失败: {exc}")

    def _start_background_preprocess(self, raw: np.ndarray) -> None:
        self._prep_token += 1
        token = self._prep_token

        def work():
            try:
                proc, _info = preprocess_xray(raw)
                self.root.after(0, lambda: self._on_prep_done(token, proc))
            except Exception:
                self.root.after(0, lambda: self._on_prep_done(token, None))

        threading.Thread(target=work, daemon=True).start()

    def _on_prep_done(self, token: int, proc: np.ndarray | None) -> None:
        if token != self._prep_token:
            return
        if proc is not None:
            self._proc_bgr = proc

    def load_model(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.status.set("正在加载模型…")
        pose_w = self.pose_weights.get().strip()
        refine_w = self.refine_weights.get().strip()
        roi_w = DEFAULT_ROI_WEIGHTS

        def work():
            try:
                pose_path = resolve_weights(pose_w)
                pose_model = YOLO(pose_path)
                refine_model = None
                refine_path = None
                if refine_w and os.path.isfile(refine_w):
                    refine_path = refine_w
                    refine_model = load_corner_model(refine_path)
                roi_model = None
                if os.path.isfile(roi_w):
                    try:
                        roi_model = YOLO(roi_w)
                    except Exception:
                        roi_model = None
                self.root.after(
                    0,
                    lambda: self._on_model_loaded(
                        pose_model, pose_path, refine_model, refine_path, roi_model
                    ),
                )
            except Exception as exc:
                msg = f"加载失败: {exc}"
                self.root.after(0, lambda m=msg: self._on_error(m))

        threading.Thread(target=work, daemon=True).start()

    def _on_model_loaded(self, pose_model, pose_path: str, refine_model, refine_path, roi_model) -> None:
        self.pose_model = pose_model
        self.refine_model = refine_model
        self.roi_model = roi_model
        self._loaded_pose = pose_path
        self._loaded_refine = refine_path
        self._busy = False
        extra = f" + refine {os.path.basename(refine_path)}" if refine_path else ""
        self.status.set(f"模型已就绪 | detect {os.path.basename(pose_path)}{extra}")

    def run_infer(self) -> None:
        if self._busy:
            return
        if self._orig_bgr is None or not self.image_path.get().strip():
            messagebox.showwarning("提示", "请先选择图片")
            return
        if self.pose_model is None:
            messagebox.showwarning("提示", "模型尚未加载完成，请稍候或点「重新加载模型」")
            return

        multi = bool(self.use_multi_crop.get())
        cascade = bool(self.use_cascade.get()) and self.refine_model is not None
        self._busy = True
        if cascade:
            self.status.set("推理中（检测+角点精修）…")
        elif multi:
            self.status.set("推理中（多候选裁剪）…")
        else:
            self.status.set("推理中（全图）…")
        pose_model = self.pose_model
        refine_model = self.refine_model if cascade else None
        roi_model = self.roi_model if multi else None
        conf = float(self.conf.get())
        orig = self._orig_bgr.copy()
        proc = None if self._proc_bgr is None else self._proc_bgr.copy()
        raw_path = self.image_path.get().strip()

        def work():
            try:
                infer_img = proc
                if infer_img is None:
                    raw = read_raw(raw_path)
                    infer_img, _ = preprocess_xray(raw)

                tip = ""
                centers = np.zeros((0, 2), dtype=np.float64)
                if cascade:
                    # 新重训权重：center(1kpt) + corner(4kpt)；旧 4+4 权重仍走旧 cascade
                    use_retrain = False
                    try:
                        dname = os.path.basename(self._loaded_pose or "")
                        rname = os.path.basename(self._loaded_refine or "")
                        use_retrain = (
                            ("center" in dname)
                            or ("corner" in rname)
                            or ("patch" in rname)
                            or ("ctx" in rname)
                            or ("reg" in rname)
                        )
                    except Exception:
                        use_retrain = False
                    if use_retrain:
                        _, quads, names, _cobb, diag = predict_retrain_cascade(
                            pose_model,
                            refine_model,
                            infer_img,
                            conf=conf,
                            imgsz=1024,
                            corner_weights_name=os.path.basename(self._loaded_refine or ""),
                        )
                        crop_name = "retrain-cascade"
                        tip = str((diag or {}).get("tip") or "")
                        raw_c = (diag or {}).get("centers") or []
                        if raw_c:
                            centers = np.asarray(raw_c, dtype=np.float64).reshape(-1, 2)
                        else:
                            centers = centers_from_quads(quads)
                    else:
                        _, quads, names, _cobb = predict_cascade(
                            pose_model,
                            refine_model,
                            infer_img,
                            conf=conf,
                            imgsz=1024,
                        )
                        crop_name = "cascade"
                        centers = centers_from_quads(quads)
                        if not quads:
                            tip = "级联无结果（检测或精修未产出椎体）"
                elif multi:
                    quads, names, _cobb, crop_name, _box, _tried = robust_pose_infer(
                        pose_model,
                        infer_img,
                        conf=conf,
                        imgsz=1024,
                        roi_model=roi_model,
                    )
                    centers = centers_from_quads(quads)
                    if not quads:
                        tip = "多候选裁剪后仍未检出椎体"
                else:
                    _, quads, names, _cobb = predict_one(
                        pose_model, infer_img, conf, 1024
                    )
                    crop_name = "full"
                    centers = centers_from_quads(quads)
                    if not quads:
                        tip = "全图检测未检出椎体（当前未跑角点精修）"

                curves = compute_cobb_curves(quads, image_height=orig.shape[0])
                cobb_summary = format_cobb_summary(curves, names)
                n = len(quads) if quads else 0
                if n == 0:
                    msg = f"无结果 | {crop_name}" + (f" | {tip}" if tip else "")
                elif tip and "全部未检出" in tip:
                    msg = f"完成 n={n}/<=17 | {crop_name} | {tip}"
                elif tip and "角点成功" in tip and "/" in tip:
                    msg = f"完成 n={n}/<=17 | {crop_name} | {tip}"
                else:
                    msg = f"完成 n={n}/<=17 | {crop_name}"
                    if tip and tip not in msg:
                        msg = f"{msg} | {tip}"
                centers_copy = centers.copy()
                self.root.after(
                    0,
                    lambda m=msg, q=quads, n=names, c=curves, s=cobb_summary, t=tip, nn=n, ctr=centers_copy: self._on_infer_done(
                        m, q, n, c, s, tip=t, n_quad=nn, centers=ctr
                    ),
                )
            except Exception as exc:
                msg = f"推理失败: {exc}"
                self.root.after(0, lambda m=msg: self._on_error(m))

        threading.Thread(target=work, daemon=True).start()

    def _rebuild_cobb_pills(self, curves) -> None:
        """按推理结果重建每个 Cobb 角的药丸开关（默认全开）。"""
        for w in self._cobb_pills:
            w.destroy()
        self._cobb_pills.clear()
        self._cobb_vars.clear()
        for child in self._cobb_pills_row.winfo_children():
            child.destroy()

        if not curves:
            self._cobb_empty = ttk.Label(self._cobb_pills_row, text="推理后显示各弯开关")
            self._cobb_empty.pack(side=tk.LEFT)
            return

        names = self._last_names
        for curve in curves:
            var = tk.BooleanVar(value=True)
            self._cobb_vars.append(var)
            pill = PillToggle(
                self._cobb_pills_row,
                text=curve.label(names),
                accent_bgr=tuple(int(x) for x in curve.color),
                variable=var,
                command=self._refresh_overlay,
            )
            pill.pack(side=tk.LEFT, padx=(0, 8), pady=2)
            self._cobb_pills.append(pill)

    def _visible_curves(self, curves):
        """仅保留药丸开关打开的 Cobb 弯。"""
        if not curves:
            return []
        if not self._cobb_vars:
            return list(curves)
        return [
            curve
            for i, curve in enumerate(curves)
            if i >= len(self._cobb_vars) or bool(self._cobb_vars[i].get())
        ]

    def _compose_view(self, quads, names, curves, centers=None):
        """按勾选状态合成显示图：推理框 / 中心线 / 各 Cobb 弯可独立开关。"""
        import importlib

        import cobb_angles as _cobb_mod

        importlib.reload(_cobb_mod)  # 确保 GUI 热更新到最新绘制逻辑
        vis = self._orig_bgr.copy()
        if quads and self.show_boxes.get():
            vis = draw(vis, quads, names or [])
        ctr = centers if centers is not None else self._last_centers
        if self.show_centers.get() and ctr is not None and len(ctr) > 0:
            vis = draw_center_curve(vis, ctr)
        visible = self._visible_curves(curves)
        if visible:
            _cobb_mod.draw_cobb_curves(vis, quads, visible, show_text=True)
        return vis

    def _on_infer_done(
        self,
        msg: str,
        quads,
        names,
        curves,
        cobb_summary: str,
        tip: str = "",
        n_quad: int = -1,
        centers=None,
    ) -> None:
        self._busy = False
        self._last_quads = quads
        self._last_names = names
        self._last_curves = curves
        if centers is None:
            self._last_centers = centers_from_quads(quads)
        else:
            self._last_centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
        self._rebuild_cobb_pills(curves or [])
        self._set_view_bgr(self._compose_view(quads, names, curves, centers=self._last_centers))
        self.status.set(msg + f" | {os.path.basename(self.image_path.get())}")
        self._redraw(fast=False)
        n = n_quad if n_quad >= 0 else (len(quads) if quads else 0)
        # 无结果或角点全失败时弹窗说明阶段
        if n == 0:
            detail = tip or "未知阶段失败"
            messagebox.showwarning(
                "无检测结果",
                f"{detail}\n\n"
                "说明：\n"
                "· 中心未检出 → 问题在中心点模型 / 置信度 / 预处理\n"
                "· 中心有、角点无 → 问题在角点精修模型或裁剪区域\n"
                "· 可尝试调低置信度后重试",
            )
        elif tip and "全部未检出" in tip:
            messagebox.showwarning("角点未检出", tip + "\n\n画面上的框是中心退化框，不是角点模型输出。")

    def _refresh_overlay(self) -> None:
        """勾选「显示推理框 / 中心线 / Cobb 药丸」时，基于最近一次结果重绘。"""
        if self._orig_bgr is None:
            return
        if self._last_quads is None and (self._last_centers is None or len(self._last_centers) == 0):
            return
        self._set_view_bgr(
            self._compose_view(
                self._last_quads or [],
                self._last_names or [],
                self._last_curves or [],
                centers=self._last_centers,
            )
        )
        self._redraw(fast=False)

    def _set_view_bgr(self, bgr: np.ndarray) -> None:
        """更新显示图并同步 RGB 缓存（缩放路径只读该缓存）。"""
        self._view_bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._pil_rgb = Image.fromarray(rgb)

    def _on_error(self, msg: str) -> None:
        self._busy = False
        self.status.set(msg)
        messagebox.showerror("错误", msg)

    def save_result(self) -> None:
        if self._view_bgr is None:
            messagebox.showinfo("提示", "还没有可保存的结果")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            initialfile="yolo_vert_pred.png",
        )
        if not path:
            return
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", self._view_bgr)
        if ok:
            buf.tofile(path)
            self.status.set(f"已保存: {path}")

    def _on_resize(self, _e=None) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(80, lambda: self._redraw(fast=False))

    def _image_layout(self, zoom: float | None = None) -> tuple[int, int, int, int, int, int]:
        """返回 (cw, ch, nw, nh, ox, oy)。"""
        img = self._view_bgr
        assert img is not None
        h, w = img.shape[:2]
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        z = float(self._zoom if zoom is None else zoom)
        fit = min(cw / w, ch / h)
        scale = fit * z
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        ox = int((cw - nw) // 2 + self._pan_x)
        oy = int((ch - nh) // 2 + self._pan_y)
        return cw, ch, nw, nh, ox, oy

    def _schedule_hq_redraw(self) -> None:
        """交互停下后再做一次双线性高质量重绘。"""
        if self._zoom_job is not None:
            self.root.after_cancel(self._zoom_job)
        self._zoom_job = self.root.after(60, lambda: self._redraw(fast=False))

    def _on_zoom(self, event) -> None:
        """鼠标滚轮缩放，以光标位置为中心。"""
        if self._view_bgr is None:
            return
        delta = getattr(event, "delta", 0)
        num = getattr(event, "num", None)
        if delta > 0 or num == 4:
            factor = 1.12
        elif delta < 0 or num == 5:
            factor = 1.0 / 1.12
        else:
            return

        old_zoom = self._zoom
        new_zoom = float(np.clip(old_zoom * factor, 0.25, 8.0))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        _, _, nw_old, nh_old, ox_old, oy_old = self._image_layout(old_zoom)
        mx = int(getattr(event, "x", self.canvas.winfo_width() // 2))
        my = int(getattr(event, "y", self.canvas.winfo_height() // 2))

        self._zoom = new_zoom
        _, _, nw_new, nh_new, _, _ = self._image_layout(new_zoom)
        # 保持鼠标下的图像点不动
        rx = nw_new / max(nw_old, 1)
        ry = nh_new / max(nh_old, 1)
        ox_new = mx - (mx - ox_old) * rx
        oy_new = my - (my - oy_old) * ry
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        self._pan_x = ox_new - (cw - nw_new) // 2
        self._pan_y = oy_new - (ch - nh_new) // 2
        self._redraw(fast=True)
        self._schedule_hq_redraw()
        return "break"

    def _on_pan_start(self, event) -> None:
        self._drag_last = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _on_pan_move(self, event) -> None:
        if self._drag_last is None or self._view_bgr is None:
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self._drag_last = (event.x, event.y)
        self._pan_x += dx
        self._pan_y += dy
        self._redraw(fast=True)

    def _on_pan_end(self, _event=None) -> None:
        self._drag_last = None
        self.canvas.config(cursor="")
        self._schedule_hq_redraw()

    def _reset_view(self, _event=None) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._redraw(fast=False)

    def _redraw(self, fast: bool = False) -> None:
        """重绘画布。

        卡顿根因曾是：每次滚轮都对整张原图做 BGR→RGB + 双线性 resize。
        现改为：缓存 RGB；只裁切可见区域再缩放；交互中用最近邻。
        """
        if self._view_bgr is None:
            return
        if self._pil_rgb is None:
            self._set_view_bgr(self._view_bgr)

        cw, ch, nw, nh, ox, oy = self._image_layout()
        pil = self._pil_rgb
        assert pil is not None
        src_w, src_h = pil.size

        # 只渲染与画布相交的区域，避免放大后对整幅超大图做 resize
        left = max(0, -ox)
        top = max(0, -oy)
        right = min(nw, cw - ox)
        bottom = min(nh, ch - oy)
        if right <= left or bottom <= top:
            self.canvas.delete("all")
            self._photo = None
            return

        scale_x = src_w / max(nw, 1)
        scale_y = src_h / max(nh, 1)
        crop_box = (
            max(0, int(left * scale_x)),
            max(0, int(top * scale_y)),
            min(src_w, int(np.ceil(right * scale_x))),
            min(src_h, int(np.ceil(bottom * scale_y))),
        )
        out_w = max(1, right - left)
        out_h = max(1, bottom - top)
        resample = Image.Resampling.NEAREST if fast else Image.Resampling.BILINEAR
        cropped = pil.crop(crop_box)
        if cropped.size != (out_w, out_h):
            cropped = cropped.resize((out_w, out_h), resample)

        self._photo = ImageTk.PhotoImage(cropped)
        self.canvas.delete("all")
        self.canvas.create_image(ox + left, oy + top, image=self._photo, anchor=tk.NW)


def main() -> None:
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
