"""局部四角 CNN 回归：固定 ctx 裁剪 → 8 维归一化角点。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE = 160
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CornerRegressor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_f = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


def preprocess_bgr(crop_bgr: np.ndarray, imgsz: int = IMG_SIZE) -> torch.Tensor:
    """BGR crop → float tensor (3,H,W) ImageNet-normalized."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


def load_corner_regressor(
    weights: str | Path,
    device: str | torch.device = "cpu",
) -> CornerRegressor:
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        imgsz = int(ckpt.get("imgsz", IMG_SIZE))
    else:
        state = ckpt
        imgsz = IMG_SIZE
    model = CornerRegressor(pretrained=False)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    model._infer_imgsz = imgsz  # type: ignore[attr-defined]
    return model


def is_corner_regressor_weights(path: str | Path) -> bool:
    name = Path(path).name.lower()
    if "corner_reg" in name or "vert_corner_reg" in name:
        return True
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return False
    if isinstance(ckpt, dict) and ckpt.get("arch") == "CornerRegressor":
        return True
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        keys = ckpt["model"].keys()
        return any(str(k).startswith("backbone.") for k in keys) and any(
            str(k).startswith("head.") for k in keys
        )
    return False


@torch.inference_mode()
def predict_corners_norm(
    model: CornerRegressor,
    crop_bgr: np.ndarray,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """返回 (4,2) 归一化到 crop 的 TL TR BL BR。"""
    imgsz = int(getattr(model, "_infer_imgsz", IMG_SIZE))
    t = preprocess_bgr(crop_bgr, imgsz=imgsz).unsqueeze(0).to(device)
    pred = model(t)[0].detach().cpu().numpy().reshape(4, 2)
    return np.clip(pred, -0.05, 1.05).astype(np.float64)
