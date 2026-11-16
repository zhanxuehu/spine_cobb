"""
角点训练：轴对齐上下文 + 双椎邻接约束损失。
从 vert_corner_ctx_best 微调，best 复制到 best_weights/vert_corner_ctx_adj_best.pt。

    python spine_pose_yolo/prepare_retrain_datasets.py --skip-patch --skip-ctx-patch --skip-center --build-ctx-adj
    python spine_pose_yolo/train_corner_adj.py --pretrained best_weights/vert_corner_ctx_best.pt
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8PoseLoss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA = HERE / "datasets" / "vert_corner_ctx_adj" / "data.yaml"
BEST_DIR = ROOT / "best_weights"


class AdjacentPoseLoss(v8PoseLoss):
    """标准 pose + 同图多椎邻接（上下边中点/倾角/高度）约束。"""

    def __init__(self, model, adj_w: float = 1.2, ang_w: float = 0.35, h_w: float = 0.25):
        super().__init__(model)
        self.adj_w = float(adj_w)
        self.ang_w = float(ang_w)
        self.h_w = float(h_w)

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, stride_tensor, target_bboxes, pred_kpts):
        kpts_loss, kpts_obj_loss = super().calculate_keypoints_loss(
            masks, target_gt_idx, keypoints, stride_tensor, target_bboxes, pred_kpts
        )
        if not masks.any() or pred_kpts.shape[-2] < 4:
            return kpts_loss, kpts_obj_loss

        batch_ids = masks.nonzero(as_tuple=True)[0]
        pred = pred_kpts[masks][..., :2]  # (N, K, 2) stride-space
        if pred.shape[0] < 2:
            return kpts_loss, kpts_obj_loss

        adj = pred.new_zeros(())
        n_pairs = 0
        for b in batch_ids.unique():
            sel = (batch_ids == b).nonzero(as_tuple=True)[0]
            if sel.numel() < 2:
                continue
            pk = pred[sel]  # (M, 4+, 2)
            cy = pk[:, :4, 1].mean(dim=1)
            order = torch.argsort(cy)
            pk = pk[order]
            for i in range(pk.shape[0] - 1):
                up, dn = pk[i], pk[i + 1]
                # TL TR BL BR
                top_up = 0.5 * (up[0] + up[1])
                bot_up = 0.5 * (up[2] + up[3])
                top_dn = 0.5 * (dn[0] + dn[1])
                bot_dn = 0.5 * (dn[2] + dn[3])
                # 上椎下边 vs 下椎上边应靠近
                adj = adj + F.l1_loss(bot_up, top_dn)
                # 倾角（用上边）
                ang_up = torch.atan2(up[1, 1] - up[0, 1], up[1, 0] - up[0, 0] + 1e-6)
                ang_dn = torch.atan2(dn[1, 1] - dn[0, 1], dn[1, 0] - dn[0, 0] + 1e-6)
                adj = adj + self.ang_w * (ang_up - ang_dn).abs()
                # 高度别差太多
                h_up = (bot_up[1] - top_up[1]).abs()
                h_dn = (bot_dn[1] - top_dn[1]).abs()
                adj = adj + self.h_w * (h_up - h_dn).abs()
                n_pairs += 1
        if n_pairs > 0:
            kpts_loss = kpts_loss + self.adj_w * (adj / n_pairs)
            kpts_loss = kpts_loss + 0.0 * pred.sum()
        return kpts_loss, kpts_obj_loss


class AdjacentPoseModel(PoseModel):
    def init_criterion(self):
        return AdjacentPoseLoss(self, adj_w=1.2, ang_w=0.35, h_w=0.25)


class AdjacentPoseTrainer(PoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            AdjacentPoseModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                data_kpt_shape=self.data["kpt_shape"],
                verbose=verbose and RANK == -1,
            )
        )
        if weights:
            model.load(weights)
        return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default="yolo11s-pose.pt")
    ap.add_argument(
        "--pretrained",
        default=str(BEST_DIR / "vert_corner_ctx_best.pt"),
    )
    ap.add_argument("--name", default="vert_corner_ctx_adj")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--lr0", type=float, default=0.005)
    args = ap.parse_args()

    if not Path(args.data).is_file():
        raise FileNotFoundError(f"缺少 {args.data}，请先 --build-ctx-adj")
    init = args.pretrained if args.pretrained and Path(args.pretrained).is_file() else args.model
    print(f"=== train {args.name} (adjacent loss) init={init} ===")
    model = YOLO(init)
    model.train(
        data=args.data,
        project=str(HERE / "runs"),
        name=args.name,
        exist_ok=True,
        device=args.device,
        workers=args.workers,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="SGD",
        lr0=args.lr0,
        pose=18.0,
        degrees=12.0,
        translate=0.08,
        scale=0.30,
        mosaic=0.20,
        mixup=0.0,
        close_mosaic=15,
        fliplr=0.0,
        flipud=0.0,
        trainer=AdjacentPoseTrainer,
    )
    best = HERE / "runs" / args.name / "weights" / "best.pt"
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = BEST_DIR / "vert_corner_ctx_adj_best.pt"
    shutil.copy2(best, dest)
    print(f"Best: {best}")
    print(f"Copied: {dest}")


if __name__ == "__main__":
    main()
