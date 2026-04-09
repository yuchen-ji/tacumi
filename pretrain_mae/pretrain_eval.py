import os
import random
import argparse
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.dataset.umi_pretrain_mae import TactileAutoencoderDataset
from .pretrain_mae import attach_last_attn_hook
from .pretrain_mae_model import TactileAutoencoderModel

# -----------------------------------------------------------------------------#
# Misc utils
# -----------------------------------------------------------------------------#
_VIRIDIS = cm.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_hwc(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().float()
    if img.max() > 1:
        img = img / 255.0
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def attention_to_heatmap(attn: torch.Tensor, patch_size: int = 16, image_size: int = 224) -> np.ndarray:
    num_heads, num_patches = attn.shape
    grid = int(num_patches ** 0.5)
    avg_map = attn.reshape(num_heads, grid, grid).mean(0, keepdim=True).unsqueeze(0)
    return (
        F.interpolate(avg_map, (image_size, image_size), mode="bilinear", align_corners=False)
        .squeeze()
        .cpu()
        .numpy()
    )


def plot_attention_overlay(rgb: torch.Tensor, heatmap: np.ndarray, path: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(_to_hwc(rgb))
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_full_comparison(
    rgb: torch.Tensor, masked: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, path: str
) -> None:
    imgs = [
        _to_hwc(rgb),
        _to_hwc(masked) if masked.ndim == 3 else masked.squeeze().cpu(),
        _to_hwc(pred) if pred.ndim == 3 else pred.squeeze().cpu(),
        _to_hwc(gt) if gt.ndim == 3 else gt.squeeze().cpu(),
    ]
    titles = ["RGB Input", "Masked Tactile", "Reconstruction", "GroundTruth"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, title, im in zip(axes, titles, imgs):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def heatmap_to_color(t: torch.Tensor) -> torch.Tensor:
    idx = (t.clamp(0, 1) * 255).long()
    palette = torch.from_numpy(_VIRIDIS).to(t.device)
    return palette[idx.squeeze()].permute(2, 0, 1).float()


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tactile auto-encoder")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--dataset", required=True, help="Path to dataset (.zarr.zip)")
    parser.add_argument("--out_dir", default="eval_outputs", help="Where to save evaluation outputs")
    parser.add_argument("--plot_images", action="store_true", help="Save qualitative images")
    return parser.parse_args()


# -----------------------------------------------------------------------------#
# Evaluation
# -----------------------------------------------------------------------------#
def main(args: argparse.Namespace) -> None:
    ckpt_path = args.checkpoint
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    cfg_file = os.path.join(ckpt_dir, "config.yaml")

    cfg = OmegaConf.load(cfg_file) if os.path.exists(cfg_file) else OmegaConf.create()
    cfg.eval_checkpoint = ckpt_path
    cfg.eval_out_dir = args.out_dir
    cfg.eval_plot_images = args.plot_images

    plot_images = cfg.eval_plot_images
    if plot_images:
        out_dir = cfg.eval_out_dir
        os.makedirs(out_dir, exist_ok=True)
        print(f"[Eval] output dir : {out_dir}")

    seed_everything(cfg.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Eval] checkpoint : {ckpt_path}")

    # No transform during evaluation
    transform = nn.Identity()

    dataset = TactileAutoencoderDataset(
        shape_meta=cfg.task.shape_meta,
        dataset_path=args.dataset, 
        cache_dir=cfg.task.get("cache_dir", None),
        val_ratio=cfg.task.get("val_ratio", 0.1),
        train_ratio=cfg.task.get("train_ratio", 1.0),
        seed=cfg.training.seed,
        transforms=[transform],
    )
    val_dataset = dataset.get_validation_dataset()

    print(f"[Eval] train episodes = {dataset.num_train_episodes}")
    print(f"[Eval] val episodes   = {dataset.num_val_episodes}")
    print(f"[Eval] val frames     = {len(val_dataset)}")

    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.dataloader.get("num_workers", 8),
        pin_memory=True,
        drop_last=False,
    )

    print("[Eval] init model…")
    model = TactileAutoencoderModel(
        embed_dim=768,
        tactile_patch_size=4,
        num_heads=8,
        attention_type=cfg.training.attention_type,
        predict_channel=cfg.training.predict_channel,
        mask_ratio_min=cfg.training.mask_ratio_min,
        mask_ratio_max=cfg.training.mask_ratio_max,
    ).to(device)

    attach_last_attn_hook(model.clip_encoder)

    def _save_attn_t2i(_, __, output: Any) -> None:
        model.cross_attn.attn_t2i.latest_attn = output[1].detach() if output[1] is not None else None

    model.cross_attn.attn_t2i.register_forward_hook(_save_attn_t2i)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "ema_state_dict" in ckpt:
        print("[Eval] Using EMA weights from checkpoint.")
        from torch.optim.swa_utils import AveragedModel
        ema_model = AveragedModel(model)
        ema_model.load_state_dict(ckpt["ema_state_dict"])
        model = ema_model.module  # unwrap
        # Re-attach hooks on the unwrapped model
        attach_last_attn_hook(model.clip_encoder)
        model.cross_attn.attn_t2i.register_forward_hook(_save_attn_t2i)
        print("[Eval] EMA weights loaded.")
    else:
        print("[Eval] Using direct model_state_dict from checkpoint.")
        model.load_state_dict(ckpt["model_state_dict"])
        print("[Eval] weights loaded.")

    model.eval()

    print("[Eval] running…")
    mse_total, n_samples = 0.0, 0

    for idx, batch in enumerate(tqdm(loader)):
        rgb = batch["rgb_cur"].to(device)
        rgb = rgb.unsqueeze(1) if rgb.ndim == 4 else rgb
        tactile = batch["tactile_cur"].to(device)

        def _capture_masked(_, inp, __):
            model.cnn.masked_seen = inp[0].detach().cpu()

        hook = model.cnn.register_forward_hook(_capture_masked)

        with torch.no_grad():
            pred = model(rgb, tactile)
            gt = (
                model.convert_12x64_to_3x24x32(tactile, False)
                if model.predict_channel == 3
                else model.convert_12x64_to_1x24x32(tactile, False)
            )
        hook.remove()

        mse_total += F.mse_loss(pred, gt, reduction="mean").item()
        n_samples += 1

        if plot_images:
            masked = model.cnn.masked_seen[0]
            pred_vis = heatmap_to_color(pred[0]) if model.predict_channel == 1 else pred[0]
            gt_vis = heatmap_to_color(gt[0]) if model.predict_channel == 1 else gt[0]
            rgb_vis = rgb[0, 0]

            vision_attn = model.clip_encoder.blocks[-1].attn.latest_attn
            heat_self = (
                attention_to_heatmap(vision_attn[0, :, 0, 1:]) if vision_attn is not None else None
            )

            cross_attn = model.cross_attn.attn_t2i.latest_attn
            heat_cross = (
                attention_to_heatmap(cross_attn[0, :, 0, :]) if cross_attn is not None else None
            )

            sample_dir = os.path.join(out_dir, f"sample_{idx:05d}")
            os.makedirs(sample_dir, exist_ok=True)
            print(sample_dir)

            save_full_comparison(rgb_vis, masked, pred_vis, gt_vis, os.path.join(sample_dir, "compare.png"))
            if heat_self is not None:
                plot_attention_overlay(rgb_vis, heat_self, os.path.join(sample_dir, "vision_self_attn.png"))
            if heat_cross is not None:
                plot_attention_overlay(rgb_vis, heat_cross, os.path.join(sample_dir, "cross_attn.png"))

    print(f"[Eval] done. samples={n_samples}")
    print(f"[Eval] cumulative MSE: {mse_total:.6f}")
    print(f"[Eval] average   MSE: {mse_total / max(n_samples, 1):.6f}")


if __name__ == "__main__":
    main(parse_args())
