"""
触觉自编码器 MAE（Masked Autoencoder）预训练脚本
====================================================

核心思路：
  1. 输入 RGB 相机图像 + 触觉传感器矩阵 (12×64)
  2. 将触觉数据转换为 viridis 色图后随机 mask 60%-80% 的 patch（MAE 策略）
  3. 用 CLIP ViT-B/16 编码 RGB 图像；用 CNN 编码被 mask 的触觉彩色图
  4. 通过 Cross-Attention 让两种模态相互交互
  5. 解码器重建完整触觉数据，训练目标为 MSE 损失

训练特性：
  - 支持 EMA（指数移动平均）模型用于更稳定的评估
  - Warmup + CosineAnnealing 学习率调度
  - 可选 CLS-based 或 Patch-based 两种 Cross-Attention 模式
  - 周期性保存 debug 可视化图（注意力热力图、训练/验证预测对比等）
"""

import os
import math
import random
import datetime
import types
from typing import Any

import hydra
import hydra.utils as hy_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel  # 用于 EMA（指数移动平均）
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import wandb

from diffusion_policy.dataset.umi_pretrain_mae import TactileAutoencoderDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from .pretrain_mae_model import TactileAutoencoderModel

# 注册 OmegaConf 自定义解析器，允许在 yaml 配置中使用 ${eval:...} 表达式
OmegaConf.register_new_resolver("eval", eval, replace=True)


# -----------------------------------------------------------------------------#
# 可复现性工具                                                                   #
# -----------------------------------------------------------------------------#
def seed_everything(seed: int) -> None:
    """固定所有随机种子，确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------#
# 可视化辅助函数                                                               #
# -----------------------------------------------------------------------------#
def _to_hwc(img: torch.Tensor) -> np.ndarray:
    """
    将 (C, H, W) 格式的图像张量转换为 (H, W, C) 的 numpy 数组。
    自动处理 [0,255] → [0,1] 的归一化。
    """
    img = img.float()
    if img.max() > 1:
        img = img / 255.0
    return img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def debug_plot_masked_input(
    rgb_batch: torch.Tensor,
    tactile_batch: torch.Tensor,
    out_dir: str,
    prefix: str = "",
    max_samples: int = 4,
) -> None:
    """
    可视化 batch 中的 masked 输入：左列显示 RGB 图像，右列显示触觉矩阵。
    用于 debug 时确认 mask 策略是否生效。
    """
    os.makedirs(out_dir, exist_ok=True)
    # 如果 rgb_batch 是 5D (B, n_cams, C, H, W) 且只有 1 个相机，去掉相机维度
    if rgb_batch.ndim == 5 and rgb_batch.size(1) == 1:
        rgb_batch = rgb_batch[:, 0]

    rgb_batch = rgb_batch.cpu()
    tactile_batch = tactile_batch.cpu().numpy()

    n = min(max_samples, rgb_batch.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(6.5, 3 * n))

    for i in range(n):
        axes[i, 0].imshow(_to_hwc(rgb_batch[i]))
        axes[i, 0].set_title(f"Masked Image [{i}]")
        axes[i, 0].axis("off")

        im = axes[i, 1].imshow(tactile_batch[i], cmap="viridis", origin="upper")
        axes[i, 1].set_title(f"Masked Tactile [{i}]")
        axes[i, 1].axis("off")
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_masked_batch.png"))
    plt.close(fig)


def attention_to_heatmap(attn: torch.Tensor, patch_size: int, image_size: int) -> np.ndarray:
    """
    将注意力权重转换为可叠加在原图上的热力图。
    
    参数:
        attn: (num_heads, num_patches) 注意力权重，来自某个 token 对所有 patch 的注意力
        patch_size: ViT 的 patch 大小（例如 16）
        image_size: 原始图像尺寸（例如 224）
    
    流程：将 (heads, patches) → reshape 为 (heads, grid, grid) → 对 heads 取均值
         → 双线性插值上采样到原图大小
    """
    h, s = attn.shape  # h=num_heads, s=num_patches
    g = int(s**0.5)    # grid 边长，例如 196→14
    # reshape 为 2D 网格，对所有 head 取均值，得到 (1, 1, g, g) 的注意力图
    avg = attn.reshape(h, g, g).mean(0, keepdim=True).unsqueeze(0)
    return (
        F.interpolate(avg, (image_size, image_size), mode="bilinear", align_corners=False)
        .squeeze()
        .cpu()
        .numpy()
    )


def plot_attention_overlay(rgb: torch.Tensor, heat: np.ndarray, path: str) -> None:
    """将注意力热力图半透明叠加在 RGB 图像上并保存。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(_to_hwc(rgb))
    plt.imshow(heat, cmap="jet", alpha=0.5)  # jet 色图，50% 透明度
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def debug_plot_train_val(
    train_batch,
    train_pred,
    val_batch,
    val_pred,
    out_dir: str = "debug_plots",
    step: int = 0,
) -> None:
    """
    并排绘制训练和验证集的预测结果，每行 3 列：
      列 0: RGB 相机图像
      列 1: 触觉真值 (Ground Truth)
      列 2: 模型预测的触觉图
    第一行固定为训练集样本，后续行为验证集随机样本（最多 10 个）。
    """
    os.makedirs(out_dir, exist_ok=True)
    val_bs = val_batch["rgb_cur"].shape[0]
    val_idx = np.random.choice(val_bs, size=min(10, val_bs), replace=False)
    rows = 1 + len(val_idx)
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows))

    def _plot(b, p, row, tag, idx):
        """绘制单行：RGB + GT触觉 + 预测触觉。"""
        axes[row, 0].imshow(_to_hwc(b["rgb_cur"][idx]))
        axes[row, 0].set_title(f"{tag}: Camera")
        axes[row, 0].axis("off")

        tact = b["tactile_cur"][idx].cpu().numpy()
        axes[row, 1].imshow(tact, cmap="viridis", origin="upper", vmin=tact.min(), vmax=tact.max())
        axes[row, 1].set_title(f"{tag}: Tactile (GT)")
        axes[row, 1].axis("off")

        if p is not None:
            c = p.shape[1]
            if c == 3:
                # 3 通道输出 → RGB 彩色触觉图
                axes[row, 2].imshow(_to_hwc(p[idx]))
            else:
                # 单通道输出 → 灰度触觉图
                axes[row, 2].imshow(p[idx].clamp(0, 1).cpu().squeeze(0), cmap="viridis", origin="upper")
            axes[row, 2].set_title(f"{tag}: Tactile (Pred)")
            axes[row, 2].axis("off")

    _plot(train_batch, train_pred, 0, "Train", 0)
    for r, i in enumerate(val_idx, start=1):
        _plot(val_batch, val_pred, r, f"Val[{i}]", i)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"debug_train_val_{step}.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------#
# 评估辅助函数                                                                 #
# -----------------------------------------------------------------------------#
def eval_autoencoder(model, loader, device="cuda", max_batches: int = 5, debug_plot=False, is_val=True):
    """
    在训练/验证集上评估自编码器的 MSE 重建损失。
    
    参数:
        model: 可以是普通模型或 AveragedModel（EMA 包装）
        loader: DataLoader
        max_batches: 最多评估的 batch 数量（避免评估时间太长）
        debug_plot: 是否保存第一个 batch 用于可视化
        is_val: 标识当前是验证集还是训练集（仅用于打印信息）
    
    返回:
        (平均 MSE 损失, 保存的 batch, 预测结果, GT)
    """
    model.eval()
    loss_sum = count = 0
    sb = sp = sg = None  # 用于 debug 可视化的 saved batch/pred/gt
    # 如果是 EMA 模型，取出内部实际模型
    base = model.module if isinstance(model, AveragedModel) else model

    with torch.no_grad():
        for i, batch in enumerate(loader):
            rgb = batch["rgb_cur"].to(device)
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(1)  # (B,C,H,W) → (B,1,C,H,W)，补上相机维度
            tactile = batch["tactile_cur"].to(device)

            pred = base(rgb, tactile)
            # 根据 predict_channel 选择不同的 GT 转换方式
            gt = (
                base.convert_12x64_to_3x24x32(tactile, False)  # 3 通道 RGB 彩色触觉图
                if base.predict_channel == 3
                else base.convert_12x64_to_1x24x32(tactile, False)  # 单通道灰度触觉图
            )

            loss_sum += F.mse_loss(pred, gt).item()
            count += 1
            if debug_plot and i == 0:
                sb, sp, sg = batch, pred, gt
            if i >= max_batches - 1:
                break

    split = "Val" if is_val else "Train"
    print(f"[eval_autoencoder] {split} MSE over {count} mini-batches = {loss_sum / max(count,1):.6f}")
    return loss_sum / max(count, 1), sb, sp, sg


def attach_last_attn_hook(vit, which: int = -1) -> None:
    """
    猴子补丁(monkey-patch) ViT 指定 Transformer block 的自注意力层，
    使其在前向传播时将注意力权重保存在 self.latest_attn 属性中。
    
    这样我们可以在训练过程中提取注意力图用于可视化，
    而无需修改 timm 库的源代码。
    
    参数:
        vit: timm 的 ViT 模型
        which: 要 hook 的 block 索引，默认 -1 表示最后一个 block
    """
    attn = vit.blocks[which].attn
    orig_fwd = attn.forward

    def new_fwd(self, x):
        B, N, C = x.shape
        # 手动计算 Q, K, V 并提取注意力权重
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        a = (q @ k.transpose(-2, -1)) * self.scale  # 缩放点积注意力
        a = a.softmax(-1)
        self.latest_attn = a.detach()  # 保存注意力权重供后续可视化
        x = (a @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    attn.forward = types.MethodType(new_fwd, attn)


# -----------------------------------------------------------------------------#
# 训练主函数                                                                    #
# -----------------------------------------------------------------------------#
@hydra.main(version_base=None, config_path="../diffusion_policy/config", config_name="pretrain_mae.yaml")
def main(cfg: Any) -> None:
    """
    MAE 预训练主循环。通过 Hydra 从 pretrain_mae.yaml 加载所有超参数。
    """
    # ========== 1. 可复现性 ==========
    seed_everything(cfg.training.seed)

    # ========== 2. 运行目录与命名 ==========
    dataset_path = cfg.task.dataset_path
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 当前时间戳
    # 权重名称, 便于区分, 在目录，pretrain_checkpoints/xxx 下，保存权重
    run_name = (
        f"checkpoint_{os.path.basename(dataset_path.rstrip(os.sep))}_{ts}"
        f"_attn-{cfg.training.attention_type}"        # cross-attn 模式: cls / patch
        f"_ema-{cfg.training.ema_decay if cfg.training.ema_pretrain else 0:.4f}"
        f"_clipLR-{cfg.training.clip_lr:.0e}"          # CLIP encoder 学习率
        f"_predCh-{cfg.training.predict_channel}"      # 预测通道数: 3(RGB) 或 1(灰度)
    )

    run_dir = os.path.join("pretrain_checkpoints", run_name)
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))  # 保存完整配置用于复现

    # ========== 3. 数据增强 ==========
    img_size = (224, 224)  # ViT 需要 224×224 输入
    tf_cfg = cfg.transforms
    tfs = []
    for t in tf_cfg:
        c = OmegaConf.to_container(t, resolve=True)
        if isinstance(t, nn.Module):
            tfs.append(t)
        elif isinstance(c, dict) and "_target_" in c:
            # 通过 Hydra 动态实例化 transform 对象
            tfs.append(hy_utils.instantiate(c))
        else:
            tfs.append(c)

    # 如果第一个 transform 是字典格式的 RandomCrop，转换为 torchvision 实际 transform
    if tfs and not isinstance(tfs[0], nn.Module):
        assert tfs[0]["type"] == "RandomCrop"
        ratio = tfs[0]["ratio"]
        tfs = [
            torchvision.transforms.RandomCrop(int(img_size[0] * ratio)),
            torchvision.transforms.Resize(img_size[0], antialias=True),
        ] + tfs[1:]

    transform = nn.Identity() if not tfs else nn.Sequential(*tfs)

    # ========== 4. 构建数据集和 DataLoader ==========
    dataset = TactileAutoencoderDataset(
        shape_meta=cfg.task.shape_meta,
        dataset_path=dataset_path,
        cache_dir=cfg.task.get("cache_dir"),
        val_ratio=cfg.training.val_ratio,
        train_ratio=1-cfg.training.val_ratio,
        seed=cfg.training.seed,
        transforms=transform,
    )
    val_dataset = dataset.get_validation_dataset()

    print(f"[DATA] Train ep  : {dataset.num_train_episodes}")
    print(f"[DATA] Val ep    : {dataset.num_val_episodes}")
    print(f"[DATA] Train len : {len(dataset)}")
    print(f"[DATA] Val len   : {len(val_dataset)}")

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.get("num_workers", 8),
        pin_memory=cfg.dataloader.get("pin_memory", False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.get("num_workers", 8),
        pin_memory=cfg.dataloader.get("pin_memory", False),
    )

    # ========== 5. 初始化 wandb 实验追踪 ==========
    wandb.init(
        project=cfg.logging.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        tags=cfg.logging.tags,
    )

    # ========== 6. 模型构建 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TactileAutoencoderModel(
        embed_dim=cfg.training.embed_dim,           # Transformer 嵌入维度（默认 768）
        tactile_patch_size=cfg.training.tact_patch_size,  # 触觉 mask patch 大小
        num_heads=cfg.training.num_heads,            # 多头注意力头数
        attention_type=cfg.training.attention_type,  # "cls" 或 "patch" 跨注意力模式
        predict_channel=cfg.training.predict_channel,  # 输出通道数: 3(RGB) 或 1(灰度)
        mask_ratio_min=cfg.training.mask_ratio_min,  # MAE 最低 mask 比例
        mask_ratio_max=cfg.training.mask_ratio_max,  # MAE 最高 mask 比例
    ).to(device)

    # ========== 7. EMA（指数移动平均）模型 ==========
    # EMA 通过维护模型参数的滑动平均来提供更稳定的评估表现
    if cfg.training.ema_pretrain:
        decay = cfg.training.ema_decay

        def _ema(p_ema, p, _):
            """EMA 更新公式: p_ema = decay * p_ema + (1-decay) * p"""
            return p_ema * decay + p * (1 - decay)

        ema_model = AveragedModel(model, avg_fn=_ema).to(device)
        # 给 EMA 模型也挂上注意力提取 hook
        attach_last_attn_hook(ema_model.module.clip_encoder)

        def _save(_, __, out):
            """Forward hook: 保存 cross-attention 的注意力权重"""
            ema_model.module.cross_attn.attn_t2i.latest_attn = out[1]

        ema_model.module.cross_attn.attn_t2i.register_forward_hook(_save)
        # EMA 模型不需要梯度
        for p in ema_model.parameters():
            p.requires_grad = False

    # 给主模型挂注意力提取 hook
    attach_last_attn_hook(model.clip_encoder)
    # 解冻 CLIP encoder 参数（微调而非冻结）
    for p in model.clip_encoder.parameters():
        p.requires_grad = True

    # ========== 8. 优化器（分组学习率） ==========
    # CLIP encoder 使用较小学习率（避免破坏预训练特征），其余模块使用正常学习率
    optimizer = optim.AdamW(
        [
            {"params": model.clip_encoder.parameters(), "lr": cfg.training.clip_lr},
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("clip_encoder")],
                "lr": cfg.training.encoder_lr,
            },
        ],
        weight_decay=2e-3,
    )

    # ========== 9. 学习率调度器 ==========
    # 前 10% 步数做线性 warmup（从 1e-6 增长到设定 lr），之后余弦退火衰减到 0
    total_steps = cfg.training.num_epochs * len(train_loader)
    warmup = int(0.1 * total_steps)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup, eta_min=0.0),
        ],
        milestones=[warmup],
    )

    # Forward hook: 保存主模型 cross-attention 的注意力权重
    def _save_attn(_, __, out):
        model.cross_attn.attn_t2i.latest_attn = out[1]

    model.cross_attn.attn_t2i.register_forward_hook(_save)

    # ========== 10. 训练循环 ==========
    global_step = 0

    def _cvt_12x64_to_3x24x32(t: torch.Tensor, mask=False):
        """
        将 12×64 触觉矩阵转换为 3×24×32 的 RGB 彩色图。
        触觉数据原始格式: (B, 12, 64)，左右各 32 列分别对应两个触觉传感器。
        通过 viridis 色图将灰度值映射为 RGB 颜色，然后上下拼接为 (B, 3, 24, 32)。
        """
        B = t.size(0)
        l, r = t[..., :32].clamp(0, 1), t[..., 32:].clamp(0, 1)
        li, ri = (l * 255).long().clamp(0, 255), (r * 255).long().clamp(0, 255)
        lc, rc = self.viridis_map[li], self.viridis_map[ri]  # type: ignore
        color = torch.cat([lc.permute(0, 3, 1, 2), rc.permute(0, 3, 1, 2)], 2)
        return self.mask_tactile_color_image(color) if mask else color  # type: ignore

    @torch.no_grad()
    def _cvt_12x64_to_1x24x32(t, _mask=False):
        """
        将 12×64 触觉矩阵转换为 1×24×32 的单通道灰度图。
        左右 (12,32) 拼接为 (24,32) 后增加通道维度。
        """
        left, right = t[..., :32], t[..., 32:]
        return torch.cat([left, right], 1).unsqueeze(1)

    def loss_batch(m, batch):
        """
        单个 batch 的前向传播 + 损失计算。
        
        流程:
          1. 将 RGB 和触觉数据移至 GPU
          2. 模型前向传播得到预测触觉图
          3. 将原始触觉转换为 GT 格式
          4. 计算 MSE 损失
        """
        rgb = batch["rgb_cur"].to(device)
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)  # 补上相机维度 → (B, 1, C, H, W)
        tactile = batch["tactile_cur"].to(device)
        pred = m(rgb, tactile)
        # 根据配置选择 GT 格式（3 通道 RGB 或 1 通道灰度）
        gt = (
            _cvt_12x64_to_3x24x32(tactile, False)
            if cfg.training.predict_channel == 3
            else _cvt_12x64_to_1x24x32(tactile, False)
        )
        return F.mse_loss(pred, gt), pred, gt

    for epoch in range(cfg.training.num_epochs):
        model.train()
        train_loss = 0

        # ---------- 每个 epoch 的训练 ----------
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} – Train"):
            optimizer.zero_grad()
            loss, pred_c, gt_c = loss_batch(model, batch)
            loss.backward()
            optimizer.step()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()  # 每步更新学习率

            # 如果启用 EMA，每步用当前模型参数更新 EMA 模型
            if cfg.training.ema_pretrain:
                ema_model.update_parameters(model)

            train_loss += loss.item()
            global_step += 1
            wandb.log({"train_loss": loss.item(), "epoch": epoch}, step=global_step)

            # ---------- 每 500 步进行一次 debug 评估和可视化 ----------
            if global_step % 500 == 0:
                # 使用 EMA 模型（如果有的话）进行评估
                tgt = ema_model if cfg.training.ema_pretrain else model
                # 分别在训练集和验证集上评估（各取 3 个 batch）
                tl, tb, tp, _ = eval_autoencoder(tgt, train_loader, device, 3, True, False)
                vl, vb, vp, _ = eval_autoencoder(tgt, val_loader, device, 3, True, True)
                wandb.log({"debug/train_loss": tl, "debug/val_loss": vl}, step=global_step)
                # 保存训练/验证预测对比图
                debug_plot_train_val(tb, tp, vb, vp, os.path.join(run_dir, "debug_plots"), global_step)

                # ---------- 可视化 ViT 自注意力 ----------
                tgtm = tgt.module if isinstance(tgt, AveragedModel) else tgt
                # 取最后一个 block 的 self-attention，CLS token 对所有 patch 的注意力权重
                attn = tgtm.clip_encoder.blocks[-1].attn.latest_attn
                if attn is not None:
                    # attn[0, :, 0, 1:] → batch 0, 所有 head, CLS token (idx=0), 对 patch tokens (idx=1:)
                    heat = attention_to_heatmap(attn[0, :, 0, 1:], 16, 224)
                    plot_attention_overlay(
                        tb["rgb_cur"][0],
                        heat,
                        os.path.join(run_dir, "debug_attention", f"vit_self_attn_{global_step}.png"),
                    )

                # ---------- 可视化 patch cross-attention ----------
                if cfg.training.attention_type == "patch":
                    # tactile→image 跨注意力权重
                    ca = tgtm.cross_attn.attn_t2i.latest_attn
                    if ca is not None:
                        heat = attention_to_heatmap(ca[0, 0], 16, 224)
                        plot_attention_overlay(
                            tb["rgb_cur"][0],
                            heat,
                            os.path.join(run_dir, "debug_attention", f"patch_cross_attn_{global_step}.png"),
                        )

        # ---------- epoch 结束：打印训练损失 ----------
        avg_train = train_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch} – Train MSE = {avg_train:.6f}")

        # ---------- 每个 epoch 结束后在完整验证集上评估 ----------
        tgt = ema_model if cfg.training.ema_pretrain else model
        tgt.eval()
        val_loss = 0
        with torch.no_grad():
            for vb in val_loader:
                vl, *_ = loss_batch(tgt, vb)
                val_loss += vl.item()
        avg_val = val_loss / max(len(val_loader), 1)
        print(f"Epoch {epoch} – Val MSE = {avg_val:.6f}")
        wandb.log({"val_loss": avg_val, "epoch": epoch, "lr": scheduler.get_last_lr()[0]}, step=global_step)

        # ---------- 保存 checkpoint ----------
        ckpt = os.path.join(run_dir, f"epoch_{epoch:04d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict() if cfg.training.ema_pretrain else None,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt,
        )
        print(f"[checkpoint] Saved → {ckpt}")


if __name__ == "__main__":
    main()
