"""
触觉自编码器模型（TactileAutoencoderModel）
=============================================

本文件实现了一个视觉-触觉跨模态的 Masked Autoencoder 模型。

整体数据流：

  RGB 图像 (B, 1, 3, 224, 224)          触觉矩阵 (B, 12, 64)
         │                                       │
         ▼                                       ▼
  ┌──────────────┐                    ┌─────────────────────┐
  │ CLIP ViT-B/16│                    │ 12×64 → 3×24×32    │  (viridis 色图映射)
  │  (预训练)     │                    │ + 随机 mask patches │  (MAE 策略)
  └──────┬───────┘                    └─────────┬───────────┘
         │                                       │
    CLS + 196 patch tokens                       ▼
    (B, 197, 768)                       ┌──────────────┐
         │                              │  SimpleCNN   │
         │                              │ 3×24×32→768  │
         │                              └──────┬───────┘
         │                                     │
         ▼                                     ▼
  ┌────────────────────────────────────────────────┐
  │           CrossAttentionBlock                  │
  │  image tokens ←→ tactile tokens 双向交互        │
  └──────────────────────┬─────────────────────────┘
                         │
                    融合特征 (B, 2×768)
                         │
                         ▼
                  ┌──────────────┐
                  │   Decoder    │
                  │ MLP → Sigmoid│
                  └──────┬───────┘
                         │
                    重建触觉 (B, C, 24, 32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import os
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import datetime


def _build_corner_L_indices(grid: int, device) -> torch.Tensor:
    """
    构建 ViT patch 网格四个角落的 L 形 patch 索引（共 12 个），
    用于 patch 模式的 cross-attention 中屏蔽角落区域。

    在 grid×grid 的 patch 网格中，四个角落各选 3 个 patch，
    构成 L 形（角落 + 相邻两个方向各一个）：

    以 14×14 网格为例：
        ┌──┬──┬─ ─ ─ ─ ─ ─ ─┬──┬──┐
        │TL│→ │             │← │TR│   TL: 左上角 L 形
        ├──┤  │             │  ├──┤   TR: 右上角 L 形
        │↓ │  │             │  │↓ │   BL: 左下角 L 形
        ├──┤  │             │  ├──┤   BR: 右下角 L 形
        │  │  │    中间      │  │  │
        │  │  │    区域      │  │  │
        ├──┤  │             │  ├──┤
        │↑ │  │             │  │↑ │
        ├──┤  │             │  ├──┤
        │BL│→ │             │← │BR│
        └──┴──┴─ ─ ─ ─ ─ ─ ─┴──┴──┘

    这些角落 patch 通常对应图像边缘/背景区域，
    在 cross-attention 中被屏蔽以聚焦于有意义的中心区域。

    参数:
        grid: patch 网格边长（例如 ViT-B/16 对 224×224 图像产生 14×14 网格，grid=14）
        device: 张量所在设备

    返回:
        (12,) 的索引张量，0-based，不含 CLS token 的偏移
    """
    g = grid
    idxs = [
        # 左上角 (Top-Left): 角落 patch, 右邻, 下邻
        0, 1, g,
        # 右上角 (Top-Right): 角落 patch, 左邻, 下邻
        g - 1, g - 2, 2*g - 1,
        # 左下角 (Bottom-Left): 角落 patch, 右邻, 上邻
        g*(g - 1), g*(g - 1) + 1, g*(g - 2),
        # 右下角 (Bottom-Right): 角落 patch, 左邻, 上邻
        g*g - 1, g*g - 2, g*(g - 1) - 1,
    ]
    return torch.tensor(idxs, device=device)


class CrossAttentionBlock(nn.Module):
    """
    视觉-触觉跨模态注意力模块。

    支持两种交互模式：
      - "cls" 模式：触觉 token 只与图像的 CLS token 交互（轻量，全局语义）
      - "patch" 模式：触觉 token 与所有 patch token 交互（重量，保留空间细节）

    每种模式都包含双向注意力：
      1. tactile → image (t2i)：触觉从视觉中汲取信息
      2. image → tactile (i2t)：视觉从触觉中汲取信息
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.0, attention_type='cls'):
        super().__init__()
        print(f"Using {attention_type} cross attention")
        # t2i: 触觉作为 Query 关注图像 (tactile-to-image)
        self.attn_t2i = nn.MultiheadAttention(embed_dim, num_heads, 
                                              dropout=dropout, batch_first=True)
        # i2t: 图像作为 Query 关注触觉 (image-to-tactile)
        self.attn_i2t = nn.MultiheadAttention(embed_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        self.ln_tact  = nn.LayerNorm(embed_dim)  # 触觉 token 的 LayerNorm
        self.ln_img   = nn.LayerNorm(embed_dim)  # 图像 token 的 LayerNorm
        self.attention_type = attention_type

    def forward(self, image_tokens, tactile_tokens):
        """
        参数:
            image_tokens:   (B, 1+P, D) — 来自 ViT 的 CLS token + P 个 patch token
            tactile_tokens: (B, Q, D)   — 来自 CNN 的触觉嵌入（Q 通常 = 1）

        返回:
            image_tokens:   更新后的图像 tokens
            tactile_tokens: 更新后的触觉 tokens
            attn_weights:   t2i 注意力权重，用于可视化
        """
        # 将图像 tokens 拆分为 CLS token 和 patch tokens
        cls_tok   = image_tokens[:, :1, :]   # (B, 1, D) — 全局语义 token
        patch_tok = image_tokens[:, 1:, :]   # (B, P, D) — P=196 个空间 patch token

        if self.attention_type == "cls":
            # ============ CLS 模式 ============
            # 只用 CLS token（全局摘要）参与跨模态交互，计算量小

            # 步骤 1: 触觉 → CLS (tactile-to-image)
            # 触觉 token 作为 Query，从 CLS token 的全局语义中提取视觉信息
            tact_out, attn_weights = self.attn_t2i(
                query  = tactile_tokens,   # (B, Q, D) — 触觉作为查询
                key    = cls_tok,          # (B, 1, D) — CLS 作为键
                value  = cls_tok,          # (B, 1, D) — CLS 作为值
                need_weights = True,
                average_attn_weights = False
            )
            # 残差连接 + LayerNorm
            tactile_tokens = self.ln_tact(tactile_tokens + tact_out)

            # 步骤 2: CLS → 触觉 (image-to-tactile)
            # CLS token 作为 Query，从触觉 token 中汲取触觉信息，使 CLS 融合触觉语义
            img_out, _ = self.attn_i2t(
                query  = cls_tok,         # (B, 1, D)
                key    = tactile_tokens,  # (B, Q, D)
                value  = tactile_tokens,  # (B, Q, D)
            )
            cls_tok = self.ln_img(cls_tok + img_out)

            # 步骤 3: 重新拼合 — 只有 CLS 被更新，patch tokens 保持不变
            image_tokens = torch.cat([cls_tok, patch_tok], dim=1)

        elif self.attention_type == "patch":
            # ============ Patch 模式 ============
            # 触觉 token 与所有 patch token 交互，保留空间细节

            B, N, D = image_tokens.shape  # N = 1 (CLS) + P (patches)
            P = N - 1
            grid = int(P ** 0.5)  # patch 网格边长，例如 196 → 14

            # 构建角落 mask：屏蔽四角 L 形的 12 个 patch（通常是背景区域）
            corner_L_patch = _build_corner_L_indices(grid, image_tokens.device)

            # key_padding_mask: True 的位置会被忽略（不参与注意力计算）
            mask = torch.zeros(B, P, dtype=torch.bool, device=image_tokens.device)
            mask[:, corner_L_patch] = True  # 屏蔽角落 patch

            # 步骤 1: 触觉 → patch tokens (tactile-to-image)
            # 触觉从所有非角落 patch 中提取空间细节信息
            tact_out, attn_weights = self.attn_t2i(
                query  = tactile_tokens,  # (B, Q, D)
                key    = patch_tok,       # (B, P, D)
                value  = patch_tok,       # (B, P, D)
                need_weights         = True,
                average_attn_weights = False,
                key_padding_mask=mask   # 屏蔽角落 patch
            )
            tactile_tokens = self.ln_tact(tactile_tokens + tact_out)

            # 步骤 2: patch tokens → 触觉 (image-to-tactile)
            # 每个 patch 从触觉信息中学习空间对应关系
            patch_out, _ = self.attn_i2t(
                query  = patch_tok,       # (B, P, D)
                key    = tactile_tokens,  # (B, Q, D)
                value  = tactile_tokens,  # (B, Q, D)
                need_weights         = True,
                average_attn_weights = False,
            )
            patch_tok = self.ln_img(patch_tok + patch_out)

            # 重新拼合 — CLS 不变，patch tokens 被更新
            image_tokens = torch.cat([cls_tok, patch_tok], dim=1)

        return image_tokens, tactile_tokens, attn_weights

class PatchAggregator(nn.Module):
    """
    可学习的注意力池化模块，将多个 patch 嵌入聚合为单个向量。

    原理：使用一个可学习的 "聚合 token" 作为 Query，
         对所有 patch embeddings 做 cross-attention，
         自适应地加权聚合最相关的 patch 信息。

    仅在 patch 模式下使用（因为 CLS 模式直接用 CLS token 作为全局表示）。

    对比其他池化方式：
      - 平均池化 (mean pooling): 等权重，不区分重要性
      - 最大池化 (max pooling): 只取最大值，丢失其他信息
      - 注意力池化 (本方法):    学习权重，聚焦最相关的 patch
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # 可学习的聚合 token，会通过训练学到如何从 patch 中提取关键信息
        self.query_tok = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query_tok, std=0.02)

        self.attn   = nn.MultiheadAttention(embed_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.ln     = nn.LayerNorm(embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        参数:
            patches: (B, P, D) — P 个 patch 嵌入

        返回:
            (B, D) — 聚合后的单个向量
        """
        B = patches.size(0)
        # 将聚合 token 复制 B 份
        q = self.query_tok.expand(B, -1, -1)           # (B, 1, D)
        # 聚合 token 作为 Query 关注所有 patch（Key, Value）
        pooled, _ = self.attn(query=q, key=patches, value=patches)
        pooled = self.ln(pooled + q)                   # 残差连接 + LayerNorm
        return pooled.squeeze(1)                       # (B, 1, D) → (B, D)


class SimpleCNN(nn.Module):
    """
    轻量级 CNN，用于将被 mask 的触觉彩色图编码为嵌入向量。

    输入:  (B, 3, 24, 32) — 经过 viridis 色图映射 + mask 后的触觉 RGB 图
    输出:  (B, out_dim)   — 触觉嵌入向量

    网络结构：3 层卷积（stride=2 逐步下采样）+ 全局平均池化 + 全连接层
    每层特征图尺寸变化：
      (3, 24, 32) → (16, 12, 16) → (32, 6, 8) → (64, 3, 4) → (64, 1, 1) → out_dim
    """
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 3, 24, 32) → (B, 16, 12, 16)
        x = F.relu(self.bn2(self.conv2(x)))  # → (B, 32, 6, 8)
        x = F.relu(self.bn3(self.conv3(x)))  # → (B, 64, 3, 4)
        x = self.pool(x)                     # → (B, 64, 1, 1)  全局平均池化
        x = x.view(x.size(0), -1)            # → (B, 64)        展平
        x = self.fc(x)                       # → (B, out_dim)   投影到嵌入空间
        return x


def make_viridis_colormap():
    """
    生成 viridis 色图查找表 (LUT)。

    viridis 是 matplotlib 的默认色图，将标量值映射为感知均匀的 RGB 颜色。
    返回 (256, 3) 的浮点张量，索引 0-255 对应归一化 RGB 值。

    用途：将单通道触觉灰度值（0-255）映射为 3 通道 RGB 彩色图，
         使触觉数据可以作为"图像"输入 CNN。
    """
    from matplotlib import cm
    viridis = cm.get_cmap('viridis', 256)
    colormap = viridis(np.arange(256))[:, :3]  # (256, 3)，丢弃 alpha 通道
    return torch.FloatTensor(colormap)


###############################################################################
# TactileAutoencoderModel — 核心模型
###############################################################################
class TactileAutoencoderModel(nn.Module):
    """
    触觉 Masked Autoencoder 模型。

    训练目标：从被随机 mask 的触觉输入 + RGB 视觉信息中，重建完整的触觉数据。
    通过这个预训练任务，模型学会了视觉-触觉之间的跨模态对应关系。

    模块组成：
      1. clip_encoder  — CLIP ViT-B/16 视觉编码器（预训练权重，微调）
      2. cnn           — SimpleCNN 触觉编码器（将 masked 触觉彩色图编码为向量）
      3. cross_attn    — CrossAttentionBlock（视觉-触觉双向注意力融合）
      4. decoder       — MLP 解码器（从融合特征重建触觉数据）
      5. patch_pool    — PatchAggregator（仅 patch 模式使用）

    参数:
        embed_dim:          Transformer 嵌入维度，默认 768（与 ViT-B 一致）
        tactile_patch_size: 触觉 mask 的 patch 大小，默认 4
        num_heads:          多头注意力的头数
        attention_type:     "cls"（仅 CLS 交互）或 "patch"（全 patch 交互）
        predict_channel:    预测通道数，3 = RGB 彩色 或 1 = 灰度
        mask_ratio_min/max: MAE mask 比例范围，默认 60%-80%
        unmask_prob:        以此概率完全不做 mask（让模型偶尔看到完整输入）
    """
    def __init__(
        self,
        embed_dim: int = 768,
        tactile_patch_size: int = 4,
        num_heads: int = 8,
        attention_type: str = "cls",
        predict_channel=3,
        mask_ratio_min: float = 0.6,
        mask_ratio_max: float = 0.8,
        unmask_prob: float = 0.05,
        *,
        save_images: bool = False,
        save_dir: str = "saved_tactile_imgs"
    ):
        super().__init__()

        # ===== 1. 视觉编码器：CLIP ViT-B/16 =====
        # 使用 OpenAI 预训练的 CLIP ViT，输出 CLS token + 196 个 patch token
        # global_pool='' 表示不做全局池化，保留所有 token
        # num_classes=0 表示不加分类头
        self.clip_encoder = timm.create_model(
            "vit_base_patch16_clip_224.openai",
            pretrained=True,
            global_pool='',
            num_classes=0
        )
        # 解冻 CLIP 参数，允许微调
        for p in self.clip_encoder.parameters():
            p.requires_grad = True

        # ===== 2. Viridis 色图查找表 =====
        # 注册为 buffer（非参数），不参与梯度计算，保存时不写入 state_dict
        self.register_buffer("viridis_map", make_viridis_colormap(), persistent=False)

        # ===== 3. 触觉 CNN 编码器 =====
        # 将 masked 触觉彩色图 (3, 24, 32) 编码为 embed_dim 维向量
        self.cnn = SimpleCNN(out_dim=embed_dim)

        self.patch_size  = tactile_patch_size

        # ===== 4. 可学习 mask token =====
        # 被 mask 的触觉 patch 位置会被替换为这个可学习的 token
        # 形状 (1, 3, patch_size, patch_size)，可广播到任意 batch
        self.mask_token = nn.Parameter(torch.zeros(1, 3, self.patch_size, self.patch_size))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        # ===== 5. 图像 patch 的位置编码 =====
        # ViT-B/16 对 224×224 图像产生 14×14 = 196 个 patch
        grid_size   = 224 // 16   # = 14
        num_patches = grid_size * grid_size  # = 196
        self.pos_embed = nn.Embedding(num_patches, embed_dim)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)

        # ===== 6. 跨模态注意力 =====
        self.attention_type = attention_type
        self.cross_attn = CrossAttentionBlock(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=0.2, attention_type=self.attention_type
        )

        # ===== 7. 解码器 =====
        self.predict_channel = predict_channel
        # 输出维度：3×24×32 = 2304（RGB）或 1×24×32 = 768（灰度）
        if self.predict_channel == 3:
            self.decoder_out_dim = 3 * 24 * 32  # 2304
        else:
            self.decoder_out_dim = 1 * 24 * 32  # 768

        hidden_dim = embed_dim
        # 两层 MLP 解码器：2D → D → out_dim
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # 融合特征 (2×768=1536) → 768
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, self.decoder_out_dim),  # 768 → 2304 或 768
        )
        # 最后一层 bias 初始化为 0，使初始输出接近 0（经过 Sigmoid 后约 0.5）
        nn.init.constant_(self.decoder[-1].bias, 0.0)

        # Sigmoid 输出激活，将预测值限制在 [0, 1]
        self.out_act   = nn.Sigmoid()

        # ===== 8. Debug 图片保存设置 =====
        self.save_images = save_images
        self.save_dir    = save_dir
        if self.save_images:
            os.makedirs(self.save_dir, exist_ok=True)
        self._internal_step = 0

        # ===== 9. Patch 模式专用的聚合模块 =====
        # CLS 模式直接用 CLS token；patch 模式需要将 196 个 patch 聚合为 1 个向量
        if attention_type == "patch":
            self.patch_pool = PatchAggregator(embed_dim, num_heads=num_heads)
        else:
            self.patch_pool = None

        # ===== 10. MAE mask 参数 =====
        self.mask_ratio_min = mask_ratio_min  # 最低 mask 比例（默认 60%）
        self.mask_ratio_max = mask_ratio_max  # 最高 mask 比例（默认 80%）
        self.unmask_prob = unmask_prob         # 完全不 mask 的概率（默认 5%）

    @torch.no_grad()
    def _dump_batch(self, masked_color: torch.Tensor, gt_color: torch.Tensor, step_id: int):
        """Debug 用：将 masked 触觉图和 GT 触觉图保存为 PNG 图片。"""
        masked_color = masked_color.cpu().clamp(0, 1)
        gt_color     = gt_color.cpu().clamp(0, 1)
        for i in range(masked_color.size(0)):
            base = f"step{step_id:06d}_{i}"
            from torchvision.utils import save_image
            save_image(masked_color[i], os.path.join(self.save_dir, f"{base}_masked.png"))
            save_image(gt_color[i],     os.path.join(self.save_dir, f"{base}_gt.png"))
    
    def encode(
        self,
        rgb_cur_unmasked: torch.Tensor,
        tactile_12x64:    torch.Tensor,
        global_step:      int = None
    ) -> torch.Tensor:
        """
        编码流程（训练时使用，触觉数据会被 mask）。

        参数:
            rgb_cur_unmasked: (B, n_cams, 3, 224, 224) — RGB 相机图像
            tactile_12x64:    (B, 12, 64)              — 原始触觉传感器数据
            global_step:      当前训练步数（用于 debug 保存）

        返回:
            fused_in: (B, 2×embed_dim) — 融合后的视觉-触觉特征

        完整流程:
            1. CLIP ViT 编码 RGB → (B, 197, 768)
            2. 添加可学习位置编码到 patch tokens
            3. 触觉 12×64 → viridis 彩色图 3×24×32 → 随机 mask → CNN 编码
            4. CrossAttention 双向融合
            5. 拼接视觉表示 + 触觉表示 → (B, 2D)
        """
        B, n_cams, C, H, W = rgb_cur_unmasked.shape

        # ---- 步骤 1: CLIP ViT 编码 RGB 图像 ----
        # 将 (B, n_cams, C, H, W) 展平为 (B*n_cams, C, H, W) 送入 ViT
        rgb_flat      = rgb_cur_unmasked.view(B * n_cams, C, H, W)
        camera_tokens = self.clip_encoder(rgb_flat)      # (B*n_cams, 197, 768)
        T             = camera_tokens.size(1)             # 197 = 1 CLS + 196 patches
        camera_tokens = camera_tokens.view(B, n_cams * T, -1)  # (B, 197, 768)

        # ---- 步骤 2: 给 patch tokens 加上可学习的位置编码 ----
        cls_tok   = camera_tokens[:, :1]    # (B, 1, D)   CLS token 不加位置编码
        patch_tok = camera_tokens[:, 1:]    # (B, 196, D) patch tokens
        pos_ids   = torch.arange(patch_tok.size(1), device=patch_tok.device)
        patch_tok = patch_tok + self.pos_embed(pos_ids).unsqueeze(0)  # 加位置编码
        camera_tokens = torch.cat([cls_tok, patch_tok], dim=1)  # (B, 197, D)

        # ---- 步骤 3: 触觉编码（带 mask） ----
        # 将 12×64 触觉矩阵转为 3×24×32 viridis 彩色图
        gt_color     = self.convert_12x64_to_3x24x32(tactile_12x64, apply_mask=False)   # 完整 GT
        masked_color = self.convert_12x64_to_3x24x32(tactile_12x64, apply_mask=True)    # 随机 mask 后
        # CNN 编码 masked 触觉图 → (B, D) → 增加 token 维度 → (B, 1, D)
        tactile_emb    = self.cnn(masked_color)       # (B, embed_dim)
        tactile_tokens = tactile_emb.unsqueeze(1)     # (B, 1, embed_dim)

        # ---- 步骤 4: 跨模态注意力融合 ----
        camera_tokens, tactile_tokens, _ = self.cross_attn(camera_tokens, tactile_tokens)

        # ---- 步骤 5: 拼接视觉 + 触觉表示 ----
        if self.attention_type == "cls":
            # CLS 模式：用更新后的 CLS token 代表图像
            image_cls = camera_tokens[:, :1, :].squeeze(1)  # (B, D)
            tactile_  = tactile_tokens.squeeze(1)           # (B, D)
            fused_in  = torch.cat([tactile_, image_cls], dim=1)  # (B, 2D)
        else:
            # Patch 模式：用 PatchAggregator 将 196 个 patch 池化为 1 个向量
            patch_tok = camera_tokens[:, 1:, :]        # (B, 196, D)
            patch_rep = self.patch_pool(patch_tok)     # (B, D) ← 注意力池化
            tactile_  = tactile_tokens.squeeze(1)      # (B, D)
            fused_in  = torch.cat([tactile_, patch_rep], dim=1)  # (B, 2D)

        return fused_in

    def encode_unmasked(
        self,
        rgb_cur_unmasked: torch.Tensor,
        tactile_12x64:    torch.Tensor,
        global_step:      int = None
    ) -> torch.Tensor:
        """
        不带 mask 的编码流程（推理/下游任务使用）。

        与 encode() 的唯一区别：触觉数据不做 mask，直接用完整触觉图编码。
        用于预训练完成后，将该编码器迁移到下游任务时提取完整的视觉-触觉特征。

        参数和返回值与 encode() 相同。
        """
        B, n_cams, C, H, W = rgb_cur_unmasked.shape

        # ---- CLIP ViT 编码 RGB ----
        rgb_flat      = rgb_cur_unmasked.view(B * n_cams, C, H, W)
        camera_tokens = self.clip_encoder(rgb_flat)
        T             = camera_tokens.size(1)
        camera_tokens = camera_tokens.view(B, n_cams * T, -1)

        # ---- 位置编码 ----
        cls_tok   = camera_tokens[:, :1]
        patch_tok = camera_tokens[:, 1:]
        pos_ids   = torch.arange(patch_tok.size(1), device=patch_tok.device)
        patch_tok = patch_tok + self.pos_embed(pos_ids).unsqueeze(0)
        camera_tokens = torch.cat([cls_tok, patch_tok], dim=1)

        # ---- 触觉编码（不做 mask！两者都用 apply_mask=False） ----
        gt_color     = self.convert_12x64_to_3x24x32(tactile_12x64, apply_mask=False)
        masked_color = self.convert_12x64_to_3x24x32(tactile_12x64, apply_mask=False)

        # ---- CNN 编码 + 跨模态注意力 ----
        tactile_emb    = self.cnn(masked_color)
        tactile_tokens = tactile_emb.unsqueeze(1)
        camera_tokens, tactile_tokens, _ = self.cross_attn(camera_tokens, tactile_tokens)

        # ---- 拼接融合特征 ----
        if self.attention_type == "cls":
            image_cls = camera_tokens[:, :1, :].squeeze(1)  # (B, D)
            tactile_  = tactile_tokens.squeeze(1)           # (B, D)
            fused_in  = torch.cat([tactile_, image_cls], dim=1)  # (B, 2D)
        else:
            patch_tok = camera_tokens[:, 1:, :]        # (B, P, D)
            patch_rep = self.patch_pool(patch_tok)     # (B, D)
            tactile_  = tactile_tokens.squeeze(1)      # (B, D)
            fused_in  = torch.cat([tactile_, patch_rep], dim=1)  # (B, 2D)

        return fused_in
            
    def forward(
        self,
        rgb_cur_unmasked: torch.Tensor,
        tactile_12x64:    torch.Tensor,
        global_step:      int = None
    ) -> torch.Tensor:
        """
        完整的前向传播：编码 + 解码。

        参数:
            rgb_cur_unmasked: (B, n_cams, 3, 224, 224) — RGB 图像
            tactile_12x64:    (B, 12, 64)              — 触觉数据

        返回:
            pred: (B, C, 24, 32) — 重建的触觉图，C=3(RGB) 或 C=1(灰度)，值域 [0,1]
        """
        # 步骤 1: 编码 → 融合特征 (B, 2×embed_dim)
        latent = self.encode(rgb_cur_unmasked, tactile_12x64, global_step)

        # 步骤 2: 解码 → 重建触觉图
        x    = self.decoder(latent)  # (B, 2D) → (B, C×24×32)
        pred = x.view(latent.size(0), self.predict_channel, 24, 32)  # reshape 为图像格式
        return self.out_act(pred)    # Sigmoid → [0, 1]

    def convert_12x64_to_3x24x32(self, tactile_map: torch.Tensor, apply_mask: bool) -> torch.Tensor:
        """
        将触觉矩阵 (B, 12, 64) 转换为 RGB 彩色图 (B, 3, 24, 32)。

        触觉传感器原始数据格式：
          - 12 行 × 64 列
          - 前 32 列 = 左传感器，后 32 列 = 右传感器

        转换流程：
          1. 左右拆分：(B, 12, 32) × 2
          2. 归一化到 [0, 1]，乘以 255 得到索引
          3. 通过 viridis 色图查找表将灰度值映射为 RGB 颜色 → (B, 12, 32, 3)
          4. 调整维度为 (B, 3, 12, 32)
          5. 左右上下拼接 → (B, 3, 24, 32)
          6. 如果 apply_mask=True，随机 mask 部分 patch

        参数:
            tactile_map: (B, 12, 64) 原始触觉矩阵
            apply_mask:  是否应用随机 mask（训练时 True，推理时 False）
        """
        B      = tactile_map.size(0)
        # 拆分左右传感器
        left   = tactile_map[..., :32].clamp(0, 1)   # (B, 12, 32)
        right  = tactile_map[..., 32:].clamp(0, 1)   # (B, 12, 32)
        # 转为 0-255 整数索引，用于查找 viridis 色图
        left_i = (left  * 255).long().clamp(0, 255)
        right_i= (right * 255).long().clamp(0, 255)
        # 通过 LUT 查表：(B, 12, 32) → (B, 12, 32, 3)
        left_c = self.viridis_map[left_i]
        right_c= self.viridis_map[right_i]
        # 转为 (B, 3, 12, 32) 的图像格式
        left_c = left_c.permute(0, 3, 1, 2)
        right_c= right_c.permute(0, 3, 1, 2)
        # 上下拼接左右传感器 → (B, 3, 24, 32)
        color_img = torch.cat([left_c, right_c], dim=2)
        if apply_mask:
            color_img = self.mask_tactile_color_image(color_img)
        return color_img

    @torch.no_grad()
    def convert_12x64_to_1x24x32(self, tactile_map, apply_mask=False):
        """
        将触觉矩阵 (B, 12, 64) 转换为单通道灰度图 (B, 1, 24, 32)。

        与 3 通道版本不同，这里不做 viridis 色图映射，直接保留原始灰度值。
        左右传感器 (12, 32) 上下拼接为 (24, 32)。
        """
        B, H, W = tactile_map.shape
        left  = tactile_map[:, : , :32]   # (B, 12, 32) — 左传感器
        right = tactile_map[:, : , 32:]   # (B, 12, 32) — 右传感器
        # 上下拼接
        stacked = torch.cat([left, right], dim=1)  # (B, 24, 32)
        return stacked.unsqueeze(1)  # (B, 1, 24, 32) — 增加通道维度

    def mask_tactile_color_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        MAE 核心：对触觉彩色图进行随机 patch mask。

        参数:
            img: (B, 3, 24, 32) — 触觉 viridis 彩色图

        流程:
          1. 以 unmask_prob（默认 5%）的概率完全不做 mask，直接返回原图
             （这让模型偶尔看到完整输入，避免过度依赖视觉补全）
          2. 将图像划分为 patch_size × patch_size 的非重叠 patch 网格
             例如 patch_size=4: (24, 32) → 6×8 = 48 个 patch
          3. 随机选择 60%-80% 的 patch 替换为可学习的 mask_token
          4. 每个 batch 样本独立随机 mask

        返回:
            被 mask 后的图像，被 mask 的位置被替换为 self.mask_token
        """
        # 以小概率跳过 mask，让模型偶尔看到完整输入
        if random.random() < self.unmask_prob:
            return img

        B, C, H, W = img.shape
        # 计算 patch 网格大小
        pH         = H // self.patch_size  # 高度方向的 patch 数，例如 24//4 = 6
        pW         = W // self.patch_size  # 宽度方向的 patch 数，例如 32//4 = 8
        total      = pH * pW              # 总 patch 数，例如 48

        # 随机确定 mask 比例和数量
        ratio    = random.uniform(self.mask_ratio_min, self.mask_ratio_max)  # 例如 [0.6, 0.8]
        num_mask = int(total * ratio)  # 要 mask 的 patch 数

        # 对 batch 中每个样本独立 mask
        for b in range(B):
            idxs = list(range(total))
            np.random.shuffle(idxs)  # 随机打乱 patch 索引
            for pid in idxs[:num_mask]:
                # 将 1D patch 索引转为 2D 像素坐标
                r0 = (pid // pW) * self.patch_size  # patch 左上角的行坐标
                c0 = (pid % pW) * self.patch_size   # patch 左上角的列坐标
                # 用可学习的 mask_token 替换该 patch 区域
                img[b, :, r0:r0 + self.patch_size, c0:c0 + self.patch_size] = self.mask_token
        return img
