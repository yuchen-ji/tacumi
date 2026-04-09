"""
Modified by yuchen, 26.03.28
添加了处理触觉数据的代码，并且支持仅视觉（触觉缺失）的代码
通过 use_tactile 来确定是否使用触觉数据
"""

import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tran
from pretrain_mae.pretrain_mae_model import TactileAutoencoderModel

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)

def make_viridis_colormap():
    """
    Returns a (256, 3) float Tensor in [0,1], which is
    the standard 'viridis' colormap from matplotlib.
    """
    import matplotlib
    from matplotlib import cm
    viridis = cm.get_cmap('viridis', 256)
    colormap = viridis(np.arange(256))[:, :3]  
    colormap = torch.FloatTensor(colormap)
    return colormap

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class SimpleCNN(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        
        # AdaptiveAvgPool2d squeezes the feature map to size (1x1),
        # making the feature dimension = 64
        self.pool  = nn.AdaptiveAvgPool2d((1,1))

        self.fc    = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.relu(self.bn3(self.conv3(x))) 
        x = self.pool(x)                     
        x = x.view(x.size(0), -1)            
        x = self.fc(x)                       
        return x


class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            use_tactile: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',

            tactile_model_choice: str='simple_cnn', 
            load_pretrain_ckpt: bool = True,
            pretrain_ckpt_path: str = None,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        self.use_tactile = use_tactile
        self.tactile_model_choice = tactile_model_choice
        print("use_tactile",self.use_tactile)
        if self.use_tactile:
            print("tactile_model_choice", self.tactile_model_choice)
        
        rgb_keys = list()
        low_dim_keys = list()
        tactile_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''
        if not (self.use_tactile and self.tactile_model_choice == 'pretrain'):
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool, # '' means no pooling
                num_classes=0            # remove classification layer
            )
        if self.use_tactile:
            if self.tactile_model_choice == 'simple_cnn':
                tactile_model = SimpleCNN(out_dim=512)
            elif self.tactile_model_choice == 'pretrain':
                self.pretrained_tactile_model = TactileAutoencoderModel(
                    embed_dim=768,
                    tactile_patch_size=4,
                    num_heads=8,
                    attention_type="cls"
                )
                # remove the decoder
                del self.pretrained_tactile_model.decoder

                if load_pretrain_ckpt and pretrain_ckpt_path is not None:
                    if os.path.exists(pretrain_ckpt_path):
                        ckpt_path = pretrain_ckpt_path
                        print(f"[TimmObsEncoder] Attempting to load TactileAutoencoderModel from {ckpt_path}")
                        ckpt = torch.load(ckpt_path, map_location="cpu")

                        raw = ckpt["ema_state_dict"]
                        # 1) Drop the EMA bookkeeping
                        filtered = {k: v for k, v in raw.items() if k != "n_averaged"}
                        # 2) Strip off any "module." prefixes
                        stripped = {
                            (k[len("module."):] if k.startswith("module.") else k): v
                            for k, v in filtered.items()
                        }
                        # 3) Remove any decoder keys
                        clean = {
                            k: v for k, v in stripped.items()
                            if not k.startswith("decoder.")
                        }
                        # 4) Load
                        self.pretrained_tactile_model.load_state_dict(clean, strict=True)
                        print("[TimmObsEncoder] TactileAutoencoderModel weights loaded successfully!")
                    else:
                        print("[TimmObsEncoder] Skipping pretrain loading (will use trained weights from main checkpoint during inference)")
                else:
                    print("[TimmObsEncoder] Skipping pretrain checkpoint loading (load_pretrain_ckpt=False or path not set)")

                self.pretrained_tactile_model.train()
                for p in self.pretrained_tactile_model.parameters():
                    p.requires_grad = True
                
            else:
                raise ValueError(f"Unknown tactile_model_choice: {tactile_model_choice}")

        if frozen:
            print("CLIP weight frozen")
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                if not (self.use_tactile and self.tactile_model_choice == 'pretrain'):
                    this_model = model if share_rgb_model else copy.deepcopy(model)
                    key_model_map[key] = this_model

                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            elif type == 'tactile':
                if self.use_tactile:
                    tactile_keys.append(key)
                    if self.use_tactile and self.tactile_model_choice == "pretrain":
                        this_tactile_model = self.pretrained_tactile_model
                    else: 
                        this_tactile_model = tactile_model
                    key_model_map[key] = this_tactile_model
                    key_transform_map[key] = transform
                else:
                    continue
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.tactile_keys = tactile_keys
        self.key_shape_map = key_shape_map
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation == 'all_tokens':
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation is not None:
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = None
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.position_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4)
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        if self.use_tactile:
            # We store it as a buffer so it moves automatically with .to(device)
            self.register_buffer(
                "viridis_map",
                make_viridis_colormap(),  # shape [256, 3], in [0,1]
                persistent=False
            )
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        if self.model_name.startswith('vit'):
            assert self.feature_aggregation is None # vit uses the CLS token
            return feature[:, 0, :]
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature
        
    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        self._latent = None

        if not (self.use_tactile and self.tactile_model_choice == 'pretrain'):
            # process rgb input
            for key in self.rgb_keys:
                img = obs_dict[key]
                B, T = img.shape[:2]
                assert B == batch_size
                assert img.shape[2:] == self.key_shape_map[key]
                img = img.reshape(B*T, *img.shape[2:])
                img = self.key_transform_map[key](img)
                raw_feature = self.key_model_map[key](img)
                feature = self.aggregate_feature(raw_feature)
                # ### DEBUG PRINTS ###
                # print(f"[Encoder debug] RGB key={key}, raw_feature.shape={raw_feature.shape}, aggregated={feature.shape}")
                assert len(feature.shape) == 2 and feature.shape[0] == B * T
                features.append(feature.reshape(B, -1))
                # print(f"[Encoder debug] Appending shape={feature.reshape(B, -1).shape} for key={key}")

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
            # print(f"[Encoder debug] Low‐dim key={key}, shape={data.reshape(B, -1).shape}")

        if self.use_tactile:
            if self.tactile_model_choice == 'simple_cnn':
                for key in self.tactile_keys:
                    tactile_data = obs_dict[key]
                    B, T = tactile_data.shape[:2]
                    assert B == batch_size
                    # Reshape to (B*T, 12, 64)
                    tactile_data = tactile_data.reshape(B * T, *tactile_data.shape[2:])
                    left_tactile = tactile_data[:, :, :32]  # shape: (B*T, 12, 32)
                    right_tactile = tactile_data[:, :, 32:]  # shape: (B*T, 12, 32)

                    # ---- GPU-based color mapping with our viridis buffer ----
                    left_tactile = left_tactile.clamp(0, 1)
                    right_tactile = right_tactile.clamp(0, 1)
                    left_index = (left_tactile * 255).long().clamp(0, 255)
                    right_index = (right_tactile * 255).long().clamp(0, 255)

                    left_color = self.viridis_map[left_index]
                    right_color = self.viridis_map[right_index]
                    tactile_images = torch.cat([left_color, right_color], dim=1)
                    tactile_images = tactile_images.permute(0, 3, 1, 2)

                    device = next(self.key_model_map[key].parameters()).device
                    tactile_images = tactile_images.to(device)

                    # Process through the tactile model (SimpleCNN)
                    feature = self.key_model_map[key](tactile_images)

                    assert len(feature.shape) == 2 and feature.shape[0] == B * T
                    # print(f"[Encoder debug] Tactile key={key}, raw_feature.shape={raw_feature.shape}, final_feat.shape={feature.shape}")
                    feat_ = feature.reshape(B, -1)
                    # print(f"[Encoder debug] Appending shape={feat_.shape} for tactile={key}")
                    features.append(feat_)
            elif self.tactile_model_choice == "pretrain":
                # Pretrain model path
                if len(self.rgb_keys) == 0:
                    raise ValueError("Expected at least 1 rgb key to pass into the pretrained TactileAutoencoderModel!")
                if len(self.tactile_keys) == 0:
                    raise ValueError("Expected at least 1 tactile key with 'pretrain' mode!")

                cam_key = self.rgb_keys[0]
                tact_key = self.tactile_keys[0]

                # shapes
                img = obs_dict[cam_key]  # (B, T, 3, H, W)
                tactile_data = obs_dict[tact_key]  # (B, T, 12, 64)

                # We flatten batch+time => (B*T, 3, H, W) and (B*T, 12, 64)
                BT = batch_size * T
                img = img.reshape(BT, *img.shape[2:])  # (BT, 3, H, W)
                img = self.key_transform_map[cam_key](img)
                tactile_data = tactile_data.reshape(BT, 12, 64)

                # The pretrained autoencoder forward expects shape (B, nCams, 3, 224, 224)
                img = img.unsqueeze(1)  # => (BT, 1, 3, H, W)

                # Now call pretrained
                with torch.set_grad_enabled(self.pretrained_tactile_model.training):
                    latent = self.pretrained_tactile_model.encode_unmasked(img, tactile_data)

                BT, D = latent.shape
                assert BT % batch_size == 0, (
                    f"Got BT={BT} from hook, but batch_size={batch_size}"
                )
                T = BT // batch_size 
                feat_ = latent.view(batch_size, T * D)
                # print(f"[Encoder debug] cross_latent.shape={latent.shape}, after reshape={feat_.shape}")
                features.append(feat_)

        # concatenate all features
        result = torch.cat(features, dim=-1)

        return result

    

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape


if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )